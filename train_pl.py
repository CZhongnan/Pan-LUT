import torchvision
import time
import json
import shutil
from copy import deepcopy
from pickle import FALSE
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
import os
import argparse
import numpy as np
from Datasets.datasets import *
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from utils.helpers import initialize_weights_new
from pytorch_lightning.loggers.wandb import WandbLogger
from compute_loss import Compute_loss
from models.models import MODELS
from utils.metrics_inference import *
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint,DeviceStatsMonitor,EarlyStopping,LearningRateMonitor
from utils.ema import EMA as EMACallback
from utils.optimizer import Lion
import collections
from utils.perceptual import PerceptualLoss2
import os
from scheduler import CosineAnnealingRestartLR
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
# select dataset
__dataset__ = {
               "wv3_dataset":Data,
               "GF2_dataset":Data,
               "wv2_dataset":Data,
               "GF2_dataset_h5":GF2_dataset,
               "wv3_dataset_h5":wv3_dataset,
               "qb_dataset_h5":qb_dataset,
               "tw_dataset":Data,
               "vif_dataset":VIF_Data,
               }
class FFTLoss(nn.Module):
    def __init__(self, loss_f=nn.L1Loss):
        super(FFTLoss, self).__init__()
        self.loss = loss_f()

    def forward(self, pred, target):
        # 图像通常是实数，所以使用 rfft2 (real FFT 2D)
        # (N, C, H, W)
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1), norm='ortho')
        target_fft = torch.fft.rfft2(target, dim=(-2, -1), norm='ortho')
        
        # 计算幅度谱
        pred_amp = torch.abs(pred_fft)
        target_amp = torch.abs(target_fft)
        
        # 计算幅度谱的损失
        return self.loss(pred_amp, target_amp)
class CoolSystem(pl.LightningModule):
    def __init__(self):
        """初始化训练的参数"""
        super(CoolSystem, self).__init__()
        # train datasets
        self.train_datasets = __dataset__[config["train_dataset"]](
                            config,
                            is_train=True,
                        )
        self.train_batchsize = config["train_batch_size"]
        # val datasets
        self.validation_datasets = __dataset__[config["train_dataset"]](
                            config,
                            is_train=False,
                        )
        self.val_batchsize = config["val_batch_size"]
        self.num_workers = config["num_workers"]
        # self.loss_per = PerceptualLoss2()
        # self.loss_F = nn.L1Loss()
        # set mode stype
        self.loss_F = FFTLoss()
        self.model =  MODELS[config["model"]](config)
        # Resume from pth ...
        if args.resume is not None:
            print("Loading from existing FusionNet chekpoint")
            ckpt = torch.load(args.resume)
            new_state_dict = collections.OrderedDict()
            for k in ckpt['state_dict']:
                            # print(k)
                            if k[:6] != 'model.':
                                continue
                            name = k[6:]
                            new_state_dict[name] = ckpt['state_dict'][k]
            self.model.load_state_dict(new_state_dict,strict=True)
            
        print(PATH)
        # print model summary.txt
        import sys
        original_stdout = sys.stdout 
        with open(PATH+"/"+"model_summary.txt", 'w+') as f:
            sys.stdout = f
            print(f'\n{self.model}\n')
            sys.stdout = original_stdout 
        shutil.copy(f'./models/{config["model"]}.py',PATH+"/"+"model.py") 
    def train_dataloader(self):
        train_loader = data.DataLoader(
                        self.train_datasets,
                        batch_size=self.train_batchsize,
                        num_workers=self.num_workers,
                        shuffle=True,
                        # pin_memory=False,
                    )
        return train_loader
    
    def val_dataloader(self):
        val_loader = data.DataLoader(
                        self.validation_datasets,
                        batch_size=self.val_batchsize,
                        num_workers=self.num_workers,
                        shuffle=False,
                        # pin_memory=False,
                    )
        return val_loader




    def configure_optimizers(self):
        """配置优化器和学习率的调整策略"""
        # Setting up optimizer.
        self.initlr =config["optimizer"]["args"]["lr"] #initial learning
        self.weight_decay = config["optimizer"]["args"]["weight_decay"] #optimizers weight decay
        self.momentum = config["optimizer"]["args"]["momentum"]
        if config["optimizer"]["type"] == "SGD":
            optimizer = optim.SGD(
                self.model.parameters(), 
                lr=self.initlr , 
                momentum =self.momentum , 
                weight_decay= self.weight_decay
            )
        elif config["optimizer"]["type"] == "ADAM":
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=self.initlr,
                weight_decay= self.weight_decay,
                betas =  [0.9, 0.999]
            )
        elif config["optimizer"]["type"] == "ADAMW":
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=self.initlr,
                weight_decay= self.weight_decay,
                betas =  [0.9, 0.999]
            )   
        elif config["optimizer"]["type"] == "Lion":
            optimizer = Lion(filter(lambda p: p.requires_grad, self.model.parameters()), 
                             lr=self.initlr,
                             betas=[0.9,0.99],
                             weight_decay=0.01)
            
            
        else:
            exit("Undefined optimizer type")
        
        # Learning rate shedule 
        if config["optimizer"]["sheduler"] == "StepLR":
            step_size=config["optimizer"]["sheduler_set"]["step_size"]
            gamma=config["optimizer"]["sheduler_set"]["gamma"]
            scheduler = optim.lr_scheduler.StepLR(  optimizer, step_size=step_size, gamma=gamma)
        elif config["optimizer"]["sheduler"] == "CyclicLR":
          scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=self.initlr,max_lr=1.2*self.initlr,cycle_momentum=False)
        elif config["optimizer"]["sheduler"] =="CosineAnnealingLR":
          scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["trainer"]["total_epochs"],eta_min =self.initlr* 1e-1)
        elif config["optimizer"]["sheduler"] =="CosineAnnealingRestartLR":
          scheduler = CosineAnnealingRestartLR(optimizer=optimizer, periods=[200,600,800], restart_weights=[1,0.5,0.5],eta_min=1e-7)
        # elif config["optimizer"]["sheduler"] =="CosineAnnealingLR":
        #   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config["trainer"]["total_epochs"],eta_min =self.initlr* 1e-1)
        # Sheduler == None时

        else:
            scheduler = None
        return [optimizer], [scheduler]
    
            
    def forward(self,MS_image, PAN_image,image_dict):
        out = self.model(MS_image,PAN_image,image_dict)
        return out
    
    def training_step(self, data):
        """trainning step"""
        # read data
        image_dict, MS_image,PAN_image, gt = data
        # print(image_dict.shape)
        # taking model output
        
        
        out = self.forward(MS_image, PAN_image,image_dict)      
        # computing loss 
        # pred = torch.clip(out['pred'],min=0,max=1)
        loss = Compute_loss(config,out,PAN_image,gt)
        #outputs = torch.clip(out['pred'],min=0,max=1)

        #loss += self.loss_per(outputs[:,0:3,:,:],gt[:,0:3,:,:]) * 0.1
        # loss += self.FrequencyLoss(outputs,gt)

        self.log('train_loss', loss,sync_dist=True,prog_bar=True)
        self.log('lr',self.trainer.optimizers[0].state_dict()['opt']['param_groups'][0]['lr'],sync_dist=True,prog_bar=True)
        
        return {'loss': loss}
    
    def on_validation_epoch_start(self):
        self.pred_dic={}
        return super().on_validation_epoch_start()

    def validation_step(self, data, batch_idx):
        """validation step"""
        # read data.
        image_dict, MS_image,PAN_image, gt = data
        # print(MS_image.min(),MS_image.max())
        # print(PAN_image.min(),PAN_image.max())
        # print(image_dict.shape)
        # taking model output
        with torch.no_grad():
            out  = self.forward(MS_image,PAN_image,image_dict)   
            pred    = out['pred'];
            # pred = torch.clip(out['pred'],min=0,max=1)
        # computing loss 
            loss = Compute_loss(config,out,PAN_image,gt)
        # computing performance metrics
        max_value = config[config["train_dataset"]]["max_value"]
        #print(max_value)
        if not config[config["train_dataset"]]['normalize']:
            predict_y = torch.clip((pred * max_value),min=0,max=max_value)
            ground_truth = torch.clip((gt * max_value),min=0,max=max_value)
        else:        
            predict_y = torch.clip(((pred+ 1) * max_value/2),min=0,max=max_value)
            ground_truth = torch.clip(((gt+ 1) * max_value/2),min=0,max=max_value)
        # print(ground_truth.min(),ground_truth.max())
        predict_y = np.array(predict_y.cpu().squeeze(0).permute(1,2,0))
        ground_truth = np.array(ground_truth.cpu().squeeze(0).permute(1,2,0))
        
        c_psnr, c_ssim, c_sam, c_ergas, c_scc, c_q = ref_evaluate(predict_y,ground_truth)
        
        self.log('val_loss', loss,sync_dist=True,prog_bar=True)
        self.log('psnr', c_psnr,sync_dist=True, prog_bar=True)
        self.log('ssim', c_ssim,sync_dist=True, prog_bar=True)
        self.log('ergas', c_ergas,sync_dist=True, prog_bar=True)
        self.log('sam',c_sam,sync_dist=True, prog_bar=True)
        # torch.save(out['llut']["0"].state_dict(), "saved_models/LUT0_%.4f.pth" % (c_ssim))
        # torch.save(out['llut']["1"].state_dict(), "saved_models/LUT1_%.4f.pth" % (c_ssim))
        # torch.save(out['llut']["2"].state_dict(), "saved_models/LUT2_%.4f.pth" % (c_ssim))
        #torch.save(out['clut'].state_dict(), "saved_models/LUT8_%.4d.pth" % (self.current_epoch))
        self.trainer.checkpoint_callback.best_model_score #save the best score model
        return {'val_loss': loss, 'psnr': psnr,'ergas':ergas,'sam':sam,'ssim':ssim}
    

def main():
    # parse the arguments
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='./configs/Train_DIRFL.json',type=str,
                            help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                            help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-r1', '--resume_ckpt', default=None, type=str,
                            help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default='2', type=str,
                            help='indices of GPUs to enable (default: all)')
    parser.add_argument('-v', '--val', default=False, type=bool,
                            help='Valdation')
    parser.add_argument('-val_path', default='/hpc2hdd/home/owentianye/LYL/Paper/TGRS-muti-focus2/Experiments/MMnet/DIV2K_Flickr2K_HR/EXP5/best_model-epoch:09-psnr:39.5784-ssim:0.9866.ckpt',type=str, help='Path to the val path')

    parser.add_argument('--local', action='store_true', default=False)
    global args
    args = parser.parse_args()
    # set resmue
    # args.resume ='/home/linyl/Workspace/Paper/DIRFL/Experiments/DIRFL/wv2_dataset/EXP1/best_model-epoch:36-psnr:31.4991-ergas:3.0113-sam:0.0751.ckpt'
    args.resume_ckpt ='/home/linyl/Workspace/Paper/DIRFL/Experiments/DIRFL/wv2_dataset/EXP3/best_model-epoch:999-psnr:42.8972-ergas:0.8094-sam:0.0198.ckpt'


    global config
    config = json.load(open(args.config))
    
    # Set seeds.
    seed = 42 #Global seed set to 42
    seed_everything(seed)
    
    # wandb log init
    global wandb_logger
    output_dir = './TensorBoardLogs'
    # logger = WandbLogger(project=config['name']+"-"+config["train_dataset"])
    logger = TensorBoardLogger(name=config['name']+"_"+config["train_dataset"],save_dir = output_dir )
    
    # Setting up path
    global PATH
    PATH = "./"+config["experim_name"]+"/"+config["train_dataset"]+"/"+str(config["tags"])
    ensure_dir(PATH+"/")
    shutil.copy2(args.config, PATH)

    # init pytorch-litening
    ddp = DDPStrategy(process_group_backend="nccl",find_unused_parameters=True)
    model = CoolSystem()
    
    # set checkpoint mode and init ModelCheckpointHook
    checkpoint_callback = ModelCheckpoint(
    monitor='psnr',
    dirpath=PATH,
    filename='best_model-epoch:{epoch:02d}-psnr:{psnr:.4f}-ssim:{ssim:.4f}-ergas:{ergas:.4f}-sam:{sam:.4f}',
    auto_insert_metric_name=False,   
    every_n_epochs=config["trainer"]["test_freq"],
    save_on_train_epoch_end=True,
    save_top_k=200,
    mode = "max"
    )
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    ema_callback = EMACallback(decay=0.999,every_n_steps=1)

    
    trainer = pl.Trainer(
        max_epochs=config["trainer"]["total_epochs"],
        accelerator='gpu', devices=[0],
        logger=logger,
        accumulate_grad_batches=4,
        # gradient_clip_algorithm="norm",
        # gradient_clip_val=0.5,
        #amp_backend="apex",
        #amp_level='01',
        #accelerator='ddp',
        # precision='16-mixed',
        callbacks =  [checkpoint_callback,lr_monitor_callback,ema_callback],
        check_val_every_n_epoch = config["trainer"]["test_freq"],
        log_every_n_steps=20,
        # fast_dev_run=True,
    )   
    
    if args.val == True:
        trainer.validate(model,ckpt_path=args.val_path)
    else:
        # resume from ckpt pytorch lightening
        # trainer.fit(model,ckpt_path=args.resume_ckpt)
        # resume from pth pytorch 
        trainer.fit(model)

if __name__ == '__main__':
    print('-----------------------------------------train_pl.py trainning-----------------------------------------')
    main()
    