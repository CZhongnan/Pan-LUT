from solver.basesolver import BaseSolver
import os, torch, time, cv2, importlib
import torch.backends.cudnn as cudnn
from Test_Datasets.data import *
from torch.utils.data import DataLoader
from torch.autograd import Variable 
import numpy as np
import json
import collections
from PGLUT import PGLUT_transform
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from models.models import MODELS
class Testsolver(BaseSolver):
    def __init__(self, cfg):
        super(Testsolver, self).__init__(cfg)
        
        net_name = self.cfg['algorithm']
        config  = json.load(open(self.cfg['test']['test_config_path']))
        self.model =  MODELS[net_name](config)


    def check(self):
        self.cuda = self.cfg['gpu_mode']
        torch.manual_seed(self.cfg['seed'])
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        if self.cuda:
            torch.cuda.manual_seed(self.cfg['seed'])
            cudnn.benchmark = True
              
            gups_list = self.cfg['gpus']
            self.gpu_ids = []
            for str_id in gups_list:
                gid = int(str_id)
                if gid >=0:
                    self.gpu_ids.append(gid)
            torch.cuda.set_device(self.gpu_ids[0]) 
            
            self.model_path = os.path.join(self.cfg['test']['model'])

            self.model = self.model.cuda(self.gpu_ids[0])
            # self.model.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))
            
            ckpt = torch.load(self.model_path, map_location=lambda storage, loc: storage)
            new_state_dict = collections.OrderedDict()
            for k in ckpt['state_dict']:
                            # print(k)
                            if k[:6] != 'model.':
                                continue
                            name = k[6:]
                            new_state_dict[name] = ckpt['state_dict'][k]
            self.model.load_state_dict(new_state_dict,strict=True)
            #  self.LUT = torch.load("/home/caizn/DIRFL/LUTPGF.pth")['LUT']
            # torch.save(self.model.LUTPGF.state_dict(), "/home/caizn/DIRFL/LUTPG31.8547.pth")
            # torch.save(self.model.LUT8.state_dict(), "/home/caizn/DIRFL/LUT8.pth")
            # torch.save(self.model.LUT00.state_dict(), "/home/caizn/DIRFL/LUT00.pth")
            # torch.save(self.model.LUT01.state_dict(), "/home/caizn/DIRFL/LUT01.pth")
            # torch.save(self.model.LUT02.state_dict(), "/home/caizn/DIRFL/LUT02.pth")
            # torch.save(self.model.LUT03.state_dict(), "/home/caizn/DIRFL/LUT03.pth")
            # torch.save(self.model.LUTPGF.state_dict(), "/home/caizn/DIRFL/LUTPGF.pth")
            


    def test(self):
        self.model.eval()
        avg_time = []
        for batch in self.data_loader:
            ms_image, lms_image, pan_image, bms_image,name = Variable(batch[0]), Variable(batch[1]),  Variable(batch[2]),Variable(batch[3]), (batch[4])
            if self.cuda:
                ms_image = ms_image.cuda(self.gpu_ids[0])
                lms_image = lms_image.cuda(self.gpu_ids[0])
                pan_image = pan_image.cuda(self.gpu_ids[0])
                bms_image = bms_image.cuda(self.gpu_ids[0])
                # print(ms_image.min(),ms_image.max())
                # print(lms_image.min(),lms_image.max())
                # print(pan_image.min(),pan_image.max())
                # print(bms_image.min(),bms_image.max())
                # print(bms_image.shape)
                # print(pan_image.shape)
                # print(bms_image.shape)
                #print(torch.max(ms_image))
                #print(torch.min(ms_image))  
            #Pan_image =  [pan_image,lpan_image]
            t0 = time.time()
            with torch.no_grad():
                # input=torch.cat([bms_image,pan_image],dim=1)
                # out = self.model(bms_image)
                out = self.model(lms_image, pan_image,bms_image)
                prediction = out["pred"]
                
                # pglut = out["ms"]
                # sdlut = out["p"]
            t1 = time.time()
            # prediction = torch.clip(prediction,min=0,max=1)
            if self.cfg['data']['normalize']:
                ms_image = (ms_image+1) /2
                lms_image = (lms_image+1) /2
                pan_image = (pan_image+1) /2
                bms_image = (bms_image+1) /2

            #print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            avg_time.append(t1 - t0)
            self.save_img(bms_image.cpu().data, name[0][0:-4]+'_bic.tif', mode='CMYK') #
            self.save_img(ms_image.cpu().data, name[0][0:-4]+'_gt.tif', mode='CMYK')
            self.save_img(prediction.cpu().data, name[0][0:-4]+'.tif', mode='CMYK')
            # self.save_img(pglut.cpu().data, name[0][0:-4]+'_ms.tif',mode='CMYK')
            # self.save_img(sdlut.cpu().data, name[0][0:-4]+'_p.tif',mode='CMYK')
        print("===> AVG Timer: %.4f sec." % (np.mean(avg_time)))
        

    def save_img(self, img, img_name, mode):
        save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
        #print((save_img.max()))
        # save img
        save_dir = os.path.join(self.cfg['test']['save_dir'], self.cfg['test']['type'])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        save_fn = save_dir +'/'+ img_name
        save_img = np.round(save_img*255).astype('uint8') #
        # save_img = np.uint8(save_img*255).astype('uint8')
        #print(save_img.max())
        save_img = Image.fromarray(save_img,mode)
        save_img.save(save_fn)
  
    def run(self):
        self.check()
        if self.cfg['test']['type'] == 'test':
            # print(self.cfg['test']['data_dir'])
            self.dataset = get_test_data(self.cfg, self.cfg['test']['data_dir'])
            self.data_loader = DataLoader(self.dataset, shuffle=False, batch_size=1,
                num_workers=self.cfg['threads'])
            self.test()
        else:
            raise ValueError('Mode error!')