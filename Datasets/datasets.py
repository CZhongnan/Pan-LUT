import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor
import cv2
import h5py
from glob import glob
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF'])

def transform():
    return Compose([
        ToTensor(),
    ])
    
def load_img(filepath):
    img = Image.open(filepath)
    #img = Image.open(filepath)
    return img

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in
# def rescale_img(img_in, scale):
#     # img_in shape: (batch_size, channels, height, width)
#     # 计算新的高度和宽度
#     new_size_in = [int(img_in.shape[2] * scale), int(img_in.shape[3] * scale)]
    
#     # 使用双三次插值进行图像缩放
#     img_in_rescaled = F.interpolate(img_in, size=new_size_in, mode='bicubic', align_corners=False)
    
#     return img_in_rescaled.squeeze(0)

def get_patch(ms_image, lms_image, pan_image, bms_image, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = lms_image.size
    (th, tw) = (scale * ih, scale * iw)

    patch_mult = scale #if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    (tx, ty) = (scale * ix, scale * iy)

    lms_image = lms_image.crop((iy,ix,iy + ip, ix + ip))
    ms_image = ms_image.crop((ty,tx,ty + tp, tx + tp))
    pan_image = pan_image.crop((ty,tx,ty + tp, tx + tp))
    bms_image = bms_image.crop((ty,tx,ty + tp, tx + tp))
    #lpan_image = lpan_image.crop((iy,ix,iy + ip, ix + ip))            
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return ms_image, lms_image, pan_image, bms_image,info_patch

def augment(ms_image, lms_image, pan_image, bms_image, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    if random.random() < 0.5 and flip_h:
        ms_image = ImageOps.flip(ms_image)
        lms_image = ImageOps.flip(lms_image)
        pan_image = ImageOps.flip(pan_image)
        bms_image = ImageOps.flip(bms_image)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            ms_image = ImageOps.mirror(ms_image)
            lms_image = ImageOps.mirror(lms_image)
            pan_image = ImageOps.mirror(pan_image)
            bms_image = ImageOps.mirror(bms_image)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            ms_image = ms_image.rotate(180)
            lms_image = lms_image.rotate(180)
            pan_image = pan_image.rotate(180)
            bms_image = bms_image.rotate(180)
            info_aug['trans'] = True
            
    return ms_image, lms_image, pan_image, bms_image, info_aug
class qb_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False
    ):
        super(qb_dataset, self).__init__()
        self.config = config
        if is_train == True:
            data_dir = self.config['qb_dataset_h5']['data_dir']['train_dir']
        else:
            data_dir = self.config['qb_dataset_h5']['data_dir']['val_dir']
        img_scale = self.config['qb_dataset_h5']['max_value']
        data = h5py.File(data_dir)  # NxCxHxW = 0x1x2x3

        print(f"loading qb_dataset: {data_dir} with {img_scale}")
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)

        # lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        # lms1 = np.array(lms1, dtype=np.float32) / img_scale
        # self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

        # if 'valid' in file_path:
        #     self.gt = self.gt.permute([0, 2, 3, 1])

        print(pan1.shape, gt1.shape, ms1.shape)
    #####必要函数
    def __getitem__(self, index):
        MS_image = self.ms[index, :, :, :].type(torch.FloatTensor)
        PAN_image = self.pan[index, :, :, :].type(torch.FloatTensor) 
        reference = self.gt[index, :, :, :].type(torch.FloatTensor)
        bms_image = bms_image = rescale_img(MS_image.unsqueeze(0),4)
        return bms_image, MS_image, PAN_image, reference


    def __len__(self):
        return self.gt.shape[0]
class wv3_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True
    ):
        super(wv3_dataset, self).__init__()
        self.config = config
        if is_train == True:
            data_dir = self.config['wv3_dataset_h5']['data_dir']['train_dir']
        else:
            data_dir = self.config['wv3_dataset_h5']['data_dir']['val_dir']
        img_scale = self.config['wv3_dataset_h5']['max_value']
        #print(self.config['wv3_dataset_h5']['data_dir']['val_dir'])
        data = h5py.File(data_dir)  # NxCxHxW = 0x1x2x3

        print(f"loading wv3_dataset: {data_dir} with {img_scale}")
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

        # if 'valid' in file_path:
        #     self.gt = self.gt.permute([0, 2, 3, 1])

        print(pan1.shape, lms1.shape, gt1.shape, ms1.shape)
    #####必要函数
    def __getitem__(self, index):
        MS_image = self.ms[index, :, :, :].type(torch.FloatTensor)
        PAN_image = self.pan[index, :, :, :].type(torch.FloatTensor)
        reference = self.gt[index, :, :, :].type(torch.FloatTensor)
        bms_image = self.lms[index, :, :, :].type(torch.FloatTensor)
        return bms_image, MS_image, PAN_image, reference


    def __len__(self):
        return self.gt.shape[0]
class GF2_dataset(data.Dataset):
    def __init__(
        self, config, is_train=True, is_dhp=False
    ):
        super(GF2_dataset, self).__init__()
        self.config = config
        if is_train == True:
            data_dir = self.config['GF2_dataset_h5']['data_dir']['train_dir']
        else:
            data_dir = self.config['GF2_dataset_h5']['data_dir']['val_dir']
        img_scale = self.config['GF2_dataset_h5']['max_value']
        data = h5py.File(data_dir)  # NxCxHxW = 0x1x2x3

        print(f"loading GF2_dataset: {data_dir} with {img_scale}")
        # tensor type:
        gt1 = data["gt"][...]  # convert to np tpye for CV2.filter
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)  # NxCxHxW:

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)

        # lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        # lms1 = np.array(lms1, dtype=np.float32) / img_scale
        # self.lms = torch.from_numpy(lms1)


        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

        # if 'valid' in file_path:
        #     self.gt = self.gt.permute([0, 2, 3, 1])

        print(pan1.shape, gt1.shape, ms1.shape)
    #####必要函数
    def __getitem__(self, index):
        MS_image = self.ms[index, :, :, :].type(torch.FloatTensor)
        PAN_image = self.pan[index, :, :, :].type(torch.FloatTensor) 
        reference = self.gt[index, :, :, :].type(torch.FloatTensor)
        bms_image = rescale_img(MS_image.unsqueeze(0),4)
        return bms_image, MS_image, PAN_image, reference


    def __len__(self):
        return self.gt.shape[0]
    
    
class Test_GF2_dataset(data.Dataset):
    def __init__(
        self, data_dir
    ):
        super(Test_GF2_dataset, self).__init__()
        img_scale = 1023.0
        data = h5py.File(data_dir)  # NxCxHxW = 0x1x2x3

        print(f"loading test_GF2_dataset: {data_dir} with {img_scale}")
        # tensor type:
        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

        # if 'valid' in file_path:
        #     self.gt = self.gt.permute([0, 2, 3, 1])
        print('test_GF2_dataset initing:')
        print(pan1.shape, ms1.shape)
    #####必要函数
    def __getitem__(self, index):
        MS_image = self.ms[index, :, :, :].type(torch.FloatTensor)
        PAN_image = self.pan[index, :, :, :].type(torch.FloatTensor) 
        return MS_image, PAN_image


    def __len__(self):
        return self.ms.shape[0]

class Data(data.Dataset):#self, config, is_train=True,
    def __init__(self,cfg, is_train=True, transform=transform()):
        super(Data, self).__init__()
        self.cfg = cfg
        self.is_train = is_train
        if is_train == True:
            data_dir_ms = cfg[cfg['train_dataset']]['data_dir']['train_dir']['data_dir_ms']
            data_dir_pan = cfg[cfg['train_dataset']]['data_dir']['train_dir']['data_dir_pan']
            self.data_augmentation = cfg[cfg['train_dataset']]['data_augmentation']
        else:
            data_dir_ms = cfg[cfg['train_dataset']]['data_dir']['val_dir']['data_dir_ms']
            data_dir_pan = cfg[cfg['train_dataset']]['data_dir']['val_dir']['data_dir_pan']
            self.data_augmentation = False
            
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        #  self.cfg[self.cfg['train_dataset']]['max_value']           
        self.patch_size = cfg[cfg['train_dataset']]['patch_size']
        self.upscale_factor = cfg[cfg['train_dataset']]['factor']
        self.transform = transform
        
        self.normalize = cfg[cfg['train_dataset']]['normalize']
        

    def __getitem__(self, index):
        
        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor, ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        
        lms_image = ms_image.resize((int(ms_image.size[0]/self.upscale_factor),int(ms_image.size[1]/self.upscale_factor)), Image.BICUBIC)       
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        
        bms_image = rescale_img(lms_image,self.upscale_factor)
        # if self.is_train == True:
        ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image, self.patch_size, scale=self.upscale_factor)
        # ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image, self.patch_size, scale=self.upscale_factor)
        
        if self.data_augmentation:
            ms_image, lms_image, pan_image,bms_image, _ = augment(ms_image, lms_image,pan_image,bms_image)
        
        if self.transform:
            ms_image = self.transform(ms_image)
            lms_image = self.transform(lms_image)
            
            pan_image = self.transform(pan_image)
            
            bms_image = self.transform(bms_image)
        if self.normalize:
            ms_image = ms_image * 2 - 1
            lms_image = lms_image * 2 - 1
            pan_image = pan_image * 2 - 1
            bms_image = bms_image * 2 - 1

        # return ms_image, lms_image, pan_image, bms_image, file
        MS_image =lms_image
        
        PAN_image = pan_image
        
        reference = ms_image
        lms = bms_image
        
        return lms, MS_image,PAN_image,reference

    def __len__(self):
        return len(self.ms_image_filenames)
    

class Data_test(data.Dataset):#self, config, is_train=True,
    def __init__(self,cfg, transform=transform()):
        super(Data_test, self).__init__()
        self.cfg = cfg
        
        data_dir_ms = cfg[cfg['train_dataset']]['data_dir']['val_dir']['data_dir_ms']
        data_dir_pan = cfg[cfg['train_dataset']]['data_dir']['val_dir']['data_dir_pan']
        self.data_augmentation = False
        
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]
        #  self.cfg[self.cfg['train_dataset']]['max_value']           
        self.patch_size = cfg[cfg['train_dataset']]['patch_size']
        self.upscale_factor = cfg[cfg['train_dataset']]['factor']
        self.transform = transform
        
        self.normalize = cfg[cfg['train_dataset']]['normalize']
        

    def __getitem__(self, index):
        
        
        original_msi = np.array(Image.open(self.ms_image_filenames[index]))
        original_pan = np.array(Image.open(self.pan_image_filenames[index]))
        '''normalization'''
        original_msi = np.float32(original_msi) / 255
        original_pan = np.float32(original_pan) / 255
        
        used_ms = cv2.resize(original_msi, (original_msi.shape[1]//4, original_msi.shape[0]//4), cv2.INTER_CUBIC)
        used_lms = cv2.resize(used_ms, (original_msi.shape[1], original_msi.shape[0]), cv2.INTER_CUBIC)
        used_pan = np.expand_dims(original_pan, -1)

        # ms_image = load_img(self.ms_image_filenames[index])
        # pan_image = load_img(self.pan_image_filenames[index])
        # _, file = os.path.split(self.ms_image_filenames[index])
        # ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor, ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        # lms_image = ms_image.resize((int(ms_image.size[0]/self.upscale_factor),int(ms_image.size[1]/self.upscale_factor)), Image.BICUBIC)       
        # pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        # bms_image = rescale_img(lms_image, self.upscale_factor)
           
        # ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image, self.patch_size, scale=self.upscale_factor)
        
        # if self.data_augmentation:
        #     ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)
        
        if self.transform:
            original_msi = self.transform(original_msi)
            original_pan = self.transform(original_pan)
            used_ms = self.transform(used_ms)
            used_lms = self.transform(used_lms)
            used_pan = self.transform(used_pan)

        # if self.normalize:
        #     ms_image = ms_image * 2 - 1
        #     lms_image = lms_image * 2 - 1
        #     pan_image = pan_image * 2 - 1
        #     bms_image = bms_image * 2 - 1

        # return ms_image, lms_image, pan_image, bms_image, file
        # RR
        reduced_msi =used_ms
        reduced_pan = used_pan
        reduced_mshr = used_lms
        gt = original_msi

        # FR
        original_msi = used_ms
        original_mshr = used_lms
        original_pan = used_pan
        return original_msi,original_mshr, original_pan, reduced_msi,reduced_mshr,reduced_pan,gt

    def __len__(self):
        return len(self.ms_image_filenames)
