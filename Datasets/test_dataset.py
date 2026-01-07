
import torch.utils.data as data
import torch, random, os
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
from random import randrange
import torch.nn.functional as F
import h5py
from glob import glob
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', 'tif', 'TIF'])


def load_img(filepath):
    img = Image.open(filepath)
    #img = Image.open(filepath)
    return img


# def rescale_img(img_in, scale):
#     # img_in shape: (batch_size, channels, height, width)
#     # 计算新的高度和宽度
#     new_size_in = [int(img_in.shape[2] * scale), int(img_in.shape[3] * scale)]
    
#     # 使用双三次插值进行图像缩放
#     img_in_rescaled = F.interpolate(img_in, size=new_size_in, mode='bicubic', align_corners=False)
    
#     return img_in_rescaled.squeeze(0)

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in
def get_patch(ms_image, lms_image, pan_image, bms_image, lpan_image, patch_size, scale, ix=-1, iy=-1):
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
    #bms_image = bms_image.crop((ty,tx,ty + tp, tx + tp))
    lpan_image = lpan_image.crop((ty,tx,ty + tp, tx + tp))            
    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return ms_image, lms_image, pan_image, bms_image, lpan_image,info_patch
# def get_patch(ms_image, lms_image, pan_image, bms_image, patch_size, scale, ix=-1, iy=-1):
#     (ih, iw) = lms_image.size
#     (th, tw) = (scale * ih, scale * iw)

#     patch_mult = scale #if len(scale) > 1 else 1
#     tp = patch_mult * patch_size
#     ip = tp // scale

#     if ix == -1:
#         ix = random.randrange(0, iw - ip + 1)
#     if iy == -1:
#         iy = random.randrange(0, ih - ip + 1)

#     (tx, ty) = (scale * ix, scale * iy)

#     lms_image = lms_image.crop((iy,ix,iy + ip, ix + ip))
#     ms_image = ms_image.crop((ty,tx,ty + tp, tx + tp))
#     pan_image = pan_image.crop((ty,tx,ty + tp, tx + tp))
#     bms_image = bms_image.crop((ty,tx,ty + tp, tx + tp))
                
#     info_patch = {
#         'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

#     return ms_image, lms_image, pan_image, bms_image, info_patch

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

class Data(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None):
        super(Data, self).__init__()
    
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor, ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        lms_image = ms_image.resize((int(ms_image.size[0]/self.upscale_factor),int(ms_image.size[1]/self.upscale_factor)), Image.BICUBIC)       
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        lpan_image = pan_image.resize((int(pan_image.size[0]/self.upscale_factor),int(pan_image.size[1]/self.upscale_factor)), Image.BICUBIC)
        bms_image = rescale_img(lms_image, 2)
           
        #ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image, self.patch_size, scale=self.upscale_factor)
        ms_image, lms_image, pan_image, bms_image, lpan_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image, lpan_image, self.patch_size, scale=self.upscale_factor)
        
        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)
        
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
        PAN_image = [pan_image,lpan_image]
        reference = ms_image
        lms = bms_image
        return lms, MS_image, PAN_image, reference

    def __len__(self):
        return len(self.ms_image_filenames)
class Test_GF2_dataset(data.Dataset):
    def __init__(
        self, data_dir
    ):
        super(Test_GF2_dataset, self).__init__()
        img_scale = 1023.0
        data = h5py.File(data_dir)  # NxCxHxW = 0x1x2x3

        print(f"loading test_GF2_dataset: {data_dir} with {img_scale}")
        # tensor type:
        #print(list(data.keys()))
        lms = data["lms"][...]  # convert to np tpye for CV2.filter
        lms = np.array(lms, dtype=np.float32) / img_scale

        self.lms = torch.from_numpy(lms)
        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

        # if 'valid' in file_path:
        #     self.gt = self.gt.permute([0, 2, 3, 1])
        print('test_GF2_dataset initing:')
        print(pan1.shape, ms1.shape,lms.shape)
    #####必要函数
    def __getitem__(self, index):
        MS_image = self.ms[index, :, :, :].type(torch.FloatTensor)
        PAN_image = self.pan[index, :, :, :].type(torch.FloatTensor) 
        bms_image = self.lms[index, :, :, :].type(torch.FloatTensor)
        return MS_image, PAN_image, {},bms_image


    def __len__(self):
        return self.ms.shape[0]
class Test_GF2_dataset_RR_h5(data.Dataset):
    def __init__(
        self, data_dir
    ):
        super(Test_GF2_dataset_RR_h5, self).__init__()
        img_scale = 1023.0
        data = h5py.File(data_dir)  # NxCxHxW = 0x1x2x3
        
        print(f"loading Test_GF2_dataset_RR_h5: {data_dir} with {img_scale}")
        # tensor type:
        # lms = data["lms"][...]  # convert to np tpye for CV2.filter
        
        # lms = np.array(lms, dtype=np.float32) / img_scale
        
        # self.lms = torch.from_numpy(lms)
        # print(self.lms.max())
        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale
        
        self.ms = torch.from_numpy(ms1)
        
        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        
        gt = data['gt'][...]
        gt = np.array(gt, dtype=np.float32)/ img_scale # Nx1xHxW
        
        self.gt = torch.from_numpy(gt)

        # if 'valid' in file_path:
        #     self.gt = self.gt.permute([0, 2, 3, 1])
        print('Test_GF2_dataset_RR_h5 initing:')
        print(pan1.shape, ms1.shape, gt.shape)
    #####必要函数
    def __getitem__(self, index):
        MS_image = self.ms[index, :, :, :].type(torch.FloatTensor)
        PAN_image = self.pan[index, :, :, :].type(torch.FloatTensor)
        GT_image = self.gt[index, :, :, :].type(torch.FloatTensor)
        bms_image = rescale_img(MS_image.unsqueeze(0),4)
        return MS_image, PAN_image,GT_image,bms_image
    
    def __len__(self):
        return self.ms.shape[0]
class Test_wv3_dataset_FR_h5(data.Dataset):
    def __init__(
        self, data_dir
    ):
        super(Test_wv3_dataset_FR_h5, self).__init__()
        img_scale = 2047.0
        data = h5py.File(data_dir)  # NxCxHxW = 0x1x2x3

        print(f"loading Test_wv3_dataset_FR_h5: {data_dir} with {img_scale}")
        # tensor type:
        lms = data["lms"][...]  # convert to np tpye for CV2.filter
        lms = np.array(lms, dtype=np.float32) / img_scale

        self.lms = torch.from_numpy(lms)
        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:

        # if 'valid' in file_path:
        #     self.gt = self.gt.permute([0, 2, 3, 1])
        print('Test_wv3_dataset_FR_h5 initing:')
        print(pan1.shape, ms1.shape,lms.shape)
    #####必要函数
    def __getitem__(self, index):
        MS_image = self.ms[index, :, :, :].type(torch.FloatTensor)
        PAN_image = self.pan[index, :, :, :].type(torch.FloatTensor)
        bms_image = self.lms[index, :, :, :].type(torch.FloatTensor)
        return MS_image, PAN_image,{},bms_image


    def __len__(self):
        return self.ms.shape[0]
class Test_wv3_dataset_RR_h5(data.Dataset):
    def __init__(
        self, data_dir
    ):
        super(Test_wv3_dataset_RR_h5, self).__init__()
        img_scale = 2047.0
        data = h5py.File(data_dir)  # NxCxHxW = 0x1x2x3
        
        print(f"loading Test_wv3_dataset_RR_h5: {data_dir} with {img_scale}")
        # tensor type:
        lms = data["lms"][...]  # convert to np tpye for CV2.filter
        lms = np.array(lms, dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms)
        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale

        self.ms = torch.from_numpy(ms1)
        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # Nx1xHxW:
        
        gt = data['gt'][...]
        gt = np.array(gt, dtype=np.float32)/ img_scale # Nx1xHxW
        self.gt = torch.from_numpy(gt)

        # if 'valid' in file_path:
        #     self.gt = self.gt.permute([0, 2, 3, 1])
        print('Test_wv3_dataset_RR_h5 initing:')
        print(pan1.shape, ms1.shape, gt.shape,lms.shape)
    #####必要函数
    def __getitem__(self, index):
        MS_image = self.ms[index, :, :, :].type(torch.FloatTensor)
        PAN_image = self.pan[index, :, :, :].type(torch.FloatTensor)
        GT_image = self.gt[index, :, :, :].type(torch.FloatTensor)
        bms_image = self.lms[index, :, :, :].type(torch.FloatTensor)
        return MS_image, PAN_image,GT_image,bms_image
    
    def __len__(self):
        return self.ms.shape[0]
class Data_test(data.Dataset):
    def __init__(self, data_dir_ms, data_dir_pan, cfg, transform=None):
        super(Data_test, self).__init__()
    
        self.ms_image_filenames = [join(data_dir_ms, x) for x in listdir(data_dir_ms) if is_image_file(x)]
        self.pan_image_filenames = [join(data_dir_pan, x) for x in listdir(data_dir_pan) if is_image_file(x)]

        self.patch_size = cfg['data']['patch_size']
        self.upscale_factor = cfg['data']['upsacle']
        self.transform = transform
        self.data_augmentation = cfg['data']['data_augmentation']
        self.normalize = cfg['data']['normalize']
        self.cfg = cfg

    def __getitem__(self, index):
        
        ms_image = load_img(self.ms_image_filenames[index])
        pan_image = load_img(self.pan_image_filenames[index])
        _, file = os.path.split(self.ms_image_filenames[index])
        ms_image = ms_image.crop((0, 0, ms_image.size[0] // self.upscale_factor * self.upscale_factor, ms_image.size[1] // self.upscale_factor * self.upscale_factor))
        if self.cfg['test']['datatype'] == 'fullGF2' or self.cfg['test']['datatype'] == 'fullWV2':
            lms_image = ms_image
        else:
            lms_image = ms_image.resize((int(ms_image.size[0]/self.upscale_factor),int(ms_image.size[1]/self.upscale_factor)), Image.BICUBIC)       
        pan_image = pan_image.crop((0, 0, pan_image.size[0] // self.upscale_factor * self.upscale_factor, pan_image.size[1] // self.upscale_factor * self.upscale_factor))
        bms_image = rescale_img(lms_image, self.upscale_factor)
           
        # ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image, self.patch_size, scale=self.upscale_factor)
        # ms_image, lms_image, pan_image, bms_image, _ = get_patch(ms_image, lms_image, pan_image, bms_image, self.patch_size, scale=self.upscale_factor)

        if self.data_augmentation:
            ms_image, lms_image, pan_image, bms_image, _ = augment(ms_image, lms_image, pan_image, bms_image)
        
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

        return ms_image, lms_image, pan_image, bms_image,file

    def __len__(self):
        return len(self.ms_image_filenames)

class VIF_Data(data.Dataset):
    def __init__(self, data_dir_vi, data_dir_ir,data_dir_fuse, cfg, transform=None):
        self.cfg = cfg
        # data_dir_vi = cfg[cfg['train_dataset']]['data_dir']['val_dir']['data_dir_vi']
        # data_dir_ir = cfg[cfg['train_dataset']]['data_dir']['val_dir']['data_dir_ir']
        # data_dir_fuse = cfg[cfg['train_dataset']]['data_dir']['val_dir']['data_dir_fuse']
        self.data_augmentation = False
        self.visible_files = sorted(glob(os.path.join(data_dir_vi, "*.*")))
        self.infrared_files = sorted(glob(os.path.join(data_dir_ir, "*.*")))
        self.other_fuse_files = sorted(glob(os.path.join(data_dir_fuse, "*.*")))
        self.transform = transform

    def __len__(self):
        l = len(self.infrared_files)
        return l

    def __getitem__(self, item):
        # print(len(self.infrared_files))
        # print(len(self.visible_files))
        # print(len(self.other_fuse_files))

        image_A_path = self.visible_files[item]
        image_B_path = self.infrared_files[item]
        other_fuse_path = self.other_fuse_files[item]
        image_A = Image.open(image_A_path).convert(mode='RGB')
        image_B = Image.open(image_B_path).convert(mode='L')   ##########
        other_fuse = Image.open(other_fuse_path).convert(mode='RGB')

        if self.transform is not None:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)
            other_fuse = self.transform(other_fuse)

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, other_fuse, name