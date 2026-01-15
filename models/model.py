import torch
import torch.nn as nn
from models.modellut import *
class YCbCrToRGB(object):
    def __call__(self, img):
        return torch.stack((img[:, 0, :, :] + (img[:, 2, :, :] - 128 / 256.) * 1.402,
                            img[:, 0, :, :] - (img[:, 1, :, :] - 128 / 256.) * 0.344136 - (img[:, 2, :, :] - 128 / 256.) * 0.714136,
                            img[:, 0, :, :] + (img[:, 1, :, :] - 128 / 256.) * 1.772),
                            dim=1)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.LUT00 = Generator3DLUTL5_identity(num_batch=4) 
        self.LUT01 = Generator3DLUTL90_identity(num_batch=4)
        self.LUT02 = Generator3DLUTL180_identity(num_batch=4)
        self.LUT03 = Generator3DLUTL270_identity(num_batch=4)
        self.LUT8 = Generator4DLUT_identity()
        self.LUTPGF = Generator43DLUT_identity()
    def forward(self, vi_image,ir_image):
        # print(vi_image.shape)
        con0  = torch.cat([vi_image,ir_image],dim=1)
        pg0  = self.LUT8(con0)
        pg0 = F.hardtanh(pg0,0,1).permute(1, 0, 2, 3)
        sd00 = self.LUT00(pg0)
        sd00 = F.hardtanh(sd00,0,1)
        sd01 = self.LUT01(sd00)
        sd01 = F.hardtanh(sd01,0,1)
        sd02 = self.LUT02(sd01)
        sd02 = F.hardtanh(sd02,0,1)
        sd03 = self.LUT03(sd02)
        sd03 = F.hardtanh(sd03,0,1).permute(1, 0, 2, 3)
        pg1 = self.LUTPGF(sd03)
        # pg1=F.hardtanh(pg1,0,1)
        return pg1
class Net_MMIF(nn.Module):
    def __init__(self):
        super(Net_MMIF, self).__init__()
        self.LUT00 = Generator3DLUTL5_identity(num_batch=2) 
        self.LUT01 = Generator3DLUTL90_identity(num_batch=2)
        self.LUT02 = Generator3DLUTL180_identity(num_batch=2)
        self.LUT03 = Generator3DLUTL270_identity(num_batch=2)
        self.LUT8 = Generator2DLUT_identity()
        self.LUTPGF = Generator1DLUT_identity()
        self.LUTCB = Generator1DLUT_identity()
        self.LUTCR = Generator1DLUT_identity()

    def forward(self, A_image,B_image,cb,cr):
        # print(vi_image.shape)
        con0  = torch.cat([A_image,B_image],dim=1)
        pg0  = self.LUT8(con0)
        pg0 = torch.clamp(pg0,0,1).permute(1, 0, 2, 3)
        sd00 = torch.clamp(self.LUT00(pg0),0,1)
        sd01 = torch.clamp(self.LUT01(sd00),0,1)
        sd02 = torch.clamp(self.LUT02(sd01),0,1)
        sd03 = torch.clamp(self.LUT03(sd02),0,1).permute(1, 0, 2, 3)
        pg1 = self.LUTPGF(sd03)
        fcb = self.LUTCB(cb)
        fcr = self.LUTCR(cr)
        # weights = F.softmax(pg1, dim=1)
        # print(weights.shape)
        # weight_map = con0[:,:1,:,:] * weights[:,:1,:,:] + con0[:,1:2,:,:] * weights[:,1:2,:,:]
        out = YCbCrToRGB()(torch.cat((pg1,fcb, fcr),dim=1))
        # weight_map=torch.clamp(weight_map,0,1)
        return out