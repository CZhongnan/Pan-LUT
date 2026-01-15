import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch
import numpy as np
import math
from torch.nn import init
from PGLUT import PGLUT_transform
from IF4DLUT import IF4DLUT_transform
from SDLUT import SDLUT_transform
from SD90LUT import SD90LUT_transform
from SD180LUT import SD180LUT_transform
from SD270LUT import SD270LUT_transform
from IFLUT import IFLUT_transform
dimPG=9
dimSD=17
dimAA=17
class GeneratorF5DLUT_identity(nn.Module):
    def __init__(self, dim=dimPG):
        super(GeneratorF5DLUT_identity, self).__init__()
        if dim == 9:
            file = open("Identity5DLUT9.txt", 'r')
        elif dim == 64:
            file = open("IdentityLUT64.txt", 'r')
        elif dim == 3:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT3.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((1,5,dim,dim,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    for l in range(0,dim):
                        for m in range(0,dim):
                            n = i * dim*dim*dim*dim + j * dim*dim*dim + k*dim*dim + l*dim + m
                            x = lines[n].split()
                            buffer[0,0,i,j,k,l,m] = float(x[0])
                            buffer[0,1,i,j,k,l,m] = float(x[1])
                            buffer[0,2,i,j,k,l,m] = float(x[2])
                            buffer[0,3,i,j,k,l,m] = float(x[3])
                            buffer[0,4,i,j,k,l,m] = float(x[4])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))

    def forward(self, x):
        output = PGLUT_transform(x, self.LUT)
        return output

class Generator5DLUT_identity(nn.Module):
    def __init__(self, dim=dimPG):
        super(Generator5DLUT_identity, self).__init__()
        if dim == 9:
            file = open("Identity5DLUT9.txt", 'r')
        elif dim == 17:
            file = open("/home/caizn/DIRFL/IdentityF5DLUT17.txt", 'r')
        elif dim == 3:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT3.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((1,4,dim,dim,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    for l in range(0,dim):
                        for m in range(0,dim):
                            n = i * dim*dim*dim*dim + j * dim*dim*dim + k*dim*dim + l*dim + m
                            x = lines[n].split()
                            buffer[0,0,i,j,k,l,m] = float(x[0])
                            buffer[0,1,i,j,k,l,m] = float(x[1])
                            buffer[0,2,i,j,k,l,m] = float(x[2])
                            buffer[0,3,i,j,k,l,m] = float(x[3])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))

    def forward(self, x):
        output = PGLUT_transform(x, self.LUT)
        return output

class Generator3DLUTL5_identity(nn.Module):
    def __init__(self, dim=dimSD,num_batch=5):
        super(Generator3DLUTL5_identity, self).__init__()
        if dim == 9:
            file = open("/home/caizn/Awaresome-pansharpening/4DLUT9.txt", 'r')
        elif dim == 6:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT6.txt", 'r')
        elif dim == 3:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT3.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((1,dim,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    for l in range(0,dim):
                        n = i * dim*dim*dim + j * dim*dim + k*dim + l
                        x = lines[n].split()
                        buffer[0,i,j,k,l] = float(x[0])
        buffer = torch.from_numpy(buffer).unsqueeze(0).repeat(num_batch,1,1,1,1,1)
        self.LUT = nn.Parameter(buffer.requires_grad_(True))

    def forward(self, x):
        output = SDLUT_transform(x,self.LUT)
        return output
    

class Generator3DLUTL90_identity(nn.Module):
    def __init__(self, dim=dimSD,num_batch=5):
        super(Generator3DLUTL90_identity, self).__init__()
        if dim == 9:
            file = open("/home/caizn/Awaresome-pansharpening/4DLUT9.txt", 'r')
        elif dim == 6:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT6.txt", 'r')
        elif dim == 3:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT3.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((1,dim,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    for l in range(0,dim):
                        n = i * dim*dim*dim + j * dim*dim + k*dim + l
                        x = lines[n].split()
                        buffer[0,i,j,k,l] = float(x[0])
        # print(buffer[0,0,0,0,:])
        buffer = torch.from_numpy(buffer).unsqueeze(0).repeat(num_batch,1,1,1,1,1)
        # print(buffer.shape)
        self.LUT = nn.Parameter(buffer.requires_grad_(True))
        # self.LUT = nn.Parameter(torch.from_numpy(buffer).unsqueeze(0).requires_grad_(True))
        # self.TrilinearL90Interpolation = TrilinearL90Interpolation()

    def forward(self, x):
        # _, output = self.TrilinearL90Interpolation(self.LUT, x)
        output = SD90LUT_transform(x,self.LUT)
        #self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
        return output
    def regularizations(self, smoothness, monotonicity):
        basis_luts = self.LUT
        tv, mn = 0, 0
        for i in range(2, basis_luts.ndimension()):
            diff = torch.diff(basis_luts.flip(i), dim=i)
            tv += torch.square(diff).sum(0).mean()
            mn += F.relu(diff).sum(0).mean()
        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        return reg_smoothness, reg_monotonicity


class Generator3DLUTL180_identity(nn.Module):
    def __init__(self, dim=dimSD,num_batch=5):
        super(Generator3DLUTL180_identity, self).__init__()
        if dim == 9:
            file = open("/home/caizn/Awaresome-pansharpening/4DLUT9.txt", 'r')
        elif dim == 6:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT6.txt", 'r')
        elif dim == 3:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT3.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((1,dim,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    for l in range(0,dim):
                        n = i * dim*dim*dim + j * dim*dim + k*dim + l
                        x = lines[n].split()
                        buffer[0,i,j,k,l] = float(x[0])
        # print(buffer[0,0,0,0,:])
        buffer = torch.from_numpy(buffer).unsqueeze(0).repeat(num_batch,1,1,1,1,1)
        # print(buffer.shape)
        self.LUT = nn.Parameter(buffer.requires_grad_(True))
        # self.LUT = nn.Parameter(torch.from_numpy(buffer).unsqueeze(0).requires_grad_(True))
        # self.TrilinearL180Interpolation = TrilinearL180Interpolation()

    def forward(self, x):
        # _, output = self.TrilinearL180Interpolation(self.LUT, x)
        output = SD180LUT_transform(x,self.LUT)
        #self.LUT, output = self.TrilinearInterpolation(self.LUT, x)
        return output
    def regularizations(self, smoothness, monotonicity):
        basis_luts = self.LUT
        tv, mn = 0, 0
        for i in range(2, basis_luts.ndimension()):
            diff = torch.diff(basis_luts.flip(i), dim=i)
            tv += torch.square(diff).sum(0).mean()
            mn += F.relu(diff).sum(0).mean()
        reg_smoothness = smoothness * tv
        reg_monotonicity = monotonicity * mn
        return reg_smoothness, reg_monotonicity

class Generator3DLUTL270_identity(nn.Module):
    def __init__(self, dim=dimSD,num_batch=5):
        super(Generator3DLUTL270_identity, self).__init__()
        if dim == 9:
            file = open("/home/caizn/Awaresome-pansharpening/4DLUT9.txt", 'r')
        elif dim == 6:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT6.txt", 'r')
        elif dim == 3:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT3.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((1,dim,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    for l in range(0,dim):
                        n = i * dim*dim*dim + j * dim*dim + k*dim + l
                        x = lines[n].split()
                        buffer[0,i,j,k,l] = float(x[0])
        # print(buffer[0,0,0,0,:])
        buffer = torch.from_numpy(buffer).unsqueeze(0).repeat(num_batch,1,1,1,1,1)
        # print(buffer.shape)
        self.LUT = nn.Parameter(buffer.requires_grad_(True))
        # self.LUT = nn.Parameter(buffer.requires_grad_(True))
        # self.LUT = nn.Parameter(torch.from_numpy(buffer).unsqueeze(0).requires_grad_(True))
        # self.TrilinearL270Interpolation = TrilinearL270Interpolation()

    def forward(self, x):
        output = SD270LUT_transform(x,self.LUT)
        return output
class Generator2DLUT_identity(nn.Module):
    def __init__(self, dim=251):
        super(Generator2DLUT_identity, self).__init__()
        if dim == 33:
            file = open("/home/caizn/Shuffle-Mamba/MIF/IdentityLUT33.txt", 'r')
        elif dim == 64:
            file = open("/home/caizn/Shuffle-Mamba/MIF/IdentityLUT64.txt", 'r')
        elif dim == 251:
            file = open("/home/caizn/LUT-Fuse/models/2DLUT251.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((2,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                        n = i * dim + j 
                        x = lines[n].split()
                        buffer[0,i,j] = float(x[0])
                        buffer[1,i,j] = float(x[1])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).unsqueeze(0).requires_grad_(True))
        # self.PentilinearInterpolation = PentilinearInterpolation()

    def forward(self, x):
        # _, output = self.PentilinearInterpolation(self.LUT, x)
        output = IFLUT_transform(x,self.LUT)
        #self.LUT, output = self.PentilinearInterpolation(self.LUT, x)
        return output
class Generator1DLUT_identity(nn.Module):
    def __init__(self, dim=251):
        super(Generator1DLUT_identity, self).__init__()
        if dim == 33:
            file = open("/home/caizn/Shuffle-Mamba/MIF/IdentityLUT33.txt", 'r')
        elif dim == 64:
            file = open("/home/caizn/Shuffle-Mamba/MIF/IdentityLUT64.txt", 'r')
        elif dim == 251:
            file = open("/home/caizn/LUT-Fuse/models/2DLUT251.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((1,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                        n = i * dim + j 
                        x = lines[n].split()
                        buffer[0,i,j] = float(x[0])

        self.LUT = nn.Parameter(torch.from_numpy(buffer).unsqueeze(0).requires_grad_(True))
        # self.PentilinearInterpolation = PentilinearInterpolation()

    def forward(self, x):
        # _, output = self.PentilinearInterpolation(self.LUT, x)
        output = IFLUT_transform(x,self.LUT)
        #self.LUT, output = self.PentilinearInterpolation(self.LUT, x)
        return output
class Generator4DLUT_identity(nn.Module):
    def __init__(self, dim=dimAA):
        super(Generator4DLUT_identity, self).__init__()
        if dim == 9:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT9.txt", 'r')
        elif dim == 17:
            file = open("/home/caizn/Mask-DiFuser/models/Identity4DLUT17.txt", 'r')
        elif dim == 3:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT3.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((4,dim,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    for l in range(0,dim):
                        n = i * dim*dim*dim + j * dim*dim + k*dim + l
                        x = lines[n].split()
                        buffer[0,i,j,k,l] = float(x[0])
                        buffer[1,i,j,k,l] = float(x[1])
                        buffer[2,i,j,k,l] = float(x[2])
                        buffer[3,i,j,k,l] = float(x[3])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).unsqueeze(0).requires_grad_(True))
        print(self.LUT.shape)
        # self.PentilinearInterpolation = PentilinearInterpolation()

    def forward(self, x):
        # _, output = self.PentilinearInterpolation(self.LUT, x)
        output = IF4DLUT_transform(x,self.LUT)
        #self.LUT, output = self.PentilinearInterpolation(self.LUT, x)
        return output
class Generator43DLUT_identity(nn.Module):
    def __init__(self, dim=dimAA):
        super(Generator43DLUT_identity, self).__init__()
        if dim == 9:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT9.txt", 'r')
        elif dim == 17:
            file = open("/home/caizn/Mask-DiFuser/models/Identity4DLUT17.txt", 'r')
        elif dim == 3:
            file = open("/home/caizn/Awaresome-pansharpening/Identity5DLUT3.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3,dim,dim,dim,dim), dtype=np.float32)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    for l in range(0,dim):
                        n = i * dim*dim*dim + j * dim*dim + k*dim + l
                        x = lines[n].split()
                        buffer[0,i,j,k,l] = float(x[0])
                        buffer[1,i,j,k,l] = float(x[1])
                        buffer[2,i,j,k,l] = float(x[2])
        self.LUT = nn.Parameter(torch.from_numpy(buffer).unsqueeze(0).requires_grad_(True))
        print(self.LUT.shape)
        # self.PentilinearInterpolation = PentilinearInterpolation()

    def forward(self, x):
        # _, output = self.PentilinearInterpolation(self.LUT, x)
        output = IF4DLUT_transform(x,self.LUT)
        #self.LUT, output = self.PentilinearInterpolation(self.LUT, x)
        return output