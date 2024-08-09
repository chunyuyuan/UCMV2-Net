import torch


import torchvision

import torch.nn as nn

import math

import time
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types

from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb
import torch.nn.functional as F

__all__ = ['UCM_NetV2']


import torch
import torch.nn.functional as F
import math





class LayerNorm(nn.Module):
    """ From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf) """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class DWConv1(nn.Module):
    def __init__(self, dim=768):
        super(DWConv1, self).__init__()
    
        self.dwconv = nn.Conv2d(2*dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = F.layer_norm(x, [C, H, W])
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x




class UCMBlock1(nn.Module):
    def __init__(self, dim,  mlp_ratio=4.,   drop=0., 
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.dwconv = DWConv1(mlp_hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Norm and DropPath from original UCMBlock
        x = self.norm2(x)
        
        # Begin merged shiftmlp forward logic
        B, N, C = x.shape
        x1 = x.clone()
        
        x = x.reshape(B * N, C).contiguous()
        x2 = x.clone()
        
        x = self.fc1(x)
        x = x.reshape(B, N, -1).contiguous()
        x+=x1

        x2[[0, B*N-1], :] = x2[[B*N-1, 0], :]
        x2 = self.fc2(x2)
        x2[[0, B*N-1], :] = x2[[B*N-1, 0], :]
        x2 = x2.reshape(B, N, -1).contiguous()
        x2+=x1
        x= torch.cat((x, x2), dim=2)
        

        x = self.dwconv(x, H, W)

        x += x1
        
        # Apply DropPath
        x = x + self.drop_path(x)
        
        return x

class ImageConv2D(nn.Module):
    
    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=1, stride=2)
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x, H, W



class UCM_NetV2(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=256, patch_size=16, in_chans=3,
                 embed_dims=[8, 16, 24, 32, 48, 64, 3],   drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv2d(embed_dims[-1], embed_dims[0], 3, stride=1, padding=1)
        self.ebn1 = nn.GroupNorm(4, embed_dims[0])
        self.ebn2 = nn.GroupNorm(4, embed_dims[1])
        self.ebn3 = nn.GroupNorm(4, embed_dims[2])
        self.norm1 = norm_layer(embed_dims[1])
        self.norm2 = norm_layer(embed_dims[2])
        self.norm3 = norm_layer(embed_dims[3])
        self.norm4 = norm_layer(embed_dims[4])
        self.norm5 = norm_layer(embed_dims[5])
        self.dnorm2 = norm_layer(embed_dims[4])
        self.dnorm3 = norm_layer(embed_dims[3])
        self.dnorm4 = norm_layer(embed_dims[2])
        self.dnorm5 = norm_layer(embed_dims[1])
        self.dnorm6 = norm_layer(embed_dims[0])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.block_0_1 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[1],  mlp_ratio=1, 
            drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)])

        self.block0 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[2],  mlp_ratio=1, 
            drop=drop_rate,  drop_path=dpr[0], norm_layer=norm_layer)])

        self.block1 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[3],  mlp_ratio=1, 
            drop=drop_rate,  drop_path=dpr[0], norm_layer=norm_layer)])

        self.block2 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[4], mlp_ratio=1,
            drop=drop_rate,drop_path=dpr[1], norm_layer=norm_layer)])

        self.block3 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[5],  mlp_ratio=1, 
            drop=drop_rate,  drop_path=dpr[1], norm_layer=norm_layer)])

        self.dblock0 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[4], mlp_ratio=1,
            drop=drop_rate,drop_path=dpr[0], norm_layer=norm_layer)])

        self.dblock1 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[3],  mlp_ratio=1, 
            drop=drop_rate,  drop_path=dpr[0], norm_layer=norm_layer)])

        self.dblock2 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[2],  mlp_ratio=1, 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        self.dblock3 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[1],  mlp_ratio=1, 
            drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        self.dblock4 = nn.ModuleList([UCMBlock1(
            dim=embed_dims[0],  mlp_ratio=1, 
            drop=drop_rate,  drop_path=dpr[1], norm_layer=norm_layer)])

        self.patch_embed1 = ImageConv2D(in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed2 = ImageConv2D(in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed3 = ImageConv2D(in_chans=embed_dims[2], embed_dim=embed_dims[3])
        self.patch_embed4 = ImageConv2D( in_chans=embed_dims[3], embed_dim=embed_dims[4])
        self.patch_embed5 = ImageConv2D(in_chans=embed_dims[4], embed_dim=embed_dims[5])

        self.decoder0 = nn.Conv2d(embed_dims[5], embed_dims[4], 1, stride=1, padding=0)
        self.decoder1 = nn.Conv2d(embed_dims[4], embed_dims[3], 1, stride=1, padding=0)
        self.decoder2 = nn.Conv2d(embed_dims[3], embed_dims[2], 1, stride=1, padding=0)
        self.decoder3 = nn.Conv2d(embed_dims[2], embed_dims[1], 1, stride=1, padding=0)
        self.decoder4 = nn.Conv2d(embed_dims[1], embed_dims[0], 1, stride=1, padding=0)
        self.decoder5 = nn.Conv2d(embed_dims[0], embed_dims[-1], 1, stride=1, padding=0)

        self.dbn0 = nn.GroupNorm(4, embed_dims[4])
        self.dbn1 = nn.GroupNorm(4, embed_dims[3])
        self.dbn2 = nn.GroupNorm(4, embed_dims[2])
        self.dbn3 = nn.GroupNorm(4, embed_dims[1])
        self.dbn4 = nn.GroupNorm(4, embed_dims[0])

        self.finalpre0 = nn.Conv2d(embed_dims[4], num_classes, kernel_size=1)
        self.finalpre1 = nn.Conv2d(embed_dims[3], num_classes, kernel_size=1)
        self.finalpre2 = nn.Conv2d(embed_dims[2], num_classes, kernel_size=1)
        self.finalpre3 = nn.Conv2d(embed_dims[1], num_classes, kernel_size=1)
        self.finalpre4 = nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)
        self.final = nn.Conv2d(embed_dims[-1], num_classes, kernel_size=1)


    def forward(self, x, inference_mode=False):
        B = x.shape[0]
        out = self.encoder1(x)
        out = F.relu(F.max_pool2d(self.ebn1(out), 2, 2))
        t1 = out

        out, H, W = self.patch_embed1(out)
        for i, blk in enumerate(self.block_0_1):
            out = blk(out, H, W)
        out = self.norm1(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        t2 = out
       
        out, H, W = self.patch_embed2(out)
        for i, blk in enumerate(self.block0):
            out = blk(out, H, W)
        out = self.norm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        t3 = out

        out, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        t4 = out

        out, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        t5 = out

        out, H, W = self.patch_embed5(out)
        for i, blk in enumerate(self.block3):
            out = blk(out, H, W)
        out = self.norm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        out = F.relu(F.interpolate(self.dbn0(self.decoder0(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t5)
        
        if not inference_mode:
            outtpre0 = F.interpolate(out, scale_factor=32, mode ='bilinear', align_corners=True)
            outtpre0 =self.finalpre0(outtpre0)
            
        
        _,_,H,W = out.shape
        
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock0):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm2(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
   
  
        out = torch.add(out,t4)
        if not inference_mode:
            outtpre1 = F.interpolate(out, scale_factor=16, mode ='bilinear', align_corners=True)
            outtpre1 =self.finalpre1(outtpre1)
            
        _,_,H,W = out.shape
        
        out = out.flatten(2).transpose(1,2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        ### Stage 3
        
        out = self.dnorm3(out)
        
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
       
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))

        out = torch.add(out,t3)
        if not inference_mode:
            outtpre2 = F.interpolate(out, scale_factor=8, mode ='bilinear', align_corners=True)   
            outtpre2 =self.finalpre2(outtpre2)

        
        _,_,H,W = out.shape
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
   
        out = torch.add(out,t2)

        if not inference_mode:
            outtpre3 = F.interpolate(out, scale_factor=4, mode ='bilinear', align_corners=True)
        
            outtpre3 =self.finalpre3(outtpre3)
      
        _,_,H,W = out.shape
        
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock3):
            out = blk(out, H, W)

        out = self.dnorm5(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)  
             
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
   
        out = torch.add(out,t1)

        
        if not inference_mode:
            outtpre4 = F.interpolate(out, scale_factor=2, mode ='bilinear', align_corners=True)
        
            outtpre4 =self.finalpre4(outtpre4)    
        
        _,_,H,W = out.shape
       
        out = out.flatten(2).transpose(1,2)
        
        for i, blk in enumerate(self.dblock4):
            out = blk(out, H, W) 
        
        out = self.dnorm6(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))

        out =  self.final(out)
        if not inference_mode:
            return ( outtpre0,outtpre1, outtpre2, outtpre3, outtpre4), out
        else:
            return out

# Function to count the number of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class InferenceModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(InferenceModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x, inference_mode=True)

def compute_gflops(model, input_size, device):
    model = model.to(device)
    input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Wrap the model
    wrapped_model = InferenceModelWrapper(model)
    
    with torch.no_grad():
        macs, params = profile(wrapped_model, inputs=(input, ), verbose=False)
    
    gflops = macs / (10**9)
    return gflops

if __name__ == "__main__":
    num_classes = 1
    input_channels = 3
    model = UCM_NetV2(num_classes=num_classes, input_channels=input_channels)
    model.cuda() 

    # Compute and print the number of trainable parameters
    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params}")

    # Example to compute GFLOPS; adjust the input size if needed
    input_size = 256
    gflops = compute_gflops(model, input_size,device=torch.device('cuda'))
    print(f"GFLOPS: {gflops:.4f}")

    # Measure FPS


    input_tensor = torch.randn(1, input_channels, input_size, input_size).cuda()
    # warm up
    with torch.no_grad():
        for _ in range(10):  # Run inference 100 times to get a stable FPS value
            model(input_tensor)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(500):  # Run inference 100 times to get a stable FPS value
            model(input_tensor)
    end_time = time.time()
    fps = 500 / (end_time - start_time)
    print(f"FPS: {fps:.4f}")
    
    
    num_classes = 1
    input_channels = 3
    model = UCM_NetV2(num_classes=num_classes, input_channels=input_channels)

    # Compute and print the number of trainable parameters
    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params}")

    model.cpu()

    input_tensor = torch.randn(1, input_channels, input_size, input_size)
    with torch.no_grad():
        for _ in range(10):  # Run inference 100 times to get a stable FPS value
            model(input_tensor)
    start_time = time.time()
    with torch.no_grad():
        for _ in range(500):  # Run inference 100 times to get a stable FPS value
            model(input_tensor)
    end_time = time.time()
    fps_cpu = 500 / (end_time - start_time)
    print(f"FPS (CPU): {fps_cpu:.2f}")
