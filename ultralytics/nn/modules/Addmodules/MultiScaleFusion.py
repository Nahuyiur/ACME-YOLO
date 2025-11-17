import torch
import torch.nn as nn

__all__ = ['MultiScaleFusion']

def get_padding_size(kernel_size, padding=None, dilation=1):
    if dilation > 1:
        if isinstance(kernel_size, int):
            kernel_size = dilation * (kernel_size - 1) + 1
        else:
            kernel_size = [dilation * (x - 1) + 1 for x in kernel_size]
    
    if padding is None:
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = [x // 2 for x in kernel_size]
    
    return padding

class ConvolutionBlock(nn.Module):
    default_act = nn.SiLU()
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        padding = get_padding_size(k, p, d)
        self.conv = nn.Conv2d(c1, c2, k, s, padding, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        
        if act is True:
            self.act = self.default_act
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()
    
    def forward(self, x):
        conv_out = self.conv(x)
        bn_out = self.bn(conv_out)
        return self.act(bn_out)
    
    def forward_fuse(self, x):
        conv_out = self.conv(x)
        return self.act(conv_out)

class SpatialPooling(nn.Module):
    def __init__(self, k=3, s=1):
        super(SpatialPooling, self).__init__()
        padding = k // 2
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=padding)
    
    def forward(self, x):
        return self.m(x)

class MultiScaleFusion(nn.Module):
    def __init__(self, c1, c2, c3, pool_size=5):
        super().__init__()
        self.c = c3
        self.layers = nn.ModuleList([
            ConvolutionBlock(c1, c3, 1, 1),
            SpatialPooling(pool_size),
            SpatialPooling(pool_size),
            SpatialPooling(pool_size)
        ])
        self.output_layer = ConvolutionBlock(4*c3, c2, 1, 1)
    
    def forward(self, x):
        feature_maps = []
        data = x
        for i, layer in enumerate(self.layers):
            data = layer(data)
            feature_maps.append(data)
        merged_features = torch.cat(feature_maps, dim=1)
        return self.output_layer(merged_features)