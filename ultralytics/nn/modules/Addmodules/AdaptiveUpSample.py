import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['AdaptiveUpSample']

def init_weights_normal(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def init_weights_constant(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class AdaptiveUpSample(nn.Module):
    def __init__(self, in_channels, scale=2, groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.groups = groups
        assert in_channels >= groups and in_channels % groups == 0
        out_channels = 2 * groups * scale ** 2
        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        init_weights_normal(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1)
            init_weights_constant(self.scope, val=0.)
        init_pos = self._init_pos()
        self.register_buffer('init_pos', init_pos)
    
    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        mesh = torch.meshgrid([h, h])
        stacked = torch.stack(mesh)
        transposed = stacked.transpose(1, 2)
        repeated = transposed.repeat(1, self.groups, 1)
        return repeated.reshape(1, -1, 1, 1)
    
    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        mesh = torch.meshgrid([coords_w, coords_h])
        coords = torch.stack(mesh)
        coords = coords.transpose(1, 2)
        coords = coords.unsqueeze(1).unsqueeze(0)
        coords = coords.type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device)
        normalizer = normalizer.view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = coords.view(B, -1, H, W)
        coords = F.pixel_shuffle(coords, self.scale)
        coords = coords.view(B, 2, -1, self.scale * H, self.scale * W)
        coords = coords.permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        x_reshaped = x.reshape(B * self.groups, -1, H, W)
        result = F.grid_sample(x_reshaped, coords, mode='bilinear',
                             align_corners=False, padding_mode="border")
        return result.view(B, -1, self.scale * H, self.scale * W)
    
    def forward(self, x):
        offset_base = self.offset(x)
        if hasattr(self, 'scope'):
            scope_sigmoid = self.scope(x).sigmoid()
            scope_weighted = scope_sigmoid * 0.5
            offset = offset_base * (scope_weighted + self.init_pos)
        else:
            offset = offset_base * 0.25 + self.init_pos
        return self.sample(x, offset)

if __name__ == '__main__':
    x = torch.rand(2, 64, 4, 7)
    dys = AdaptiveUpSample(64)
    print(dys(x).shape)