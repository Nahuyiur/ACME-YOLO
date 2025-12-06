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

class MultiScaleFusion(nn.Module):
    """
    Adaptive parallel multi-scale fusion module with residual branches and 
    ultra-lightweight learnable branch weighting for small object detection.
    Each branch preserves fine details via residual connection, and branch
    contributions are dynamically balanced through scalar weights.
    """
    def __init__(self, c1, c2, c3, dilations=(1, 2, 3)):
        """
        Args:
            c1 (int): input channels
            c2 (int): output channels
            c3 (int): intermediate channel width for each branch
            dilations (tuple[int]): dilation factors for the parallel branches
        """
        super().__init__()
        self.reduce = ConvolutionBlock(c1, c3, 1, 1)
        self.num_branches = len(dilations) + 1  # +1 for identity

        branches = []
        for d in dilations:
            padding = get_padding_size(3, None, d)
            branches.append(
                nn.Sequential(
                    nn.Conv2d(
                        c3,
                        c3,
                        kernel_size=3,
                        stride=1,
                        padding=padding,
                        dilation=d,
                        groups=c3,
                        bias=False,
                    ),
                    nn.BatchNorm2d(c3),
                    nn.SiLU(inplace=True),
                )
            )
        self.branches = nn.ModuleList(branches)
        
        # Ultra-lightweight branch weighting: only num_branches scalar parameters
        self.branch_weights = nn.Parameter(torch.ones(self.num_branches))
        
        self.output_layer = ConvolutionBlock(c3 * self.num_branches, c2, 1, 1)
    
    def forward(self, x):
        reduced = self.reduce(x)
        
        # Normalize weights via softmax for stable training
        weights = torch.softmax(self.branch_weights, dim=0)
        
        # Identity branch (index 0)
        feature_maps = [weights[0] * reduced]
        
        # Dilated branches
        for i, branch in enumerate(self.branches):
            feature_maps.append(weights[i + 1] * branch(reduced))
        
        merged_features = torch.cat(feature_maps, dim=1)
        return self.output_layer(merged_features)