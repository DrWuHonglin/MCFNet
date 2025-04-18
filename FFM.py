import torch
import torch.nn as nn
from thop import profile, clever_format


class PoolAttention(nn.Module):
    def __init__(self, dim):
        super(PoolAttention, self).__init__()
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv2d(dim, dim // 2, kernel_size=3, dilation=2, stride=1, padding=2)
        self.conv2 = nn.Conv2d(dim // 2, dim, kernel_size=1, stride=1, padding=0)
        self.avg_pool_w, self.avg_pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        self.max_pool_w, self.max_pool_h = nn.AdaptiveMaxPool2d((1, None)), nn.AdaptiveMaxPool2d((None, 1))
        self.bn1 = nn.BatchNorm2d(dim // 2)
        self.bn2 = nn.BatchNorm2d(dim)
        
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)

    def forward(self, x):
        residual = x
        x = self.gelu(self.bn1(self.conv1(x)))
        x_avg_h, x_avg_w = self.avg_pool_h(x), self.avg_pool_w(x)
        x_avg = torch.matmul(x_avg_h, x_avg_w)
        x_max_h, x_max_w = self.max_pool_h(x), self.max_pool_w(x)
        x_max = torch.matmul(x_max_h, x_max_w)
        x = self.w1 * x_avg + self.w2 * x_max
        x = self.gelu(self.bn2(self.conv2(x)))
        return residual + x

    
class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.pool_rgb = PoolAttention(dim=in_channels)
        self.pool_dsm = PoolAttention(dim=in_channels)

    def forward(self, x1, x2):
        x1 = self.pool_rgb(x1)
        x2 = self.pool_dsm(x2)
        return x1 + x2

# Example usage
if __name__ == "__main__":
    from thop import profile, clever_format
    in_channels, out_channels = 256, 256
    x1 = torch.randn(1, in_channels, 32, 32)
    x2 = torch.randn(1, in_channels, 32, 32)
    ffm = FeatureFusionModule(in_channels=in_channels, out_channels=out_channels)
    out = ffm(x_rgb, x_dsm)
    print(out.shape)
    
