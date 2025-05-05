import torch
from torch import nn
from thop import profile, clever_format
import torch.nn.functional as F


def channel_shuffle(x, groups: int):
    batchsize, N, num_channels = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, N, groups, channels_per_group)

    # Transpose operation is not valid for 5D tensor, so we need to use permute
    x = x.permute(0, 1, 3, 2).contiguous()

    # flatten
    x = x.view(batchsize, N, -1)

    return x


class Conv2dBnRelu(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2dBnRelu, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MEM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MEM, self).__init__()

        # self.branch1 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        # )

        self.mid = nn.Sequential(
            Conv2dBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        )

        self.down1 = Conv2dBnRelu(in_ch, 1, kernel_size=7, stride=2, padding=3)

        self.down2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=2, padding=2)

        self.down3 = nn.Sequential(
            Conv2dBnRelu(1, 1, kernel_size=3, stride=2, padding=1),
            Conv2dBnRelu(1, 1, kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = Conv2dBnRelu(1, 1, kernel_size=5, stride=1, padding=2)
        self.conv1 = Conv2dBnRelu(1, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = x
        # b1 = F.interpolate(b1, size=(h, w), mode='bilinear', align_corners=True)

        mid = self.mid(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = F.interpolate(x3, size=(h // 4, w // 4), mode='bilinear', align_corners=True)

        x2 = self.conv2(x2)
        x = x2 + x3
        x = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True)

        x1 = self.conv1(x1)
        x = x + x1
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        x = torch.mul(x, mid)
        x = x + b1
        return x


class Cross_Atten_Lite_split(nn.Module):
    def __init__(self, inc1, inc2):
        super(Cross_Atten_Lite_split, self).__init__()
        self.midc1 = torch.tensor(inc1 // 4)
        self.midc2 = torch.tensor(inc2 // 4)

        self.bn_x1 = nn.BatchNorm2d(inc1)
        self.bn_x2 = nn.BatchNorm2d(inc2)

        self.kq1 = nn.Linear(inc1, self.midc2 * 2)
        self.kq2 = nn.Linear(inc2, self.midc2 * 2)

        self.v_conv = nn.Linear(inc1, 2 * self.midc1)
        self.out_conv = nn.Linear(2 * self.midc1, inc1)

        self.bn_last = nn.BatchNorm2d(inc1)
        self.dropout = nn.Dropout(0.2)

        # Add MEM for multi-scale feature extraction for K, Q and V
        self.mem_kq = MEM(inc1 // 4, inc1 // 4)
        self.mem_v = MEM(inc1 // 4, inc1 // 4)

        self.w = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w.data.fill_(0)

        self._init_weight()

    def forward(self, x, x1, x2):
        MC = False
        batch_size = x.size(0)
        h = x.size(2)
        w = x.size(3)

        x1 = self.bn_x1(x1)
        x2 = self.bn_x2(x2)

        kq1 = self.kq1(x1.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        kq2 = self.kq2(x2.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        kq = channel_shuffle(torch.cat([kq1, kq2], dim=2), 2)
        k1, q1, k2, q2 = torch.split(kq, self.midc2, dim=2)

        if MC:
            # Reshape K and Q to (batch_size, c // 4, h, w)
            k1 = k1.permute(0, 2, 1).view(batch_size, -1, h, w)
            k2 = k2.permute(0, 2, 1).view(batch_size, -1, h, w)
            q1 = q1.permute(0, 2, 1).view(batch_size, -1, h, w)
            q2 = q2.permute(0, 2, 1).view(batch_size, -1, h, w)

            # Apply multi-scale module to K and Q
            k1 = self.mem_kq(k1).permute(0, 2, 3, 1).view(batch_size, h * w, -1)
            k2 = self.mem_kq(k2).permute(0, 2, 3, 1).view(batch_size, h * w, -1)
            q1 = self.mem_kq(q1).permute(0, 2, 3, 1).view(batch_size, h * w, -1)
            q2 = self.mem_kq(q2).permute(0, 2, 3, 1).view(batch_size, h * w, -1)

        v = self.v_conv(x.permute(0, 2, 3, 1).view(batch_size, h * w, -1))
        v1, v2 = torch.split(v, self.midc1, dim=2)

        if MC:
            # Reshape V to (batch_size, c // 4, h, w)
            v1 = v1.permute(0, 2, 1).view(batch_size, -1, h, w)
            v2 = v2.permute(0, 2, 1).view(batch_size, -1, h, w)

            # Apply multi-scale module to V
            v1 = self.mem_v(v1).permute(0, 2, 3, 1).view(batch_size, h * w, -1)
            v2 = self.mem_v(v2).permute(0, 2, 3, 1).view(batch_size, h * w, -1)

        mat = torch.matmul(q1, k1.permute(0, 2, 1))
        mat = mat / torch.sqrt(self.midc2)
        mat = nn.Softmax(dim=-1)(mat)
        mat = self.dropout(mat)
        v1 = torch.matmul(mat, v1)

        mat = torch.matmul(q2, k2.permute(0, 2, 1))
        mat = mat / torch.sqrt(self.midc2)
        mat = nn.Softmax(dim=-1)(mat)
        mat = self.dropout(mat)
        v2 = torch.matmul(mat, v2)

        v = torch.cat([v1, v2], dim=2).view(batch_size, h, w, -1)
        v = self.out_conv(v)
        v = v.permute(0, 3, 1, 2)
        v = self.bn_last(v)

        v = self.w * v + x

        return v

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)


if __name__ == "__main__":
    in_channels, out_channels = 512, 512
    x1 = torch.randn(1, in_channels, 16, 16)
    x2 = torch.randn(1, in_channels, 16, 16)
    ffm = Cross_Atten_Lite_split(512, 512)
    out = ffm(x1 + x2, x1, x2)
    print(out.shape)
