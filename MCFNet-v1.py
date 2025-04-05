import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from FFM import FeatureFusionModule
from attention import Cross_Atten_Lite_split
from torchvision.models import ResNet50_Weights, ResNet34_Weights
from torchvision import models

__all__ = ["ResNet34", "ResNet50"]


def ResNet34():
    resnet34 = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    return resnet34


def ResNet50():
    resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    return resnet50


class MLP(nn.Module):
    """
    Linear Embedding:
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=None,
                 num_classes=6,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):

        super(DecoderHead, self).__init__()
        if in_channels is None:
            in_channels = [64, 128, 256, 512]
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs

        # ############# MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

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


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(Conv2d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation=dilation, bias=bias)
        )

    def forward(self, x):
        return self.conv(x)


class MCFNet(nn.Module):
    def __init__(self):
        super(MCFNet, self).__init__()
        resnet34 = ResNet34()
        self.dsm_init = nn.Conv2d(1, 3, 1)

        # Replace 7 * 7 convolution with three 3 * 3 convolutions
        conv1 = nn.Sequential(
            Conv2dBnRelu(3, 64, kernel_size=3, stride=2, padding=1),
            Conv2dBnRelu(64, 64, kernel_size=3, stride=1, padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        )

        self.rgb_conv1 = conv1
        self.dsm_conv1 = copy.deepcopy(conv1)

        self.rgb_bn1 = resnet34.bn1
        self.dsm_bn1 = copy.deepcopy(resnet34.bn1)

        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool

        self.rgb_layer1 = resnet34.layer1
        self.dsm_layer1 = copy.deepcopy(resnet34.layer1)

        self.rgb_layer2 = resnet34.layer2
        self.dsm_layer2 = copy.deepcopy(resnet34.layer2)

        self.rgb_layer3 = resnet34.layer3
        self.dsm_layer3 = copy.deepcopy(resnet34.layer3)

        self.rgb_layer4 = resnet34.layer4
        self.dsm_layer4 = copy.deepcopy(resnet34.layer4)

        # FFM
        self.ffm1 = FeatureFusionModule(64, 64)
        self.ffm2 = FeatureFusionModule(128, 128)
        self.ffm3 = FeatureFusionModule(256, 256)
        self.ffm4 = FeatureFusionModule(512, 512)

        # MCFCM
        self.mcfcm = Cross_Atten_Lite_split(512, 512)

        #Decoder
        self.decoder = DecoderHead()

    def forward(self, x, y):
        _, _, h, w = x.shape
        SE, MC = True, True
        features = []

        y = y.unsqueeze(1)
        y = self.dsm_init(y)

        x0 = self.relu(self.rgb_bn1(self.rgb_conv1(x)))
        y0 = self.relu(self.dsm_bn1(self.dsm_conv1(y)))

        x = self.maxpool(x0)
        y = self.maxpool(y0)

        x1 = self.rgb_layer1(x)
        y1 = self.dsm_layer1(y)
        if SE:
            fusion1 = self.ffm1(x1, y1)
        else:
            fusion1 = x1 + y1
        features.append(fusion1)

        x2 = self.rgb_layer2(fusion1)
        y2 = self.dsm_layer2(y1)
        if SE:
            fusion2 = self.ffm2(x2, y2)
        else:
            fusion2 = x2 + y2
        features.append(fusion2)

        x3 = self.rgb_layer3(fusion2)
        y3 = self.dsm_layer3(y2)
        if SE:
            fusion3 = self.ffm3(x3, y3)
        else:
            fusion3 = x3 + y3
        features.append(fusion3)

        x4 = self.rgb_layer4(fusion3)
        y4 = self.dsm_layer4(y3)
        if SE:
            fusion4 = self.ffm4(x4, y4)
        else:
            fusion4 = x4 + y4
        if MC:
            fusion4 = self.mcfcm(fusion4, x4, y4)
        features.append(fusion4)

        out_dec = self.decoder(features)
        out = F.interpolate(out_dec, size=(h, w), mode='bilinear', align_corners=False)
        return out


if __name__ == '__main__':
    from thop import profile, clever_format
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 512, 512)
    y = torch.randn(1, 512, 512)
    net = MCFNet()
    params_dict = dict(net.named_parameters())
    for key in params_dict.keys():
        print(key)
    flops, params = profile(net, inputs=(x, y))
    flops, params = clever_format([flops, params], "%.2f")
    print(f"Params: {params}")
    print(f"FLOPS: {flops}")
    out = net(x, y)
    print(out.shape)
