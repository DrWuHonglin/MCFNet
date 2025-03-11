import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from FFM import FeatureFusionModule
from attention import Cross_Atten_Lite_split
from collections import OrderedDict


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)
    
    
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


class ResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet34, self).__init__()
        pretrained = torchvision.models.resnet34(pretrained=pretrained)

        for module_name in [
            "conv1",
            "bn1",
            "relu",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
        ]:
            self.add_module(module_name, getattr(pretrained, module_name))

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        b1 = self.layer1(x)
        b2 = self.layer2(b1)
        b3 = self.layer3(b2)
        b4 = self.layer4(b3)

        return b1, b2, b3, b4


class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(Encoder, self).__init__()
        self.rgb_backbone = ResNet34(pretrained=pretrained)
        self.dsm_backbone = ResNet34(pretrained=pretrained)
        self.ffm1 = FeatureFusionModule(in_channels=64, out_channels=64)
        self.ffm2 = FeatureFusionModule(in_channels=128, out_channels=128)
        self.ffm3 = FeatureFusionModule(in_channels=256, out_channels=256)
        self.ffm4 = FeatureFusionModule(in_channels=512, out_channels=512)
        self.att = Cross_Atten_Lite_split(inc1=512, inc2=512)

    def forward(self, rgb, dsm):
        # Process RGB and DSM images separately up to layer1
        rgb_b1, rgb_b2, rgb_b3, rgb_b4 = self.rgb_backbone(rgb)
        dsm_b1, dsm_b2, dsm_b3, dsm_b4 = self.dsm_backbone(dsm)

        # Feature fusion after layer1
        fused_b1 = self.ffm1(rgb_b1, dsm_b1)
        fused_b2_rgb = self.rgb_backbone.layer2(fused_b1)
        fused_b2_dsm = self.dsm_backbone.layer2(dsm_b1)

        # Feature fusion after layer2
        fused_b2 = self.ffm2(fused_b2_rgb, fused_b2_dsm)
        fused_b3_rgb = self.rgb_backbone.layer3(fused_b2)
        fused_b3_dsm = self.dsm_backbone.layer3(fused_b2_dsm)

        # Feature fusion after layer3
        fused_b3 = self.ffm3(fused_b3_rgb, fused_b3_dsm)
        fused_b4_rgb = self.rgb_backbone.layer4(fused_b3)
        fused_b4_dsm = self.dsm_backbone.layer4(fused_b3_dsm)

        # Feature fusion after layer4
        fused_b4 = self.ffm4(fused_b4_rgb, fused_b4_dsm)

        # attention
        fused_b4 = self.att(fused_b4, fused_b4_rgb, fused_b4_dsm)

        return [fused_b1, fused_b2, fused_b3, fused_b4]


class MCFNet(nn.Module):
    def __init__(self):
        super(MCFNet, self).__init__()
        self.init_DSM = nn.Conv2d(1, 3, 1)
        self.encoder = Encoder()
        self.decoder = DecoderHead()

    def forward(self, rgb, dsm):
        # b, c, h, w
        _, _, h, w = rgb.shape
        dsm = dsm.unsqueeze(1)
        dsm = self.init_DSM(dsm)
        output_enc = self.encoder(rgb, dsm)
        output_dec = self.decoder(output_enc)
        out = F.interpolate(output_dec, size=(h, w), mode='bilinear', align_corners=False)
        return out


if __name__ == '__main__':
    from thop import profile, clever_format
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb = torch.randn(1, 3, 512, 512)
    dsm = torch.randn(1, 512, 512)
    net = MCFNet()
    params_dict = dict(net.named_parameters())
    for key in params_dict.keys():
        print(key)
    flops, params = profile(net, inputs=(rgb, dsm))
    flops, params = clever_format([flops, params], "%.2f")
    print(f"Params: {params}")
    print(f"FLOPS: {flops}")
    out = net(rgb, dsm)
    print(out.shape)

