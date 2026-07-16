# BTSCD.py
from torchvision import models
import time
from models.layers import *

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------
# DiffFormer implementation
# -----------------------
class DiffFormer(nn.Module):
    """
    Difference-Token Transformer:
    - 输入: x1, x2 (高层特征)，shape [B, C, H, W]
    - 输出: xc_map, shape [B, C, H, W] （与输入高层特征通道数一致）
    """
    def __init__(self,
                 in_ch=128,
                 token_dim=128,
                 num_tokens=16,
                 nheads=8,
                 nlayers=3,
                 patch_size=8,
                 dropout=0.0):
        super(DiffFormer, self).__init__()
        self.in_ch = in_ch
        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.patch_size = patch_size

        # Project input channels to token_dim
        self.proj = nn.Conv2d(in_ch, token_dim, kernel_size=1, bias=False)

        # LayerNorm for patch tokens (applied on last dim)
        self.patch_norm = nn.LayerNorm(token_dim)

        # learnable difference tokens
        self.token_param = nn.Parameter(torch.randn(1, num_tokens, token_dim))

        # positional embedding for patches (created lazily)
        self.pos_embed = None

        # Transformer encoder (batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim,
                                                   nhead=nheads,
                                                   dim_feedforward=token_dim * 4,
                                                   dropout=dropout,
                                                   activation='gelu',
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # map tokens -> channel map
        self.token_to_map = nn.Linear(token_dim, in_ch)

        # optional final conv to refine the xc_map
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )

    def _pad_to_multiple(self, x, patch_size):
        """
        Pads spatial dims so H and W are divisible by patch_size.
        Returns padded tensor and padding info (pad_h, pad_w).
        """
        B, C, H, W = x.shape
        pad_h = (patch_size - (H % patch_size)) % patch_size
        pad_w = (patch_size - (W % patch_size)) % patch_size
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h))  # (left,right,top,bottom)
        return x, (pad_h, pad_w)

    def forward(self, x1, x2, xc_cst=None):
        """
        x1, x2: [B, C, H, W]  (C == in_ch)
        xc_cst: optional change_specific_transfer output for possible conditioning (not used directly here)
        returns: xc_map [B, in_ch, H, W]
        """
        B, C, H, W = x1.shape
        assert C == self.in_ch, f"DiffFormer expected in_ch={self.in_ch}, got {C}"

        # project
        z1 = self.proj(x1)  # [B, td, H, W]
        z2 = self.proj(x2)

        # pad to patch multiple
        z1, (pad_h, pad_w) = self._pad_to_multiple(z1, self.patch_size)
        z2, _ = self._pad_to_multiple(z2, self.patch_size)  # same pad

        _, td, Hp, Wp = z1.shape
        p = self.patch_size

        # Patch embedding via unfold: produces patches of shape [B, td * p*p, N]
        patches1 = F.unfold(z1, kernel_size=p, stride=p)  # [B, td * p*p, N]
        patches2 = F.unfold(z2, kernel_size=p, stride=p)

        # shape to [B, N, td] by mean pooling within patch
        Bf, Cpatch, N = patches1.shape
        patches1 = patches1.view(Bf, td, p*p, N).mean(dim=2).permute(0, 2, 1)  # [B, N, td]
        patches2 = patches2.view(Bf, td, p*p, N).mean(dim=2).permute(0, 2, 1)  # [B, N, td]

        # LayerNorm over last dim
        patches1 = self.patch_norm(patches1)
        patches2 = self.patch_norm(patches2)

        # difference patches
        diff_p = patches1 - patches2  # [B, N, td]

        # prepare tokens
        tokens = self.token_param.expand(B, -1, -1)  # [B, K, td]

        # create pos_embed if needed
        Np = diff_p.size(1)
        if (self.pos_embed is None) or (self.pos_embed.shape[1] != Np):
            self.pos_embed = torch.zeros(1, Np, self.token_dim, device=diff_p.device)
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        diff_p = diff_p + self.pos_embed  # add pos embed

        # concat tokens + diff_p
        seq = torch.cat([tokens, diff_p], dim=1)  # [B, K+N, td]

        # transformer
        trans_out = self.transformer(seq)  # [B, K+N, td]

        # take the K token outputs (global diff tokens)
        out_tokens = trans_out[:, :self.num_tokens, :]  # [B, K, td]

        # global representation
        tokens_mean = out_tokens.mean(dim=1)  # [B, td]
        map_ch = self.token_to_map(tokens_mean)  # [B, in_ch]

        # make spatial map and upsample to padded spatial size
        token_map = map_ch.unsqueeze(-1).unsqueeze(-1)  # [B, in_ch, 1, 1]
        token_map_up = F.interpolate(token_map, size=(Hp, Wp), mode='bilinear', align_corners=False)  # [B, in_ch, Hp, Wp]

        # unpad if necessary to original spatial H,W
        if pad_h != 0 or pad_w != 0:
            token_map_up = token_map_up[..., :H, :W]

        # refine
        xc_map = self.refine(token_map_up)  # [B, in_ch, H, W]

        return xc_map


# -----------------------
# Original FCN and BTSCD (with DiffFormer integrated)
# -----------------------
class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.mlfa = Multi_Level_Feature_Aggreagation()

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)  # size:1/2
        x = self.maxpool(x)  # size:1/4
        x_low = self.layer1(x)  # size:1/4
        x1 = self.layer2(x_low)  # size:1/8
        x2 = self.layer3(x1)
        x3 = self.layer4(x2)
        x = self.mlfa(x1, x2, x3)
        return x, x_low

class BTSCD(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(BTSCD, self).__init__()
        self.FCN = FCN(in_channels, pretrained=True)

        # self.change_specific_transfer = Change_Specific_Transfer(128) # BCFE

        self.DecCD = decoder(128, 64)
        self.Dec1 = decoder(128, 64)
        self.Dec2 = decoder(128, 64)

        # self.task_interaction = task_interaction_module()

        self.classifierSem1 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierSem2 = nn.Conv2d(64, num_classes, 1, 1, 0, bias=False)
        self.classifierCD = nn.Conv2d(64, 2, 1, 1, 0, bias=False)

        # self.boundary_decoder = Boundary_Decoder()
        # self.eca = ECA()
        # self.boundary_classifier = nn.Sequential(
        #     CBA3x3(64, 32),
        #     nn.Conv2d(32, 1, 1, 1, 0),
        #     nn.Sigmoid()
        # )

        self.diff_former = DiffFormer(in_ch=128)

        # 方案1：差异特征扩展 + 融合 + 通道压缩
        self.diff_expand = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0),  # 128→256通道
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.fuse_compress = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),  # 256→128通道
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        x_size = x1.size()

        x1, x1_low = self.FCN(x1)
        x2, x2_low = self.FCN(x2)

        # xc = self.change_specific_transfer(x1, x2)

        xc = self.diff_former(x1, x2)

        # x_diff = torch.abs(x1 - x2)  # 差异特征（突出变化区域）
        # x_concat = torch.cat([x1, x2], dim=1)  # 拼接特征（保留原始语义信息）
        #
        # # 调用预定义的层（自动继承模型的设备，GPU/CPU 一致）
        # x_diff_expand = self.diff_expand(x_diff)  # 128→256通道，GPU张量
        # x_fuse = x_concat + x_diff_expand  # 维度一致（256通道），可正常相加
        # xc = self.fuse_compress(x_fuse)  # 256→128通道


        x1 = self.Dec1(x1, x1_low)
        x2 = self.Dec2(x2, x2_low)

        xc_low = torch.abs(x1 - x2)
        xc = self.DecCD(xc, xc_low)

        #Classifier
        # change = self.classifierCD(xc)
        # new_xc, pixel_sim_loss = self.task_interaction(x1, x2, xc, change)
        new_xc = xc

        out1 = self.classifierSem1(x1)
        out2 = self.classifierSem2(x2)
        new_change = self.classifierCD(new_xc)

        out1 = F.interpolate(out1, x_size[2:], mode='bilinear')
        out2 = F.interpolate(out2, x_size[2:], mode='bilinear')
        change_out = F.interpolate(new_change, x_size[2:], mode='bilinear')

        # boundary_x1 = self.boundary_decoder(x1, x_size[2:])
        # boundary_x2 = self.boundary_decoder(x2, x_size[2:])
        # boundary_change = self.boundary_decoder(new_xc, x_size[2:])
        #
        # boundary_sem = self.eca(boundary_x1 + boundary_x2)
        # boundary_sem = self.boundary_classifier(boundary_sem)
        # boundary_change = self.boundary_classifier(boundary_change)

        return change_out, out1, out2
        # return change_out, out1, out2, pixel_sim_loss, boundary_sem, boundary_change


if __name__ == '__main__':
    x1 = torch.randn(1, 3, 512, 512).cuda().float()
    x2 = torch.randn(1, 3, 512, 512).cuda().float()

    model = BTSCD(3, num_classes=7).cuda()
    model.eval()  # 将模型设置为推理模式
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, (x1, x2))
    total = sum([param.nelement() for param in model.parameters()])
    print("Params_Num: %.2fM" % (total/1e6))
    print("FLOPs: %.2fG" % (flops.total()/1e9))

    with torch.no_grad():
        for _ in range(10):
            _ = model(x1, x2)

    # 正式计时
    start_time = time.time()
    with torch.no_grad():
        output = model(x1, x2)
    end_time = time.time()

    inference_time = end_time - start_time
    print(f"Inference time: {inference_time * 1000:.2f} ms")
