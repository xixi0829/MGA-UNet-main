import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.layers import trunc_normal_
from mamba_ssm import Mamba



class A_CDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(A_CDC, self).__init__()

        # 1. 基础卷积 (提取语义)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)

        # 2. Theta 生成器 (空间注意力)
        self.theta_learner = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # 初始化 theta_learner
        # 初始 bias设为 -2.0，Sigmoid(-2.0) ≈ 0.12，避免初始噪声过大
        nn.init.constant_(self.theta_learner[0].weight, 0)
        nn.init.constant_(self.theta_learner[0].bias, -2.0)

    def forward(self, x):
        # [Step 1] 标准卷积
        out_normal = self.conv(x)

        # [Step 2] 计算中心差分项 (Gradient Feature)
        # 计算核权重之和
        kernel_sum = self.conv.weight.sum(dim=[2, 3], keepdim=True)

        # [CRITICAL FIX] 计算局部均值项时，必须设 bias=None
        # 否则：out_normal (Wx+b) - out_center (SumW*x + b) 会导致 b 被抵消
        # 或者：(Wx+b) + (SumW*x + b) 会导致 b 被加倍
        out_center_term = F.conv2d(input=x, weight=kernel_sum, bias=None,
                                   stride=self.conv.stride, padding=0, groups=self.conv.groups)

        # 边缘特征 = 原始特征 - 局部均值 (High Frequency)
        edge_feature = out_normal - out_center_term

        # [Step 3] 空间自适应融合
        # 结果 = 原始语义 + (注意力图 * 边缘纹理)
        theta_map = self.theta_learner(x)
        return out_normal + theta_map * edge_feature


# ==============================================================================
#        Inverted Residual Block (保持不变)
# ==============================================================================
class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, expand_ratio=4):
        super(InvertedResidual, self).__init__()
        hidden_dim = in_planes * expand_ratio
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_planes, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1,
                      groups=hidden_dim, bias=False),
            nn.GroupNorm(4, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, out_planes, kernel_size=1, bias=False),
            nn.GroupNorm(4, out_planes),
        )

    def forward(self, x):
        return self.bottleneck(x)


# ==============================================================================
#        Attention Modules (CBAM - 保持不变)
# ==============================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, max(1, in_planes // ratio), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(1, in_planes // ratio), in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# ==============================================================================
#        G-PPM (保持不变)
# ==============================================================================
class AG_PPM(nn.Module):
    def __init__(self, in_planes):
        super(AG_PPM, self).__init__()
        self.pyramid_scales = [1, 2, 4]
        self.out_planes = in_planes
        internal_planes = in_planes // 4

        self.grad_att = nn.Sequential(
            A_CDC(in_planes, in_planes, kernel_size=3, padding=1, groups=in_planes, bias=False),
            nn.GroupNorm(4, in_planes),
            nn.Sigmoid()
        )

        self.pool_branches = nn.ModuleList()
        for scale in self.pyramid_scales:
            self.pool_branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_planes, internal_planes, kernel_size=1, bias=False),
                nn.GroupNorm(4, internal_planes),
                nn.GELU()
            ))

        self.fusion = nn.Sequential(
            nn.Conv2d(in_planes + len(self.pyramid_scales) * internal_planes, self.out_planes, kernel_size=1,
                      bias=False),
            nn.GroupNorm(4, self.out_planes),
            nn.GELU()
        )

    def forward(self, x):
        input_size = x.shape[2:]
        att_map = self.grad_att(x)
        x_guided = x * (1 + att_map)
        res = [x]
        for branch in self.pool_branches:
            pooled = branch(x_guided)
            upsampled = F.interpolate(pooled, size=input_size, mode='bilinear', align_corners=True)
            res.append(upsampled)
        out = torch.cat(res, dim=1)
        out = self.fusion(out)
        return out


# ==============================================================================
#        [FIXED] MambaCDCBlock
#        修复说明：将 GELU 移至中间，确保残差分支可以进行正负双向调节
# ==============================================================================
class MambaCDCBlock(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2, num_parallel=4):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_parallel = num_parallel

        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // self.num_parallel,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))


        self.cnn_enhancer = nn.Sequential(
            A_CDC(output_dim, output_dim, kernel_size=3, padding=1,
                   groups=output_dim, bias=False),
            nn.GELU(),  # 激活函数放中间
            nn.Conv2d(output_dim, output_dim, kernel_size=1, bias=False)  # 最后一层保持线性
        )

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]

        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        chunks = torch.chunk(x_norm, self.num_parallel, dim=2)
        processed_chunks = []
        for chunk in chunks:
            mamba_out = self.mamba(chunk) + self.skip_scale * chunk
            processed_chunks.append(mamba_out)
        x_mamba = torch.cat(processed_chunks, dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)

        # Mamba + SA-CDC 双流融合
        cnn_out = self.cnn_enhancer(out)
        final_out = out + cnn_out
        return final_out



class MGA_UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64],
                 bridge=True, **kwargs):
        super().__init__()
        self.bridge = bridge
        shallow_expand_ratio = 4

        self.encoder1 = InvertedResidual(input_channels, c_list[0], expand_ratio=shallow_expand_ratio)
        self.encoder2 = InvertedResidual(c_list[0], c_list[1], expand_ratio=shallow_expand_ratio)
        self.encoder3 = InvertedResidual(c_list[1], c_list[2], expand_ratio=shallow_expand_ratio)

        self.encoder4 = nn.Sequential(MambaCDCBlock(input_dim=c_list[2], output_dim=c_list[3]))
        self.encoder5 = nn.Sequential(MambaCDCBlock(input_dim=c_list[3], output_dim=c_list[4]))
        self.encoder6 = nn.Sequential(MambaCDCBlock(input_dim=c_list[4], output_dim=c_list[5]))

        self.bottleneck = AG_PPM(in_planes=c_list[5])

        if bridge:
            self.cbam1 = CBAM(c_list[0], ratio=max(1, c_list[0] // 4), kernel_size=7)
            self.cbam2 = CBAM(c_list[1], ratio=max(1, c_list[1] // 4), kernel_size=7)
            self.cbam3 = CBAM(c_list[2], ratio=max(1, c_list[2] // 4), kernel_size=7)
            self.cbam4 = CBAM(c_list[3], ratio=max(1, c_list[3] // 4), kernel_size=7)
            self.cbam5 = CBAM(c_list[4], ratio=max(1, c_list[4] // 4), kernel_size=7)
            self.cbam6 = CBAM(c_list[5], ratio=max(1, c_list[5] // 4), kernel_size=7)

        self.decoder1 = nn.Sequential(MambaCDCBlock(input_dim=c_list[5], output_dim=c_list[5]))
        self.decoder2 = nn.Sequential(MambaCDCBlock(input_dim=c_list[5], output_dim=c_list[4]))
        self.decoder3 = nn.Sequential(MambaCDCBlock(input_dim=c_list[4], output_dim=c_list[3]))

        self.decoder4 = InvertedResidual(c_list[3], c_list[2], expand_ratio=shallow_expand_ratio)
        self.decoder5 = InvertedResidual(c_list[2], c_list[1], expand_ratio=shallow_expand_ratio)
        self.decoder6 = InvertedResidual(c_list[1], c_list[0], expand_ratio=shallow_expand_ratio)

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        # 执行初始化
        self.apply(self._init_weights)


        for m in self.modules():
            if isinstance(m, MambaCDCBlock):
                last_conv = m.cnn_enhancer[2]  # 索引变为 2
                nn.init.constant_(last_conv.weight, 0)
                if last_conv.bias is not None:
                    nn.init.constant_(last_conv.bias, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, (nn.Conv2d, A_CDC)):
            if isinstance(m, A_CDC):
                m = m.conv  # 处理 A_CDC 内部的卷积
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):


        # Stage 1
        x1 = self.encoder1(x)
        t1 = x1  # [关键] 在 Pooling 之前保存全分辨率特征
        out = F.gelu(F.max_pool2d(x1, 2, 2))

        # Stage 2
        x2 = self.encoder2(out)
        t2 = x2  # [关键] 保存特征
        out = F.gelu(F.max_pool2d(x2, 2, 2))

        # Stage 3
        x3 = self.encoder3(out)
        t3 = x3
        out = F.gelu(F.max_pool2d(x3, 2, 2))

        # Stage 4 (Mamba Block 通常输入输出同尺寸，这里用 MaxPool 下采样)
        x4 = self.encoder4(out)
        t4 = x4
        out = F.max_pool2d(x4, 2, 2)

        # Stage 5
        x5 = self.encoder5(out)
        t5 = x5
        out = F.max_pool2d(x5, 2, 2)

        # Stage 6
        x6 = self.encoder6(out)
        t6 = x6
        out = F.max_pool2d(x6, 2, 2)

        # ==========================================
        # Bridge
        # ==========================================
        if self.bridge:
            t1 = self.cbam1(t1)
            t2 = self.cbam2(t2)
            t3 = self.cbam3(t3)
            t4 = self.cbam4(t4)
            t5 = self.cbam5(t5)
            t6 = self.cbam6(t6)

        # ==========================================
        # Bottleneck
        # ==========================================
        out = self.bottleneck(t6)

        # ==========================================
        # Decoder
        # ==========================================
        def align_and_add(decoder_feat, encoder_feat):
            target_size = encoder_feat.size()[2:]
            if decoder_feat.size()[2:] != target_size:
                decoder_feat = F.interpolate(decoder_feat, size=target_size, mode='bilinear', align_corners=True)
            return torch.add(decoder_feat, encoder_feat)

        out6 = self.decoder1(out)
        out6 = align_and_add(out6, t6)

        out5 = self.decoder2(out6)
        # 此时 t5 是 t6 的 2倍分辨率，decoder 输出需要上采样
        out5 = F.interpolate(out5, size=t5.shape[2:], mode='bilinear', align_corners=True)
        out5 = torch.add(out5, t5)

        out4 = self.decoder3(out5)
        out4 = F.interpolate(out4, size=t4.shape[2:], mode='bilinear', align_corners=True)
        out4 = torch.add(out4, t4)

        out3 = F.interpolate(self.decoder4(out4), size=t3.shape[2:], mode='bilinear', align_corners=True)
        out3 = F.gelu(out3)
        out3 = torch.add(out3, t3)

        out2 = F.interpolate(self.decoder5(out3), size=t2.shape[2:], mode='bilinear', align_corners=True)
        out2 = F.gelu(out2)
        out2 = torch.add(out2, t2)

        out1 = F.interpolate(self.decoder6(out2), size=t1.shape[2:], mode='bilinear', align_corners=True)
        out1 = F.gelu(out1)
        out1 = torch.add(out1, t1)

        out0 = self.final(out1)

        # 恢复到原始输入尺寸 (因为 t1 即使是全分辨率，如果在 InvertedResidual 里有 stride，这里也需要对齐)
        out0 = F.interpolate(out0, size=x.shape[2:], mode='bilinear', align_corners=True)

        return torch.sigmoid(out0)