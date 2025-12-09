import torch
import torch.nn as nn
import torch.nn.functional as F                    
from models.base_block import *
from models.HRNet import hrnet32
import warnings
                              
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP_PRED(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.dropout = nn.Dropout2d(0.1)
    def forward(self, x):

        x = x.flatten(2).transpose(1, 2)

        x = self.proj(x)
        x = self.dropout(x)
        return x
    
class ChannelExchange(nn.Module):
    def __init__(self, p=2):
        super(ChannelExchange, self).__init__()
        self.p = p

    def forward(self, x1, x2):
        N, C, H, W = x1.shape
        exchange_mask = torch.arange(C, device=x1.device) % self.p == 0  # 生成交换掩码
        exchange_mask = exchange_mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # 形状变为 [1, C, 1, 1]
        exchange_mask = exchange_mask.expand((N, -1, H, W))  # 扩展到 [N, C, H, W]

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask] = x1[~exchange_mask]  # 未交换的通道保持不变
        out_x2[~exchange_mask] = x2[~exchange_mask]
        out_x1[exchange_mask] = x2[exchange_mask]  # 交换的通道互换
        out_x2[exchange_mask] = x1[exchange_mask]

        return out_x1, out_x2
    
#Intermediate prediction module
def make_prediction(in_channels, out_channels, sigmoid=False):#生成BCD
    if sigmoid:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            # nn.Sigmoid()
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

#3D+3D+3D+2D+2D
class fuse_3D(nn.Module):
    def __init__(self, in_channels, exchange_ratio=2):
        super(fuse_3D, self).__init__()
        self.exchange = ChannelExchange(p=exchange_ratio)  # 通道交换模块
        # 三个连续的三维卷积
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=[3, 3, 3], stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm3d(1),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=[3, 3, 3], stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm3d(1),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(1, 1, kernel_size=[3, 3, 3], stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm3d(1),
        )

        # 两个二维卷积
        self.fuse_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels // 2),
        )
        self.fuse_conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels // 2),
        )

        # Dropout
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        tensor1 = x[0]  # 第一个输入张量，形状为 (b, c, h, w)
        tensor2 = x[1]  # 第二个输入张量，形状为 (b, c, h, w)
        tensor1, tensor2 = self.exchange(tensor1, tensor2)

        # 按通道拼接两个张量，形状为 (b, c*2, h, w)
        fused = torch.cat((tensor1, tensor2), dim=1)

        # 增加一个维度，变为 (b, 1, c*2, h, w)
        fused = fused.unsqueeze(1)

        # 三个连续的三维卷积
        fused = self.conv3d_1(fused)  # 形状仍为 (b, 1, c*2, h, w)
        fused = self.conv3d_2(fused)  # 形状仍为 (b, 1, c*2, h, w)
        fused = self.conv3d_3(fused)  # 形状仍为 (b, 1, c*2, h, w)

        # 去除多余的维度，恢复为 (b, c*2, h, w)
        fused = fused.squeeze(1)

        # 两个二维卷积
        fused = self.fuse_conv1(fused)  # 形状为 (b, c//2, h, w)
        fused = self.dropout(fused)    # 应用 Dropout
        fused = self.fuse_conv2(fused)  # 形状仍为 (b, c//2, h, w)
        fused = self.dropout(fused)    # 再次应用 Dropout

        return fused


class CD_Decoder(nn.Module):
    def __init__(self, in_channels = [32, 64, 128, 256], embedding_dim= 64, output_nc=7, feature_strides=[2, 4, 8, 16]):
        super(CD_Decoder, self).__init__()
        #assert
        assert len(feature_strides) == len(in_channels)
        assert min(feature_strides) == feature_strides[0]
        
        #settings
        self.feature_strides = feature_strides
        self.in_channels     = in_channels
        self.embedding_dim   = embedding_dim
        self.output_nc       = output_nc
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        #MLP decoder heads
        self.linear_c4 = MLP_PRED(input_dim=c4_in_channels, embed_dim=self.embedding_dim)
        self.linear_c3 = MLP_PRED(input_dim=c3_in_channels, embed_dim=self.embedding_dim)
        self.linear_c2 = MLP_PRED(input_dim=c2_in_channels, embed_dim=self.embedding_dim)
        self.linear_c1 = MLP_PRED(input_dim=c1_in_channels, embed_dim=self.embedding_dim)

        self.cosine_4 = Cosine_Similarity(c4_in_channels)
        self.cosine_3 = Cosine_Similarity(c3_in_channels)
        self.cosine_2 = Cosine_Similarity(c2_in_channels)
        self.cosine_1 = Cosine_Similarity(c1_in_channels)

        self.fuse_c4 = fuse_3D(in_channels=2*self.embedding_dim)
        self.fuse_c3 = fuse_3D(in_channels=2*self.embedding_dim)
        self.fuse_c2 = fuse_3D(in_channels=2*self.embedding_dim)
        self.fuse_c1 = fuse_3D(in_channels=2*self.embedding_dim)

        #taking outputs from middle of the encoder
        self.in_channels = [32, 64, 128, 256]

        self.make_pred_bcd = make_prediction(in_channels=self.embedding_dim, out_channels=1,sigmoid = True)

        self.linear_fuse_bcd = nn.Sequential(
            nn.Conv2d(   in_channels=self.embedding_dim*4, out_channels=self.embedding_dim, kernel_size=1),
            nn.BatchNorm2d(self.embedding_dim)
        )
        #Final predction head
        self.dense_2x   = nn.Sequential( ResidualBlock(self.embedding_dim))
        self.dense_1x   = nn.Sequential( ResidualBlock(self.embedding_dim))


    def forward(self, inputs1, inputs2):

        c1_1, c2_1, c3_1, c4_1 = inputs1
        c1_2, c2_2, c3_2, c4_2 = inputs2

        c1_1, c1_2 = self.cosine_1(c1_1, c1_2)
        c2_1, c2_2 = self.cosine_2(c2_1, c2_2)
        c3_1, c3_2 = self.cosine_3(c3_1, c3_2)
        c4_1, c4_2 = self.cosine_4(c4_1, c4_2)

        ############## decoder on C1-C4 ###########
        n, _, h, w = c4_1.shape
        outputs = []

        # Stage 4: x1/32 scale
        _c4_1 = self.linear_c4(c4_1).permute(0,2,1).reshape(n, -1, c4_1.shape[2], c4_1.shape[3])
        _c4_2 = self.linear_c4(c4_2).permute(0,2,1).reshape(n, -1, c4_2.shape[2], c4_2.shape[3])
        feat_c4 = self.fuse_c4([_c4_1, _c4_2])
        feat_c4_up = resize(feat_c4, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        
        # Stage 3: x1/16 scale
        _c3_1 = self.linear_c3(c3_1).permute(0,2,1).reshape(n, -1, c3_1.shape[2], c3_1.shape[3])
        _c3_2 = self.linear_c3(c3_2).permute(0,2,1).reshape(n, -1, c3_2.shape[2], c3_2.shape[3])
        feat_c3 = self.fuse_c3([_c3_1, _c3_2]) + F.interpolate(feat_c4, scale_factor=2, mode="bilinear")
        feat_c3_up = resize(feat_c3, size=c1_2.size()[2:], mode='bilinear', align_corners=False)
        
        # Stage 2: x1/8 scale
        _c2_1 = self.linear_c2(c2_1).permute(0,2,1).reshape(n, -1, c2_1.shape[2], c2_1.shape[3])
        _c2_2 = self.linear_c2(c2_2).permute(0,2,1).reshape(n, -1, c2_2.shape[2], c2_2.shape[3])
        feat_c2 = self.fuse_c2([_c2_1, _c2_2]) + F.interpolate(feat_c3, scale_factor=2, mode="bilinear")
        feat_c2_up = resize(feat_c2, size=c1_2.size()[2:], mode='bilinear', align_corners=False)

        # Stage 1: x1/4 scale
        _c1_1 = self.linear_c1(c1_1).permute(0,2,1).reshape(n, -1, c1_1.shape[2], c1_1.shape[3])
        _c1_2 = self.linear_c1(c1_2).permute(0,2,1).reshape(n, -1, c1_2.shape[2], c1_2.shape[3])
        feat_c1 = self.fuse_c1([_c1_1, _c1_2]) + F.interpolate(feat_c2, scale_factor=2, mode="bilinear")

        #Linear Fusion of difference image from all scales
        _c_bcd = self.linear_fuse_bcd(torch.cat([feat_c4_up, feat_c3_up, feat_c2_up, feat_c1], dim=1))

        x = self.dense_2x(_c_bcd)
        x = self.dense_1x(x)
        p_bcd = self.make_pred_bcd(x)     

        outputs = p_bcd

        return outputs
    

class Cosine_Similarity(nn.Module):
    def __init__(self, in_channels):
        super(Cosine_Similarity, self).__init__()
        self.in_channels = in_channels
        
    def forward(self, features1, features2):
        batch_size, channels, height, width = features1.size()
        
        # 将特征图展平为 [B, C, H*W]
        features1_flat = features1.view(batch_size, channels, -1)
        features2_flat = features2.view(batch_size, channels, -1)
        
        # 计算每个位置的余弦相似度
        similarities = F.cosine_similarity(features1_flat, features2_flat, dim=1)  # 形状: [B, H*W]
        
        # 将相似度重新形状为 [B, H, W]
        similarities = similarities.view(batch_size, height, width)
        
        # 反转相似度，使相似度越高，权重越小
        weights = 1 - similarities  # 形状: [B, H, W]
        
        # 将权重扩展到与特征张量的形状一致
        weights = weights.unsqueeze(1).expand_as(features1)  # 形状: [B, C, H, W]
        
        # 将注意力权重应用于输入特征
        weighted_features1 = features1 * weights
        weighted_features2 = features2 * weights
        
        return weighted_features1, weighted_features2

class HRNet_3D(nn.Module):

    def __init__(self, input_nc=3, output_nc=5, decoder_softmax=False, embed_dim=256):
        super(HRNet_3D, self).__init__()
        self.embed_dims = [32, 64, 128, 256]
        self.hrnet = hrnet32(pretrained=True)
        self.classifier1 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 7, kernel_size=1))
        self.classifier2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Conv2d(128, 7, kernel_size=1))

        self.embedding_dim = embed_dim
        self.CD_Decoder   = CD_Decoder(in_channels = self.embed_dims, embedding_dim= self.embedding_dim, output_nc=output_nc, 
                    feature_strides=[2, 4, 8, 16])
        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=480,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0)
        )  

    
    def base_forward(self, x):
        x1,x2,x3,x4 = self.hrnet(x)
        # 将 x2, x3, x4 上采样到 x1 的尺寸 (128x128)

        s2 =  [x1,x2,x3,x4]
        x2 = F.interpolate(x2, size=(128, 128), mode='bilinear', align_corners=True)  # (64, 128, 128)
        x3 = F.interpolate(x3, size=(128, 128), mode='bilinear', align_corners=True)  # (128, 128, 128)
        x4 = F.interpolate(x4, size=(128, 128), mode='bilinear', align_corners=True)  # (256, 128, 128)

        # 按通道拼接四个特征图
        fused_features = torch.cat([x1, x2, x3, x4], dim=1)  # (32 + 64 + 128 + 256, 128, 128) = (480, 128, 128)
        s1 = self.last_layer(fused_features)

        return s1, s2, x1


    def forward(self, x1, x2):
        x_size = x1.size()
        ss1, s1, x1 = self.base_forward(x1)
        ss2, s2, x2 = self.base_forward(x2)#提取特征
        
        cp = self.CD_Decoder(s1, s2)#变化检测

        out1 = self.classifier1(ss1)#语义分割
        out2 = self.classifier2(ss2)


        return F.upsample(cp, x_size[2:], mode='bilinear'),F.upsample(out1, x_size[2:], mode='bilinear'),F.upsample(out2, x_size[2:], mode='bilinear')