import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .postion_encoding import PositionEmbeddingLearned
from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv
# from darknet import BaseConv, CSPDarknet, CSPLayer, DWConv

class YOLOPAFPN(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )


    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)  
        return P3_out

class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        #---------------------------------------------------#
        #   inputs输入
        #   P3_out  80, 80, 256
        #   P4_out  40, 40, 512
        #   P5_out  20, 20, 1024
        #---------------------------------------------------#
        outputs = []
        for k, x in enumerate(inputs):
            #---------------------------------------------------#
            #   利用1x1卷积进行通道整合
            #---------------------------------------------------#
            x       = self.stems[k](x)
            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            cls_feat    = self.cls_convs[k](x)
            #---------------------------------------------------#
            #   判断特征点所属的种类
            #   80, 80, num_classes
            #   40, 40, num_classes
            #   20, 20, num_classes
            #---------------------------------------------------#
            cls_output  = self.cls_preds[k](cls_feat)

            #---------------------------------------------------#
            #   利用两个卷积标准化激活函数来进行特征提取
            #---------------------------------------------------#
            reg_feat    = self.reg_convs[k](x)
            #---------------------------------------------------#
            #   特征点的回归系数
            #   reg_pred 80, 80, 4
            #   reg_pred 40, 40, 4
            #   reg_pred 20, 20, 4
            #---------------------------------------------------#
            reg_output  = self.reg_preds[k](reg_feat)
            #---------------------------------------------------#
            #   判断特征点是否有对应的物体
            #   obj_pred 80, 80, 1
            #   obj_pred 40, 40, 1
            #   obj_pred 20, 20, 1
            #---------------------------------------------------#
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
        return outputs
    

class CSWF(nn.Module):
    def __init__(self,in_channel, out_channel):
        super().__init__()
        self.conv_1 = nn.Sequential(
            BaseConv(in_channel, in_channel//2, 1, 1),
            BaseConv(in_channel//2, in_channel, 1, 1)
        )
        self.conv_2 = nn.Sequential(
            BaseConv(in_channel, in_channel//2, 1, 1),
            BaseConv(in_channel//2, in_channel, 1, 1, act="sigmoid")
        )
        self.conv = nn.Sequential(
            BaseConv(in_channel, in_channel//2, 1, 1),
            BaseConv(in_channel//2, out_channel, 1, 1)
        )
        
    def forward(self, r_feat, c_feat):
        m_feat = r_feat + c_feat
        m_feat = self.conv_2(self.conv_1(m_feat))
        m_feat = self.conv(c_feat*m_feat + r_feat*(1-m_feat))
        
        return m_feat
    
class MSA(nn.Module):
    def __init__(self, channels=[128,256,512], num_frame=5, dim=1024):
        super().__init__()
        self.num_frame = num_frame
        self.K = nn.Sequential(
            BaseConv(channels[0], channels[0],3,2),
            BaseConv(channels[0], channels[0],1,1)
        )
        self.V = nn.Sequential(
            BaseConv(channels[0], channels[0],3,2),
            BaseConv(channels[0], channels[0],1,1)
        )
        self.Q = nn.Sequential(
            BaseConv(channels[0], channels[0],3,2),
            BaseConv(channels[0], channels[0],1,1)
        )
        self.position = PositionEmbeddingLearned(num_pos_feats=64)
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=8)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Linear(dim, dim)

    def forward(self, ref, cur):
        B, C, H, W = cur.shape
        K, V = self.K(ref), self.V(ref)
        Q = self.Q(cur)
        # pos = self.position(Q).reshape(B,C,-1)
        # attn, _ = self.attn(Q.reshape(B,C,-1)+pos, K.reshape(B,C,-1)+pos, V.reshape(B,C,-1)+pos)
        attn, _ = self.attn(Q.reshape(B,C,-1), K.reshape(B,C,-1), V.reshape(B,C,-1))
        attn = self.norm(attn+Q.reshape(B,C,-1))
        attn = self.norm(attn + self.ffn(attn)).reshape(B,C//4,H,W)
        return attn

class Neck(nn.Module):
    def __init__(self, channels=[128,256,512] ,num_frame=5):
        super().__init__()
        self.num_frame = num_frame
        #  关键帧与参考帧融合
        self.conv_ref = nn.Sequential(
            BaseConv(channels[0]*(self.num_frame-1), channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1,act='sigmoid')
        )
        self.conv_cur = nn.Sequential(
            BaseConv(channels[0], channels[0],3,1),
            BaseConv(channels[0], channels[0],3,1)
        )
        
        # 参考帧分别与关键帧融合
        for i in range(1,num_frame):
            self.__setattr__("attn_%d"%i, MSA(channels=channels,num_frame=num_frame,dim=1024)) 

        # 最终融合
        self.conv_gl_mix = nn.Sequential(
            BaseConv(channels[0]//4*(self.num_frame-1), channels[0],3,1),
            BaseConv(channels[0],channels[0],3,1)
        )
        self.conv_cr_mix = nn.Sequential(
            BaseConv(channels[0]*2, channels[0]*2,3,1),
            BaseConv(channels[0]*2,channels[0],3,1)
        )
        self.conv_final = CSWF(channels[0], channels[0])

    def forward(self, feats):
        f_feats = []
        # 空间
        r_feat = torch.cat([feats[j] for j in range(self.num_frame-1)],dim=1)
        r_feat = self.conv_ref(r_feat)
        c_feat = self.conv_cur(r_feat*feats[-1])
        c_feat = self.conv_cr_mix(torch.cat([c_feat, feats[-1]], dim=1))
        
        # 时间
        r_feats = torch.cat([self.__getattr__("attn_%d"%i)(feats[i-1], feats[-1]) 
                             for i in range(1, self.num_frame)], dim=1)

        r_feat= self.conv_gl_mix(r_feats)
        
        # Complementary symmetry weighting Fusion
        c_feat = self.conv_final(r_feat,c_feat)
        f_feats.append(c_feat)
        return f_feats

class Tasanet(nn.Module):
    def __init__(self, num_classes, fp16=False, num_frame=5):
        super(Tasanet, self).__init__()
        self.num_frame = num_frame
        self.backbone = YOLOPAFPN(0.33,0.50) 

        #-----------------------------------------#
        #   尺度感知模块
        #-----------------------------------------#
        self.neck = Neck(channels=[128], num_frame=num_frame)
        #----------------------------------------------------------#
        #   head
        #----------------------------------------------------------#
        self.head = YOLOXHead(num_classes=num_classes, width = 1.0, in_channels = [128], act = "silu")

    def forward(self, inputs):
        feat = []
        for i in range(self.num_frame):
            feat.append(self.backbone(inputs[:,:,i,:,:]))
        """[b,128,32,32][b,256,16,16][b,512,8,8]"""
        
        if self.neck:
            feat = self.neck(feat)
        outputs  = self.head(feat)
        return  outputs   # 计算损失那边 的anchor 应该是 [1, M, 4] size的


if __name__ == "__main__":
    
    from yolo_training import YOLOLoss
    net = Tasanet(num_classes=1, num_frame=5)

    bs = 4
    a = torch.randn(bs, 3, 5, 256, 256)
    out = net(a)
    for item in out:
        print(item.size())
        
    yolo_loss    = YOLOLoss(num_classes=1, fp16=False, strides=[16])

    target = torch.randn([bs, 1, 5]).cuda()
    target = nn.Softmax()(target)
    target = [item for item in target]

    loss = yolo_loss(out, target)
    print(loss)
