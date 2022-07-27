# coding=utf-8
import copy
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .PVT import pvt_small


class Conv2dReLU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=pad),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Attention(nn.Module):
    def __init__(self, channels, ratio=1):
        super().__init__()
        self.conv = Conv2dReLU(channels, channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels//ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels//ratio, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.conv(x)
        y = self.fc(self.avg_pool(y))
        return x * y + x


class SalHead(nn.Module):
    def __init__(self, in_channels, kernel_size=3, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2
        hidden_channels = in_channels // 2
        self.conv1 = Conv2dReLU(in_channels, hidden_channels, kernel_size)
        self.conv2 = Conv2dReLU(hidden_channels, 1, kernel_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, img_up_size):
        out = self.conv2(self.conv1(x))
        out = torch.sigmoid(self.dropout(out))
        out = F.interpolate(out, size=img_up_size, mode='bilinear', align_corners=True)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, t_in_chans=[64,128,320,512],c_out_chans=[16,64,64,128,256,256],
            pretrained='./pretrained/pvt_small.pth'):
        super().__init__()
        self.trans_encoder = pvt_small()
        checkpoint = torch.load(pretrained)
        for k in ['cls_token', 'head.weight', 'head.bias', 'norm.weight', 'norm.bias']:
            if k in checkpoint:
                #print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        # interpolate position embedding
        checkpoint = self.init_pos_embed(checkpoint, self.trans_encoder.patch_embed1, self.trans_encoder.pos_embed1, stage=1)
        checkpoint = self.init_pos_embed(checkpoint, self.trans_encoder.patch_embed2, self.trans_encoder.pos_embed2, stage=2)
        checkpoint = self.init_pos_embed(checkpoint, self.trans_encoder.patch_embed3, self.trans_encoder.pos_embed3, stage=3)
        checkpoint = self.init_pos_embed(checkpoint, self.trans_encoder.patch_embed4, self.trans_encoder.pos_embed4, stage=4)
        self.trans_encoder.load_state_dict(checkpoint, strict=True)

        self.conv1 = nn.Sequential(
            Conv2dReLU(3, c_out_chans[0], 3),
            Conv2dReLU(c_out_chans[0], c_out_chans[0], 3),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            Conv2dReLU(c_out_chans[0], c_out_chans[1], 3),
            Conv2dReLU(c_out_chans[1], c_out_chans[1], 3),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            Conv2dReLU(t_in_chans[0]+c_out_chans[1], c_out_chans[2], 1),
            Conv2dReLU(c_out_chans[2], c_out_chans[2], 3),
            Conv2dReLU(c_out_chans[2], c_out_chans[2], 3),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            Conv2dReLU(t_in_chans[1]+c_out_chans[2], c_out_chans[3], 1),
            Conv2dReLU(c_out_chans[3], c_out_chans[3], 3),
            Conv2dReLU(c_out_chans[3], c_out_chans[3], 3),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
            Conv2dReLU(t_in_chans[2]+c_out_chans[3], c_out_chans[4], 1),
            Conv2dReLU(c_out_chans[4], c_out_chans[4], 3),
            Conv2dReLU(c_out_chans[4], c_out_chans[4], 3),
        )
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Sequential(
            Conv2dReLU(t_in_chans[3]+c_out_chans[4], c_out_chans[5], 1),
            Conv2dReLU(c_out_chans[5], c_out_chans[5], 3),
        )

    def forward(self, x):
        size = x.size()[2:]
        trans = self.trans_encoder(x)[::-1]

        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = torch.cat([trans[0], pool2], dim=1)
        conv3 = self.conv3(conv3)
        pool3 = self.pool3(conv3)

        conv4 = torch.cat([trans[1], pool3], dim=1)
        conv4 = self.conv4(conv4)
        pool4 = self.pool4(conv4)

        conv5 = torch.cat([trans[2], pool4], dim=1)
        conv5 = self.conv5(conv5)
        pool5 = self.pool5(conv5)

        conv6 = torch.cat([trans[3], pool5], dim=1)
        conv6 = self.conv6(conv6)
        return trans[::-1], [conv6, conv5, conv4, conv3, conv2, conv1]


    def init_pos_embed(self, checkpoint, patch_embed, pos_embed, stage=1):
        pos_embed_name = 'pos_embed' + str(stage)
        pos_embed_checkpoint = checkpoint[pos_embed_name]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = patch_embed.num_patches
        num_extra_tokens = pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint[pos_embed_name] = new_pos_embed
        return checkpoint


class VisionTransformer(nn.Module):
    def __init__(self, chans=[256,256,128,64,64,16], down_chans=[256,128,64,32,32,8],
            cat_chans=[256,256,128,64,64,16], sal_chans=[128,64,32,32,8,8],
            trans_chans=[512,320,128,64], trans_down_chans=[128,64,32,16],
            trans_cat_chans=[128,128,64,32], trans_sal_chans=[64,32,16,16],
            dec_trans_cat_chans=[128,192,96,48]):
        super().__init__()
        # Encoder
        self.encoder = EncoderBlock()

        # Decoder1
        self.trans_downs = nn.ModuleList([Conv2dReLU(chan, trans_down_chans[id])
            for id, chan in enumerate(trans_chans)])
        self.trans_attention = nn.ModuleList([Attention(chan) for chan in trans_cat_chans])
        self.trans_cat_downs = nn.ModuleList([Conv2dReLU(chan, trans_sal_chans[id])
            for id, chan in enumerate(trans_cat_chans)])
        self.trans_sal_head = SalHead(trans_sal_chans[-1])

        # Decoder2
        self.num_trans = len(trans_chans)
        self.dec_trans_downs = nn.ModuleList([Conv2dReLU(chan, trans_down_chans[id])
            for id, chan in enumerate(trans_cat_chans)])
        self.dec_trans_cat = nn.ModuleList([Conv2dReLU(chan, chan) for chan in dec_trans_cat_chans])

        self.dec_downs = nn.ModuleList([Conv2dReLU(chan, down_chans[id])
            for id, chan in enumerate(chans)])
        self.dec_conv_cat = nn.ModuleList([Conv2dReLU(chan, chan) for chan in cat_chans])

        t_chans = trans_down_chans + [0, 0]
        t_cat_chans = dec_trans_cat_chans + [0, 0]
        self.dec_attention = nn.ModuleList([Attention(chan+t_chans[id])
            for id, chan in enumerate(cat_chans)])
        self.dec_cat_downs = nn.ModuleList([Conv2dReLU(chan+t_chans[id], sal_chans[id])
            for id, chan in enumerate(cat_chans)])
        self.dec_sal_head = nn.ModuleList([SalHead(chan) for chan in sal_chans])

    def forward(self, x):
        img_up_size = x.size()[2:]
        trans_outputs, enc_outputs = self.encoder(x)

        outs = []
        trans_outs = []
        for id, enc in enumerate(trans_outputs):
            enc = self.trans_downs[id](enc)
            if id == 0:
                dec = enc
            else:
                dec = F.interpolate(dec, size=enc.size()[2:], mode='bilinear', align_corners=True)
                dec = torch.cat([enc, dec], dim=1)
            dec = self.trans_attention[id](dec)
            trans_outs.append(dec)
            dec = self.trans_cat_downs[id](dec)
        outs.append(self.trans_sal_head(dec, img_up_size))

        for id, enc in enumerate(enc_outputs):
            #Conv
            enc = self.dec_downs[id](enc)
            if id == 0:
                dec_conv = dec = enc
            else:
                dec_up = F.interpolate(dec, size=enc.size()[2:], mode='bilinear', align_corners=True)
                dec_conv = self.dec_conv_cat[id](torch.cat([enc, dec_up], dim=1))
                dec = torch.cat([enc, dec_up], dim=1)

            #Trans
            dec_trans = None
            if id < self.num_trans:
                trans = self.dec_trans_downs[id](trans_outs[id])
                dec = torch.cat([dec, trans], dim=1)

            dec = self.dec_attention[id](dec)
            dec = self.dec_cat_downs[id](dec)
            outs.append(self.dec_sal_head[id](dec, img_up_size))
        return tuple(outs[::-1])
