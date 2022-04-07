#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Zhu Wenjing
# Date: 2022-03-07
# E-mail: zhuwenjing02@duxiaoman.com

import torch
import torch.nn as nn
import torch.nn.functional as F
from area_attention import AreaAttention
from g_mlp_pytorch import gMLP
from g_mlp_pytorch import SpatialGatingUnit

class MultiConv(nn.Module):
    '''
    Multi-scale block without short-cut connections
    '''
    def __init__(self, channels = 16, **kwargs):
        super(MultiConv, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=channels, out_channels=channels, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=channels, out_channels=channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = torch.cat((x3,x5),1)
        x = self.bn(x)
        x = F.relu(x)
        return x

class ResMultiConv(nn.Module):
    '''
    Multi-scale block with short-cut connections
    '''
    def __init__(self, channels = 16, **kwargs):
        super(ResMultiConv, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=channels, out_channels=channels, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=channels, out_channels=channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):
        x3 = self.conv3(x) + x
        x5 = self.conv5(x) + x
        x = torch.cat((x3,x5),1)
        x = self.bn(x)
        x = F.relu(x)
        return x    

class ResConv3(nn.Module):
    '''
    Resnet with 3x3 kernels
    '''
    def __init__(self, channels = 16, **kwargs):
        super(ResConv3, self).__init__()
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=channels, out_channels=2*channels, padding=1)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):        
        x = self.conv3(x) + torch.cat((x,x),1)
        x = self.bn(x)
        x = F.relu(x)
        return x    

class ResConv5(nn.Module):
    '''
    Resnet with 5x5 kernels
    '''
    def __init__(self, channels = 16, **kwargs):
        super(ResConv5, self).__init__()
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=channels, out_channels=2*channels, padding=2)
        self.bn = nn.BatchNorm2d(channels*2)

    def forward(self, x):        
        x = self.conv5(x) + torch.cat((x,x),1)
        x = self.bn(x)
        x = F.relu(x)
        return x    

class CNN_Area(nn.Module):
    '''Area attention, Mingke Xu, 2020
    '''
    def __init__(self, height=3,width=3,out_size=4, shape=(26,63), **kwargs):
        super(CNN_Area, self).__init__()
        self.height=height
        self.width=width
        # self.conv1 = nn.Conv2D(32, (3,3), padding='same', data_format='channels_last',)
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        # self.conv6 = nn.Conv2D(128, (3, 3), padding= )#
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        # self.gap = nn.AdaptiveAvgPool2d(1)

        i = 80 * ((shape[0]-1)//2) * ((shape[1]-1)//4) 
        self.fc = nn.Linear(in_features=i, out_features=4)

    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa=F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x=F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        return x

class CNN_AttnPooling(nn.Module):
    '''Head atention, Mingke Xu, 2020
    Attention pooling, Pengcheng Li, Interspeech, 2019
    '''
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(CNN_AttnPooling, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.top_down = nn.Conv2d(kernel_size=(1, 1), in_channels=80, out_channels=4)
        self.bottom_up = nn.Conv2d(kernel_size=(1, 1), in_channels=80, out_channels=1)
        # i = 80 * ((shape[0]-1)//4) * ((shape[1]-1)//4) 
        # self.fc = nn.Linear(in_features=i, out_features=4)

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)

        x1 = self.top_down(x)
        x1 = F.softmax(x1,1)
        x2 = self.bottom_up(x)

        x = x1 * x2

        # x = x.sum((2,3))        
        x = x.mean((2,3))        

        return x

class CNN_GAP(nn.Module):
    '''Head atention, Mingke Xu, 2020
    Attention pooling, Pengcheng Li, Interspeech, 2019
    '''
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(CNN_GAP, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = 80 * ((shape[0]-1)//4) * ((shape[1]-1)//4) 
        self.fc = nn.Linear(in_features=i, out_features=4)

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class MHCNN(nn.Module):
    '''
    Multi-Head Attention
    '''
    def __init__(self, head=4, attn_hidden=64,**kwargs):
        super(MHCNN, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features=self.attn_hidden, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

class MHCNN_AreaConcat(nn.Module):
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(MHCNN_AreaConcat, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = self.attn_hidden * ((shape[0]-1)//2) * ((shape[1]-1)//4) * self.head
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        # x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

class MHCNN_AreaConcat_gap(nn.Module):
    def __init__(self, head=4, attn_hidden=64,**kwargs):
        super(MHCNN_AreaConcat_gap, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = self.attn_hidden 
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

class MHCNN_AreaConcat_gap1(nn.Module):
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(MHCNN_AreaConcat_gap1, self).__init__()
        self.head = head
        self.attn_hidden = 32
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        self.gap = nn.AdaptiveAvgPool2d(1)
        i = self.attn_hidden 
        self.fc = nn.Linear(in_features=(shape[0]-1)//2, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=80, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        xa = self.conv1a(input[0])
        xa = self.bn1a(xa)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)

        x= F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)

        x = F.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)

        x = F.relu(x)
        # #attention

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 3)
        x = attn
        x = x.contiguous().permute(0, 2, 3, 1)
        x = F.relu(x)
        x = self.gap(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x

class AACNN(nn.Module):
    '''
    Area Attention, ICASSP 2020
    '''
    def __init__(self, height=3,width=3,out_size=4, shape=(26,63), **kwargs):
        super(AACNN, self).__init__()
        self.height=height
        self.width=width
        # self.conv1 = nn.Conv2D(32, (3,3), padding='same', data_format='channels_last',)
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=16, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=16, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        # self.conv6 = nn.Conv2D(128, (3, 3), padding= )#
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        # self.gap = nn.AdaptiveAvgPool2d(1)

        i = 80 * ((shape[0] - 1)//2) * ((shape[1] - 1)//4)
        self.fc = nn.Linear(in_features=i, out_features=4)
        # self.dropout = nn.Dropout(0.5)

        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=height,
            max_area_width=width,
            dropout_rate=0.5,
        )


    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa=F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2)

        x = self.conv2(x)
        x = self.bn2(x)
        x=F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        shape = x.shape
        x = x.contiguous().permute(0, 2, 3, 1).view(shape[0], shape[3]*shape[2], shape[1])
        x = self.area_attention(x,x,x)
        x = F.relu(x)
        x = x.reshape(*shape)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        
        return x

class AACNN_HeadConcat(nn.Module):
    def __init__(self, height=3,width=3,out_size=4, shape=(26,63), **kwargs):
        super(AACNN_HeadConcat, self).__init__()
        self.height=height
        self.width=width
        self.conv1a = nn.Conv2d(kernel_size=(10, 2), in_channels=1, out_channels=8, padding=(4, 0))
        self.conv1b = nn.Conv2d(kernel_size=(2, 8), in_channels=1, out_channels=8, padding=(0, 3))
        self.conv2 = nn.Conv2d(kernel_size=(3, 3), in_channels=16, out_channels=32, padding=1)
        self.conv3 = nn.Conv2d(kernel_size=(3, 3), in_channels=32, out_channels=48, padding=1)
        self.conv4 = nn.Conv2d(kernel_size=(3, 3), in_channels=48, out_channels=64, padding=1)
        self.conv5 = nn.Conv2d(kernel_size=(3, 3), in_channels=64, out_channels=80, padding=1)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(8)
        self.bn1b = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(48)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(80)
        i = 80 * ((shape[0] - 1)//4) * ((shape[1] - 1)//4)
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)
        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=height,
            max_area_width= width,
            dropout_rate=0.5,
            # top_k_areas=0
        )


    def forward(self, *input):
        x = input[0]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa=F.relu(xa)
        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)
        x = torch.cat((xa, xb), 1)

        x = self.conv2(x)
        x = self.bn2(x)
        x=F.relu(x)
        x = self.maxp(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxp(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.contiguous().permute(0, 2, 3, 1).view(shape[0], shape[3]*shape[2], shape[1])
        
        x = self.area_attention(x,x,x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        
        return x

class GLAM5(nn.Module):
    '''
    GLobal-Aware Multiscale block with 5x5 convolutional kernels in CNN architecture
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(GLAM5, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(5, 1), in_channels=1, out_channels=16, padding=(2, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 5), in_channels=1, out_channels=16, padding=(0, 2))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class GLAM(nn.Module):
    '''
    GLobal-Aware Multiscale block with 3x3 convolutional kernels in CNN architecture
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(GLAM, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class gMLPResConv3(nn.Module):
    '''
    GLAM - Multiscale
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPResConv3, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResConv3(16)
        self.conv3 = ResConv3(32)
        self.conv4 = ResConv3(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class gMLPResConv5(nn.Module):
    '''
    GLAM5 - Multiscale
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPResConv5, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResConv5(16)
        self.conv3 = ResConv5(32)
        self.conv4 = ResConv5(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class gMLPMultiConv(nn.Module):
    '''
    GLAM - Resnet
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPMultiConv, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = MultiConv(16)
        self.conv3 = MultiConv(32)
        self.conv4 = MultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class MHResMultiConv3(nn.Module):
    '''
    Multi-Head-Attention with Multiscale blocks
    '''
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(MHResMultiConv3, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        i = self.attn_hidden * self.head * (shape[0]//2) * (shape[1]//4)
        self.fc = nn.Linear(in_features=i, out_features=4)

        self.dropout = nn.Dropout(0.5)
        self.attention_query = nn.ModuleList()
        self.attention_key = nn.ModuleList()
        self.attention_value = nn.ModuleList()

        for i in range(self.head):
            self.attention_query.append(nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_key.append(nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1))
            self.attention_value.append(nn.Conv2d(in_channels=128, out_channels=self.attn_hidden, kernel_size=1))

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        attn = None
        for i in range(self.head):
            Q = self.attention_query[i](x)
            K = self.attention_key[i](x)
            V = self.attention_value[i](x)
            attention = F.softmax(torch.mul(Q, K),dim=1)
            attention = torch.mul(attention, V)

            if (attn is None):
                attn = attention
            else:
                attn = torch.cat((attn, attention), 2)
        x = attn
        x = F.relu(x)

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc(x)
        return x

class AAResMultiConv3(nn.Module):
    '''
    Area Attention with Multiscale blocks
    '''
    def __init__(self, head=4, attn_hidden=64, shape=(26,63), **kwargs):
        super(AAResMultiConv3, self).__init__()
        self.head = head
        self.attn_hidden = attn_hidden
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)

        i = 128 * (shape[0]//2) * (shape[1]//4) 
        self.fc = nn.Linear(in_features=i, out_features=4)
        # self.dropout = nn.Dropout(0.5)

        self.area_attention = AreaAttention(
            key_query_size=80,
            area_key_mode='mean',
            area_value_mode='sum',
            max_area_height=3,
            max_area_width=3,
            dropout_rate=0.5,
        )

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        shape = x.shape
        x = x.contiguous().permute(0, 2, 3, 1).view(shape[0], shape[3]*shape[2], shape[1])
        x = self.area_attention(x,x,x)
        x = F.relu(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        
        return x

class ResMultiConv3(nn.Module):
    '''
    GLAM - gMLP
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(ResMultiConv3, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class ResMultiConv5(nn.Module):
    '''
    GLAM5 - gMLP
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(ResMultiConv5, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(5, 1), in_channels=1, out_channels=16, padding=(2, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 5), in_channels=1, out_channels=16, padding=(0, 2))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class gMLPgResMultiConv3(nn.Module):
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPgResMultiConv3, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//4) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())
        self.sgu = SpatialGatingUnit(dim = shape[0] * shape[1] * 2, dim_seq = 16, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 26, 63)
        xa = self.bn1a(xa) # (32, 16, 26, 63)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        shape = x.shape
        x = x.view(*x.shape[:-2],-1)
        x = self.sgu(x)
        x = x.view(shape[0], shape[1], shape[2]//2, shape[3])

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class aMLPResMultiConv3(nn.Module):
    def __init__(self, shape=(26,63), **kwargs):
        super(aMLPResMultiConv3, self).__init__()
        self.conv1a = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=16, padding=(1, 0))
        self.conv1b = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=16, padding=(0, 1))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, attn_dim = 64, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa = self.conv1a(input[0]) # (32, 16, 25, 62)
        xa = self.bn1a(xa) # (32, 16, 25, 62)

        xa = F.relu(xa)
        xb = self.conv1b(input[0])
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

class gMLPResMultiConv35(nn.Module):
    '''
    Temporal and Spatial convolution with multiscales
    '''
    def __init__(self, shape=(26,63), **kwargs):
        super(gMLPResMultiConv35, self).__init__()
        self.conv1a1 = nn.Conv2d(kernel_size=(3, 1), in_channels=1, out_channels=8, padding=(1, 0))
        self.conv1a2 = nn.Conv2d(kernel_size=(5, 1), in_channels=1, out_channels=8, padding=(2, 0))
        self.conv1b1 = nn.Conv2d(kernel_size=(1, 3), in_channels=1, out_channels=8, padding=(0, 1))
        self.conv1b2 = nn.Conv2d(kernel_size=(1, 5), in_channels=1, out_channels=8, padding=(0, 2))
        self.conv2 = ResMultiConv(16)
        self.conv3 = ResMultiConv(32)
        self.conv4 = ResMultiConv(64)
        self.conv5 = nn.Conv2d(kernel_size=(5, 5), in_channels=128, out_channels=128, padding=2)
        self.maxp = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn1a = nn.BatchNorm2d(16)
        self.bn1b = nn.BatchNorm2d(16)
        self.bn5 = nn.BatchNorm2d(128)
        dim = (shape[0]//2) * (shape[1]//4)
        i = 128 * dim
        self.fc = nn.Linear(in_features=i, out_features=4)
        self.dropout = nn.Dropout(0.5)

        self.gmlp = gMLP(dim = dim, depth = 1, seq_len = 128, act = nn.Tanh())

    def forward(self, *input):
        # input[0]: torch.Size([32, 1, 26, 63])
        xa1 = self.conv1a1(input[0]) # (32, 8, 26, 63)
        xa2 = self.conv1a2(input[0]) # (32, 8, 26, 63)
        xa = torch.cat((xa1,xa2),1)
        xa = self.bn1a(xa) # (32, 16, 26, 63)
        xa = F.relu(xa)

        xb1 = self.conv1b1(input[0])
        xb2 = self.conv1b2(input[0])
        xb = torch.cat((xb1,xb2),1)
        xb = self.bn1b(xb)

        xb = F.relu(xb)
        x = torch.cat((xa, xb), 2) # (32, 16, 50, 62)

        x = self.conv2(x) # (32, 32, 50, 62)
        x = self.maxp(x)
        x = self.conv3(x) # (32, 64, 25, 31)
        x = self.maxp(x)
        x = self.conv4(x) # (32, 128, 12, 15)

        x = self.conv5(x)  # (32, 128, 12, 15)
        x = self.bn5(x)
        x = F.relu(x)

        # flatten
        shape = x.shape
        x = x.view(*x.shape[:-2],-1)

        x = self.gmlp(x)
        x = F.relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
