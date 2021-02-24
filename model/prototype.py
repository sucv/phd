import torch
import torch.nn as nn
from torchvision.models import resnet50, vgg13
import torch.nn.functional as F
import torch.nn.init as init

from model.arcface_model import Backbone
from model.temporal_convolutional_model import TemporalConvNet

import os
import argparse
import torch
from torch import nn
from torch.nn.init import xavier_uniform_, constant_

class my_res50(nn.Module):
    def __init__(self, num_classes=8, use_pretrained=True):
        super().__init__()
        self.backbone = Backbone(num_layers=50, drop_ratio=0.6, mode="ir_se")
        if use_pretrained:
            state_dict = torch.load("load/model_ir_se50.pth", map_location='cpu')
            self.backbone.load_state_dict(state_dict)
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.logits = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.logits(x)
        return x

class my_2d1d(nn.Module):
    def __init__(self, backbone_model_name, feature_dim, channels_1D, output_dim, kernel_size, dropout=0.1):
        super().__init__()

        CNN_spatial = my_res50()
        path = os.path.join("load", backbone_model_name + ".pth")
        state_dict = torch.load(path, map_location='cpu')
        CNN_spatial.load_state_dict(state_dict)
        self.CNN_spatial = CNN_spatial.backbone

        for param in self.CNN_spatial.parameters():
            param.requires_grad = False
        self.CNN_temporal = TemporalConvNet(num_inputs=feature_dim, num_channels=channels_1D, kernel_size=kernel_size, dropout=dropout)
        self.regressor = nn.Linear(feature_dim // 4, output_dim)

    def forward(self, x):
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.CNN_spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).transpose(1, 2).contiguous()
        x = self.CNN_temporal(x).transpose(1, 2).contiguous()
        x = x.contiguous().view(num_batches * length, -1)
        x = self.regressor(x)
        x = x.view(num_batches, length, -1)
        return x

class my_2dlstm(nn.Module):
    def __init__(self, backbone_model_name, feature_dim, output_dim, dropout=0.5):
        super().__init__()

        CNN_spatial = my_res50()
        path = os.path.join("load", backbone_model_name + ".pth")
        state_dict = torch.load(path, map_location='cpu')
        CNN_spatial.load_state_dict(state_dict)
        self.CNN_spatial = CNN_spatial.backbone

        for param in self.CNN_spatial.parameters():
            param.requires_grad = False
        self.temporal = nn.LSTM(input_size=feature_dim, hidden_size=64, num_layers=2,
                                batch_first=True, bidirectional=True, dropout=dropout)
        self.regressor = nn.Linear(feature_dim // 4, output_dim)

    def forward(self, x):
        num_batches, length, channel, width, height = x.shape
        x = x.view(-1, channel, width, height)
        x = self.CNN_spatial(x)
        _, feature_dim = x.shape
        x = x.view(num_batches, length, feature_dim).contiguous()
        x, _ = self.temporal(x)
        x = x.contiguous().view(num_batches * length, -1)
        x = self.regressor(x)
        x = x.view(num_batches, length, -1)
        return x


if __name__ == "__main__":
    device = torch.device("cuda:2")
    data = torch.randint(0, 254, (2, 300, 3, 122, 122), dtype=torch.uint8)
    model = my_2d1d(feature_dim=512, channels_1D=[128, 128, 128], output_dim=2, kernel_size=3, dropout=0.1)

