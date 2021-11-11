from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler


from torchvision import transforms


from pytorchvideo.models import x3d
from pytorchvideo.data import (
    Ucf101,
    RandomClipSampler,
    UniformClipSampler,
    Kinetics,
    SSv2
)


from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)


import torchinfo

import numpy as np
from tqdm.notebook import tqdm
import itertools
import os
import pickle
import random
import matplotlib.pyplot as plt
import shutil
from sklearn import mixture
from sklearn import svm
from sklearn import decomposition
import os.path as osp
import argparse
import configparser


class ReconstructNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = torch.hub.load(
            'facebookresearch/pytorchvideo', "x3d_m", pretrained=True)
        self.model_num_features = model.blocks[5].proj.in_features
        self.num_class = 101

        self.net_bottom = nn.Sequential(
            model.blocks[0],
            model.blocks[1],
            model.blocks[2],
            model.blocks[3],
        )

        self.blocks4 = model.blocks[4]

        self.net_top = nn.Sequential(
            model.blocks[5].pool,
            model.blocks[5].dropout
        )

        # self.linear = model.blocks[5].proj
        self.linear = nn.Linear(self.model_num_features, self.num_class)

        # 学習させるパラメータ名
        self.update_param_names = ["linear.weight", "linear.bias"]
        # 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
        for name, param in self.named_parameters():
            if name in self.update_param_names:
                param.requires_grad = True
                # print(name)
            else:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net_bottom(x)
        x = self.blocks4(x)
        x = self.net_top(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        x = x.view(-1, self.num_class)
        return x


class Adapter2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.bn2 = nn.BatchNorm2d(dim)

    def video_to_frame(self, inputs):
        batch_size = inputs.size(0)
        num_frame = inputs.size(2)

        inputs = inputs.permute(0, 2, 1, 3, 4)
        outputs = inputs.reshape(batch_size * num_frame,
                                 inputs.size(2),
                                 inputs.size(3),
                                 inputs.size(4))

        return outputs

    def frame_to_video(
            self, input: torch.Tensor, batch_size, num_frame, channel, height, width) -> torch.Tensor:
        output = input.reshape(batch_size, num_frame, channel, height, width)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def forward(self, x):
        batch_size = x.size(0)
        num_frame = x.size(2)
        channel = x.size(1)
        height = x.size(3)

        x = self.video_to_frame(x)
        residual = x
        out = self.bn1(x)
        out = self.conv1(out)
        out += residual
        out = self.bn2(out)

        out = self.frame_to_video(
            out, batch_size, num_frame, channel, height, height)
        # print(out.shape)

        return out


class TemporalAdapter(nn.Module):
    def __init__(self, channel_dim, frame_dim):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(channel_dim)
        self.conv1 = nn.Conv2d(frame_dim, frame_dim, 1)
        self.bn2 = nn.BatchNorm3d(channel_dim)

    def swap_channel_frame(self, inputs):
        batch_size = inputs.size(0)
        channel = inputs.size(1)
        num_frame = inputs.size(2)

        # inputs = inputs.permute(0, 2, 1, 3, 4)
        outputs = inputs.reshape(batch_size * channel,
                                 num_frame,
                                 inputs.size(3),
                                 inputs.size(4))

        return outputs

    def frame_to_video(
            self, input: torch.Tensor, batch_size, num_frame, channel, height, width) -> torch.Tensor:
        output = input.reshape(batch_size, channel, num_frame, height, width)
        # output = output.permute(0,2,1,3,4)
        return output

    def forward(self, x):
        batch_size = x.size(0)
        channel = x.size(1)
        num_frame = x.size(2)
        height = x.size(3)
        width = x.size(4)

        residual = x

        out = self.bn1(x)
        out = self.swap_channel_frame(out)
        out = self.conv1(out)
        out = self.frame_to_video(
            out, batch_size, num_frame, channel, height, width)
        out += residual
        out = self.bn2(out)

        return out


class SpaceTemporalAdapter(nn.Module):
    def __init__(self, channel_dim, frame_dim):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(channel_dim)
        self.conv1 = nn.Conv3d(channel_dim, channel_dim *
                               frame_dim, (frame_dim, 1, 1))
        self.bn2 = nn.BatchNorm3d(channel_dim)
        self.channel_dim = channel_dim
        self.frame_dim = frame_dim

    def reshape_dim(self, inputs):
        batch_size = inputs.size(0)
        output = inputs.reshape(
            batch_size, self.channel_dim, self.frame_dim, inputs.size(3), inputs.size(4))
        return output

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)
        out = self.reshape_dim(out)
        out += residual
        out = self.bn2(out)

        return out


class EfficientSpaceTemporalAdapter(nn.Module):
    def __init__(self, channel_dim, frame_dim):
        super().__init__()
        self.video2frame_adapter = Adapter2D(channel_dim)
        self.temporal_adapter = TemporalAdapter(channel_dim, frame_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.video2frame_adapter(x)
        out = self.relu(out)
        out = self.temporal_adapter(x)
        return out


def select_adapter(adapter, channel_dim, frame_dim):
    if adapter == "video2frame":
        adp = Adapter2D(channel_dim)
    elif adapter == "temporal":
        adp = TemporalAdapter(channel_dim, frame_dim)
    elif adapter == "space_temporal":
        adp = SpaceTemporalAdapter(channel_dim, frame_dim)
    elif adapter == "efficient_space_temporal":
        adp = EfficientSpaceTemporalAdapter(channel_dim, frame_dim)
    else:
        raise NameError("アダプタの名前が正しくないです．")
    return adp


class ReconstructNetInAdapter(nn.Module):
    def __init__(self, adapter_mode):
        super().__init__()
        model = torch.hub.load(
            'facebookresearch/pytorchvideo', "x3d_m", pretrained=True)
        self.model_num_features = model.blocks[5].proj.in_features
        self.num_class = 101
        self.num_frame = 16

        self.net_bottom = nn.Sequential(
            model.blocks[0],
            model.blocks[1],
            model.blocks[2],
            model.blocks[3],
        )

        self.adapter0 = select_adapter(adapter_mode, 96, self.num_frame)

        self.blocks4 = model.blocks[4]

        self.adapter1 = select_adapter(adapter_mode, 192, self.num_frame)

        self.net_top = nn.Sequential(
            model.blocks[5].pool,
            model.blocks[5].dropout
        )

        # self.linear = model.blocks[5].proj
        self.linear = nn.Linear(self.model_num_features, self.num_class)

        # 学習させるパラメータ名
        self.update_param_names = ["adapter0.bn1.weight", "adapter0.bn1.bias",
                                   "adapter0.conv1.weight", "adapter0.conv1.bias",
                                   "adapter0.bn2.weight", "adapter0.bn2.bias",
                                   "adapter1.bn1.weight", "adapter1.bn1.bias",
                                   "adapter1.conv1.weight", "adapter1.conv1.bias",
                                   "adapter1.bn2.weight", "adapter1.bn2.bias",
                                   "linear.weight", "linear.bias"]
        # self.update_param_names = ["adapter1.bn1.weight", "adapter1.bn1.bias",
        #                            "adapter1.conv1.weight", "adapter1.conv1.bias",
        #                            "adapter1.bn2.weight", "adapter1.bn2.bias",
        #                            "linear.weight", "linear.bias"]
        # 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
        for name, param in self.named_parameters():
            if name in self.update_param_names:
                param.requires_grad = True
                # print(name)
            else:
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = x.permute(0,2,1,3,4)
        x = self.net_bottom(x)
        x = self.adapter0(x)
        x = self.blocks4(x)
        x = self.adapter1(x)
        x = self.net_top(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        x = x.view(-1, self.num_class)
        return x


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=5,)
    parser.add_argument("--batch_size", type=int, default=32,)
    parser.add_argument("--num_workers", type=int, default=32,)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--adapter_mode", type=str, choices=[
        "video2frame",
        "temporal",
        "space_temporal",
        "efficient_space_temporal"])
    return parser.parse_args()


def main():
    config = configparser.ConfigParser()
    args = get_arguments()
    model = ReconstructNetInAdapter(args.adapter_mode)
    torchinfo.summary(
        model,
        input_size=(1, 3, 16, 224, 224),
        depth=2,
        col_names=["input_size",
                   "output_size"],
        row_settings=("var_names",)
    )


if __name__ == '__main__':
    main()
