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
from tqdm import tqdm
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
import torch.onnx
import netron


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


class ParallelAdapter2D(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv0 = nn.Conv3d(in_dim, out_dim, (1, 1, 1), stride=(1, 2, 2))
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(out_dim, out_dim, 1)
        self.bn1 = nn.BatchNorm2d(out_dim)

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
        x = self.relu(self.conv0(x))

        batch_size = x.size(0)
        num_frame = x.size(2)
        channel = x.size(1)
        height = x.size(3)

        x = self.video_to_frame(x)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out += residual

        out = self.frame_to_video(
            out, batch_size, num_frame, channel, height, height)

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

        # self.adapter0 = select_adapter(adapter_mode, 96, self.num_frame)

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
        # x = self.adapter0(x)
        x = self.blocks4(x)
        x = self.adapter1(x)
        x = self.net_top(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        x = x.view(-1, self.num_class)
        return x


class ReconstructNetInParallelAdapter(nn.Module):
    def __init__(self, adapter_mode):
        super().__init__()
        model = torch.hub.load(
            'facebookresearch/pytorchvideo', "x3d_m", pretrained=True)
        self.model_num_features = model.blocks[5].proj.in_features
        self.num_class = 101
        self.num_frame = 16

        # mod_list = []
        # for child in model.children():
        #     for c in child.childen():
        #         mod_list.append(c)
        #         mod_list.append(my_adapter)
        # self.adapter_net = nn.ModuleList(mod_list)

        self.net_bottom = nn.Sequential(
            model.blocks[0],
            model.blocks[1],
            model.blocks[2],
            model.blocks[3],
        )

        # self.adapter0 = select_adapter(adapter_mode, 96, self.num_frame)

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
        # x = self.adapter0(x)
        x = self.blocks4(x)
        x = self.adapter1(x)
        x = self.net_top(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        x = x.view(-1, self.num_class)
        return x


def get_kinetics(subset, args):
    """
    Kinetics400のデータセットを取得

    Args:
        subset (str): "train" or "val" or "test"

    Returns:
        pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset: 取得したデータセット
    """

    train_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(args.num_frame),
                transforms.Lambda(lambda x: x / 255.),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                RandomShortSideScale(min_size=256, max_size=320,),
                RandomCrop(224),
                RandomHorizontalFlip(),
            ]),
        ),
        RemoveKey("audio"),
    ])

    val_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(args.num_frame),
                transforms.Lambda(lambda x: x / 255.),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                ShortSideScale(256),
                CenterCrop(224),
            ]),
        ),
        RemoveKey("audio"),
    ])

    transform = val_transform if subset == "val" else train_transform

    root_kinetics = '/mnt/dataset/Kinetics400/'

    if subset == "test":
        dataset = Kinetics(
            data_path=root_kinetics + "test_list.txt",
            video_path_prefix=root_kinetics + 'test/',
            clip_sampler=RandomClipSampler(
                clip_duration=64 / 30),
            video_sampler=RandomSampler,
            decode_audio=False,
            transform=transform,
        )
        return dataset
    else:
        dataset = Kinetics(
            data_path=root_kinetics + subset,
            video_path_prefix=root_kinetics + subset,
            clip_sampler=RandomClipSampler(
                clip_duration=64 / 30),
            video_sampler=RandomSampler,
            decode_audio=False,
            transform=transform,
        )
        return dataset

    return False


def get_ucf101(subset, args):
    """
    ucf101のデータセットを取得

    Args:
        subset (str): "train" or "test"

    Returns:
        pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset: 取得したデータセット
    """
    subset_root_Ucf101 = 'ucfTrainTestlist/trainlist01.txt' if subset == "train" else 'ucfTrainTestlist/testlist.txt'

    train_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(args.num_frame),
                transforms.Lambda(lambda x: x / 255.),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                RandomShortSideScale(min_size=256, max_size=320,),
                RandomCrop(224),
                RandomHorizontalFlip(),
            ]),
        ),
        ApplyTransformToKey(
            key="label",
            transform=transforms.Lambda(lambda x: x - 1),
        ),
        RemoveKey("audio"),
    ])

    val_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(args.num_frame),
                transforms.Lambda(lambda x: x / 255.),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                ShortSideScale(256),
                CenterCrop(224),
            ]),
        ),
        ApplyTransformToKey(
            key="label",
            transform=transforms.Lambda(lambda x: x - 1),
        ),
        RemoveKey("audio"),
    ])

    transform = train_transform if subset == "train" else val_transform

    root_ucf101 = '/mnt/dataset/UCF101/'
    # root_ucf101 = '/mnt/NAS-TVS872XT/dataset/UCF101/'

    dataset = Ucf101(
        data_path=root_ucf101 + subset_root_Ucf101,
        video_path_prefix=root_ucf101 + 'video/',
        clip_sampler=RandomClipSampler(
            clip_duration=16 / 25),
        video_sampler=RandomSampler,
        decode_audio=False,
        transform=transform,
    )

    return dataset


class LimitDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


def make_loader(dataset, args):
    """
    データローダーを作成

    Args:
        dataset (pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset): get_datasetメソッドで取得したdataset

    Returns:
        torch.utils.data.DataLoader: 取得したデータローダー
    """
    loader = DataLoader(LimitDataset(dataset),
                        batch_size=args.batch_size,
                        drop_last=True,
                        num_workers=args.num_workers,
                        shuffle=True)
    return loader


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    https://github.com/machine-perception-robotics-group/attention_branch_network/blob/ced1d97303792ac6d56442571d71bb0572b3efd8/utils/misc.py#L59
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def top1(outputs, targets):
    batch_size = outputs.size(0)
    _, predicted = outputs.max(1)
    return predicted.eq(targets).sum().item() / batch_size


def train_adapter(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    train_dataset = get_ucf101("train", args)
    val_dataset = get_ucf101("val", args)
    train_loader = make_loader(train_dataset, args)
    val_loader = make_loader(val_dataset, args)

    model = ReconstructNetInAdapter(args.adapter_mode)
    model = model.to(device)
    # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    lr = args.learning_rate
    weight_decay = args.weight_decay
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=lr,
    #     momentum=0.9,
    #     weight_decay=5e-4)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    hyper_params = {
        "Dataset": "UCF101",
        "epoch": args.epoch,
        "batch_size": args.batch_size,
        "num_frame": args.num_frame,
        "optimizer": "Adam(0.9, 0.999)",
        "learning late": lr,
        "weight decay": weight_decay,
        "mode": "train temporal adapter",
        "Adapter": "adp:1",
    }

    experiment = Experiment(
        api_key="TawRAwNJiQjPaSMvBAwk4L4pF",
        project_name="feeature-extract",
        workspace="kazukiomi",
    )

    experiment.add_tag('pytorch')
    experiment.log_parameters(hyper_params)

    step = 0
    best_acc = 0

    with tqdm(range(args.epoch)) as pbar_epoch:
        for epoch in pbar_epoch:
            pbar_epoch.set_description("[Epoch %d]" % (epoch))

            """Training mode"""

            train_loss = AverageMeter()
            train_acc = AverageMeter()

            with tqdm(enumerate(train_loader),
                      total=len(train_loader),
                      leave=True) as pbar_train_batch:

                model.train()

                for batch_idx, batch in pbar_train_batch:
                    pbar_train_batch.set_description(
                        "[Epoch :{}]".format(epoch))

                    inputs = batch['video'].to(device)
                    labels = batch['label'].to(device)

                    bs = inputs.size(0)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss.update(loss, bs)
                    train_acc.update(top1(outputs, labels), bs)

                    pbar_train_batch.set_postfix_str(
                        ' | loss_avg={:6.04f} , top1_avg={:6.04f}'
                        ' | batch_loss={:6.04f} , batch_top1={:6.04f}'
                        ''.format(
                            train_loss.avg, train_acc.avg,
                            train_loss.val, train_acc.val,
                        ))

                    experiment.log_metric(
                        "batch_accuracy", train_acc.val, step=step)
                    experiment.log_metric(
                        "batch_loss", train_loss.val, step=step)
                    step += 1

            """Val mode"""
            model.eval()
            val_loss = AverageMeter()
            val_acc = AverageMeter()

            with torch.no_grad():
                for batch_idx, val_batch in enumerate(val_loader):
                    inputs = val_batch['video'].to(device)
                    labels = val_batch['label'].to(device)

                    bs = inputs.size(0)

                    val_outputs = model(inputs)
                    loss = criterion(val_outputs, labels)

                    val_loss.update(loss, bs)
                    val_acc.update(top1(val_outputs, labels), bs)
            """Finish Val mode"""

            pbar_epoch.set_postfix_str(
                ' train_loss={:6.04f} , val_loss={:6.04f}, train_acc={:6.04f}, val_acc={:6.04f}'
                ''.format(
                    train_loss.avg,
                    val_loss.avg,
                    train_acc.avg,
                    val_acc.avg)
            )

            experiment.log_metric("train_accuracy",
                                  train_acc.avg,
                                  step=epoch + 1)
            experiment.log_metric("train_loss",
                                  train_loss.avg,
                                  step=epoch + 1)
            experiment.log_metric("val_accuracy",
                                  val_acc.avg,
                                  step=epoch + 1)
            experiment.log_metric("val_loss",
                                  val_loss.avg,
                                  step=epoch + 1)

    experiment.end()


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10,)
    parser.add_argument("--batch_size", type=int, default=32,)
    parser.add_argument("--num_frame", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=32,)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--adapter_mode", type=str, choices=[
        "video2frame",
        "temporal",
        "space_temporal",
        "efficient_space_temporal"])
    return parser.parse_args()


def model_info(model):
    torchinfo.summary(
        # model,
        # input_size=(1, 3, 16, 224, 224),
        model.blocks[4].res_blocks[0],
        input_size=(1, 96, 16, 14, 14),
        depth=8,
        col_names=["input_size",
                   "output_size"],
        row_settings=("var_names",)
    )


def save_onnx(model):
    # input = torch.randn(1, 3, 16, 224, 224)
    input = torch.randn(1, 96, 16, 14, 14)
    input_names = ["input"]
    output_names = ["output"]
    torch.onnx.export(
        model,
        input,
        "./x3d_m.onnx",
        verbose=True,
        input_names=input_names,
        output_names=output_names)
    # netron.start("./x3d_m.onnx")


def main():
    # config = configparser.ConfigParser()
    args = get_arguments()
    # train_adapter(args)
    # model = ReconstructNetInParallelAdapter(args.adapter_mode)
    # model = ParallelAdapter2D(96, 192)
    # model_info(model)
    model = torch.hub.load(
        'facebookresearch/pytorchvideo', "x3d_m", pretrained=True)
    model_info(model)


if __name__ == '__main__':
    main()
