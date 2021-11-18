from re import X
from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler


from torchvision import transforms

import pytorchvideo
from pytorchvideo.models import x3d
from pytorchvideo.data import (
    Ucf101,
    RandomClipSampler,
    UniformClipSampler,
    Kinetics,
    SSv2,
    kinetics
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
import time


class TestAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def ident(self, x):
        return x

    def forward(self, x):
        x = self.ident(x)
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


class MyNet(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        model = torch.hub.load(
            'facebookresearch/pytorchvideo', "x3d_m", pretrained=True)
        self.dim_features = model.blocks[5].proj.in_features
        self.num_frame = args.num_frame
        self.class_dict = make_class_dict(args, config)

        mod_list = []
        for child in model.children():
            for g_child in child.children():
                if isinstance(
                        g_child, pytorchvideo.models.head.ResNetBasicHead) == False:
                    mod_list.append(g_child)
                    mod_list.append(TestAdapter())

        self.module_list = nn.ModuleList(mod_list)
        self.head_bottom = nn.Sequential(
            model.blocks[5].pool,
            model.blocks[5].dropout
        )

        head_dict = {}
        for name in args.dataset_names:
            head = nn.Linear(self.dim_features, self.class_dict[name])
            head_dict[name] = head
        self.head_top_dict = head_dict

    def forward(self, x: torch.Tensor, domain) -> torch.Tensor:
        for f in self.module_list:
            x = f(x)
        x = self.head_bottom(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.head_top_dict[domain](x)
        x = x.view(-1, self.class_dict[domain])
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


def make_named_loader(dataset_name, subset, args):
    if dataset_name == "Kinetics":
        dataset = get_kinetics(subset, args)
    elif dataset_name == "UCF101":
        dataset = get_ucf101(subset, args)
    else:
        raise NameError("データセット名が正しくないです")
    loader = make_loader(dataset, args)
    return loader


def loader_list(dataset_list, args):
    train_loader_list = []
    val_loader_list = []
    for dataset_name in dataset_list:
        train_loader_list.append(
            make_named_loader(
                dataset_name, "train", args))
        val_loader_list.append(make_named_loader(dataset_name, "val", args))
    return train_loader_list, val_loader_list


def loader_dict(dataset_list, args):
    train_loader_list = []
    val_loader_list = []
    for dataset_name in dataset_list:
        train_loader_list.append(
            make_named_loader(
                dataset_name, "train", args))
        val_loader_list.append(make_named_loader(dataset_name, "val", args))
    train_loader_dict = dict(zip(dataset_list, train_loader_list))
    val_loader_dict = dict(zip(dataset_list, val_loader_list))
    return train_loader_dict, val_loader_dict


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


def train(args):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    dataset_name_list = ["UCF101", "Kinetics"]
    train_loader_list, val_loader_list = loader_list(dataset_name_list, args)
    loader_itr_list = []
    for d in train_loader_list:
        loader_itr_list.append(iter(d))

    model = MyNet(args.adapter_mode)
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
        "Dataset": "UCF101, Kinetics",
        # "epoch": args.epoch,
        "batch_size": args.batch_size,
        "num_frame": args.num_frame,
        "optimizer": "Adam(0.9, 0.999)",
        "learning late": lr,
        "weight decay": weight_decay,
        # "mode": args.adapter_mode,
        # "Adapter": "adp:1",
    }

    experiment = Experiment(
        api_key="TawRAwNJiQjPaSMvBAwk4L4pF",
        project_name="feeature-extract",
        workspace="kazukiomi",
    )

    experiment.add_tag('pytorch')
    experiment.log_parameters(hyper_params)

    step = 0
    # best_acc = 0

    num_iters = 100

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    for _ in dataset_name_list:
        train_acc_list.append(AverageMeter())
        train_loss_list.append(AverageMeter())
        val_acc_list.append(AverageMeter())
        val_loss_list.append(AverageMeter())

    with tqdm(range(num_iters)) as pbar_itrs:
        for itr in pbar_itrs:
            pbar_itrs.set_description("[Iteration %d]" % (itr))

            """Training mode"""

            model.train()
            batch_list = []
            for i, loader in enumerate(loader_itr_list):
                try:
                    batch = next(loader)
                    batch_list.append(batch)
                except StopIteration:
                    loader_itr_list[i] = iter(train_loader_list[i])
                    batch = next(loader_itr_list[i])
                    batch_list.append(batch)

            for i, batch in enumerate(batch_list):
                inputs = batch['video'].to(device)
                labels = batch['label'].to(device)

                bs = inputs.size(0)

                if i == 0:
                    optimizer.zero_grad()

                outputs = model(inputs, dataset_name_list[i])
                loss = criterion(outputs, labels)
                loss.backward()

                if i == len(dataset_name_list) - 1:
                    optimizer.step()

                train_loss_list[i].update(loss, bs)
                train_acc_list[i].update(top1(outputs, labels), bs)

            for i, name in enumerate(dataset_name_list):
                experiment.log_metric(
                    "batch_accuracy_" + name, train_acc_list[i].val, step=step)
                experiment.log_metric(
                    "batch_loss_" + name, train_loss_list[i].val, step=step)
            step += 1

            if (itr + 1) % 30 == 0:
                """Val mode"""
                model.eval()

                # val_acc_list = []
                # val_loss_list = []
                # for _ in dataset_name_list:
                #     val_acc_list.append(AverageMeter)
                #     val_loss_list.append(AverageMeter)

                with torch.no_grad():
                    for i, loader in enumerate(val_loader_list):
                        for val_batch in loader:
                            inputs = val_batch['video'].to(device)
                            labels = val_batch['label'].to(device)

                            bs = inputs.size(0)

                            val_outputs = model(
                                inputs, dataset_name_list[i])
                            loss = criterion(val_outputs, labels)

                            val_loss_list[i].update(loss, bs)
                            val_acc_list[i].update(
                                top1(val_outputs, labels), bs)

                    for i, name in enumerate(dataset_name_list):
                        experiment.log_metric(
                            "train_accuracy_" + name, train_acc_list[i].avg, step=step)
                        experiment.log_metric(
                            "train_loss_" + name, train_loss_list[i].avg, step=step)
                        experiment.log_metric(
                            "val_accuracy_" + name, val_acc_list[i].avg, step=step)
                        experiment.log_metric(
                            "val_loss_" + name, val_loss_list[i].avg, step=step)
                        train_acc_list[i].reset()
                        train_loss_list[i].reset()
                        val_acc_list[i].reset()
                        val_loss_list[i].reset()

                """Finish Val mode"""
                model.train()

    experiment.end()


def model_info(model):
    torchinfo.summary(
        model,
        input_size=(1, 3, 16, 224, 224),
        # model.blocks[4].res_blocks[0],
        # input_size=(1, 96, 16, 14, 14),
        depth=4,
        col_names=["input_size",
                   "output_size"],
        row_settings=("var_names",)
    )


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
    parser.add_argument("--dataset_names",
                        nargs="*",
                        default=["UCF101", "Kinetics"])
    return parser.parse_args()


def test_batch_process():
    args = get_arguments()
    dataset_name_list = ["UCF101", "Kinetics"]
    train_loader_list, val_loader_list = loader_list(dataset_name_list, args)
    loader_iters = []
    for d in train_loader_list:
        loader_iters.append(iter(d))

    data_list = []
    for i, loader in enumerate(loader_iters):
        try:
            data = next(loader)
            data_list.append(data)
        except StopIteration:
            loader_iters[i] = iter(train_loader_list[i])
            data = next(loader_iters[i])
            data_list.append(data)
    print(len(data_list))
    print(data_list[0]["video"].shape)


def adapter_test():
    model = torch.hub.load(
        'facebookresearch/pytorchvideo', "x3d_m", pretrained=True)
    for child in model.children():
        for c in child.children():
            print(type(c))
            if isinstance(
                    c, pytorchvideo.models.head.ResNetBasicHead) == False:
                print("Yes")
            print("------------------------------------------------------------")


def make_class_dict(args, config):
    config.read("config.ini")
    num_class_dict = {}
    for name in args.dataset_names:
        num_class_dict[name] = int(config[name]["num_class"])
    return num_class_dict


def main():
    args = get_arguments()
    config = configparser.ConfigParser()
    config.read("config.ini")
    # train(args)
    model = MyNet(args, config)
    # model_info(model)
    # adapter_test()
    input = torch.randn(1, 3, 16, 224, 224)
    out = model(input, "UCF101")
    print(out.shape)


if __name__ == '__main__':
    main()
