from comet_ml import Experiment
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler
import torch.nn.functional as F


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
import os.path as osp
import shutil
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


class Swish(nn.Module):
    """
    Wrapper for the Swish activation function.
    Imported from https://github.com/facebookresearch/pytorchvideo/blob/168e16859a6029ef8ebeb476f9163bebb6c6b87d/pytorchvideo/layers/swish.py#L7
    """

    def forward(self, x):
        return SwishFunction.apply(x)


class SwishFunction(torch.autograd.Function):
    """
    Implementation of the Swish activation function: x * sigmoid(x).
    Searching for activation functions. Ramachandran, Prajit and Zoph, Barret
    and Le, Quoc V. 2017
    """

    @staticmethod
    def forward(ctx, x):
        result = x * torch.sigmoid(x)
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


class Adapter(nn.Module):
    def __init__(self, feature_list, frame):
        super().__init__()
        self.channel = feature_list[0]
        self.height = feature_list[1]
        # self.norm1 = nn.LayerNorm([channel, frame, height, height])
        self.norm1 = nn.BatchNorm3d(self.channel)
        self.act = nn.ReLU()  # TODO(omi): implement swish

    def swap(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def unswap(self, input, bs, frames=16) -> torch.Tensor:
        return input

    def forward(self, x):
        batch_size, channel, frames, height, width = x.size()

        out = self.swap(x)
        out = self.conv1(out)
        out = self.unswap(out, batch_size, frames)
        out = self.norm1(out)

        out += x
        out = self.act(out)

        return out


class TestAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def ident(self, x):
        return x

    def forward(self, x):
        x = self.ident(x)
        return x


class FrameWise2dAdapter(Adapter):
    def __init__(self, feature_list, frame):
        super().__init__(feature_list, frame)
        self.conv1 = nn.Conv2d(self.channel, self.channel, 1)

    def swap(self, input: torch.Tensor) -> torch.Tensor:
        # input: B,C,T,H,W
        batch_size, channel, frames, height, width = input.size()
        # B,C,T,H,W --> B,T,C,H,W
        output = input.permute(0, 2, 1, 3, 4)
        # B,T,C,H,W --> BT,C,H,W
        output = output.reshape(batch_size * frames, channel, height, width)
        return output

    def unswap(self, input, bs, frames=16) -> torch.Tensor:
        # input: BT,C,H,W
        batchs_frames, channel, height, width = input.size()
        frames = int(batchs_frames / bs)
        # BT,C,H,W --> B,T,C,H,W
        output = input.reshape(bs, frames, channel, height, width)
        # B,T,C,H,W --> B,C,T,H,W
        output = output.permute(0, 2, 1, 3, 4)
        return output


class ChannelWise3dAdapter(Adapter):
    def __init__(self, feature_list, frame):
        super().__init__(feature_list, frame)
        self.conv1 = nn.Conv2d(frame, frame, 1)

    def swap(self, input: torch.Tensor) -> torch.Tensor:
        # input: B,C,T,H,W
        batch_size, channel, frames, height, width = input.size()
        # B,C,T,H,W --> BC,T,H,W
        output = input.reshape(batch_size * channel, frames, width, height)
        return output

    def unswap(self, input, bs, frames=16) -> torch.Tensor:
        # input: BC,T,H,W
        batchs_channels, frames, height, width = input.size()
        channel = int(batchs_channels / bs)
        # BC,T,H,W --> B,C,T,H,W
        output = input.reshape(bs, channel, frames, height, width)
        return output


class VideoAdapter(Adapter):
    def __init__(self, feature_list, frame):
        super().__init__(feature_list, frame)
        self.conv1 = nn.Conv3d(self.channel, self.channel,
                               3, padding=(1, 1, 1))


class EfficientSpaceTemporalAdapter(nn.Module):
    def __init__(self, feature_list, frame):
        super().__init__()
        self.video2frame_adapter = FrameWise2dAdapter(feature_list, frame)
        self.temporal_adapter = ChannelWise3dAdapter(feature_list, frame)

    def forward(self, x):
        out = self.video2frame_adapter(x)
        out = self.temporal_adapter(x)
        return out


def select_adapter(adp_mode, feature_list, frame):
    if adp_mode == "video2frame":
        adp = FrameWise2dAdapter(feature_list, frame)
    elif adp_mode == "temporal":
        adp = ChannelWise3dAdapter(feature_list, frame)
    elif adp_mode == "space_temporal":
        adp = VideoAdapter(feature_list, frame)
    elif adp_mode == "efficient_space_temporal":
        adp = EfficientSpaceTemporalAdapter(feature_list, frame)
    else:
        raise NameError("invalide adapter name")
    return adp


class MyHeadDict(nn.Module):
    def __init__(self, in_channel, dataset_names, class_dict):
        super().__init__()
        self.head = nn.ModuleDict({})

        head_dict = {}
        for name in dataset_names:
            head = nn.Linear(in_channel, class_dict[name])
            head_dict[name] = head
        self.head.update(head_dict)

    def forward(self, x, domain):
        x = self.head[domain](x)
        return x


class MyAdapterDict(nn.Module):
    def __init__(self, args, feature_list):
        super().__init__()
        channel = feature_list[0]
        height = feature_list[1]
        self.adapter = nn.ModuleDict({})

        adp_dict = {}
        for name in args.dataset_names:
            adp = select_adapter(
                args.adp_mode, feature_list, args.num_frames)
            adp_dict[name] = adp
        self.adapter.update(adp_dict)

        self.norm = nn.LayerNorm([channel, args.num_frames, height, height])

    def forward(self, x, domain):
        x = self.adapter[domain](x)
        x = self.norm(x)
        return x


def make_mod_list(model, args):
    mod_list = []
    feature_list = args.feature_list
    index = 0
    module = model.children().__next__()
    if args.adp_place == "stages":
        for child in module.children():
            if isinstance(
                    child, pytorchvideo.models.head.ResNetBasicHead) == False:
                mod_list.append(child)
                mod_list.append(
                    MyAdapterDict(args, feature_list[index]))
                index += 1
    elif args.adp_place == "all":
        for child in module.children():
            if isinstance(child, pytorchvideo.models.stem.ResNetBasicStem):
                mod_list.append(child)
                mod_list.append(
                    MyAdapterDict(args, feature_list[index]))
                index += 1
            elif isinstance(child, pytorchvideo.models.head.ResNetBasicHead):
                pass
            elif isinstance(child, pytorchvideo.models.resnet.ResStage):
                g = child.children()
                for g_c in g:
                    for i, g_g in enumerate(g_c):
                        mod_list.append(g_g)
                        mod_list.append(MyAdapterDict(
                            args, feature_list[index]))
                index += 1
            else:
                raise NameError("Failed to create module list.")
    elif args.adp_place == "blocks":
        for child in module.children():
            if isinstance(child, pytorchvideo.models.stem.ResNetBasicStem):
                mod_list.append(child)
            elif isinstance(child, pytorchvideo.models.head.ResNetBasicHead):
                pass
            elif isinstance(child, pytorchvideo.models.resnet.ResStage):
                g = child.children()
                for g_c in g:
                    for i, g_g in enumerate(g_c):
                        mod_list.append(g_g)
                        if i != len(g_c) - 1:
                            mod_list.append(MyAdapterDict(
                                args, feature_list[index + 1]))
                index += 1
            else:
                raise NameError("Failed to create module list.")
    elif args.adp_place == "No":
        for child in module.children():
            if isinstance(
                    child, pytorchvideo.models.head.ResNetBasicHead) == False:
                mod_list.append(child)
    else:
        raise NameError("invalide adapter place")
    return mod_list


class MyNet(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        model = torch.hub.load(
            'facebookresearch/pytorchvideo', "x3d_m", pretrained=args.pretrained)
        self.dim_features = model.blocks[5].proj.in_features
        self.num_frames = args.num_frames
        self.class_dict = make_class_dict(args, config)

        mod_list = make_mod_list(model, args)

        self.module_list = nn.ModuleList(mod_list)
        self.head_bottom = nn.Sequential(
            model.blocks[5].pool,
            model.blocks[5].dropout
        )

        self.head_top_dict = MyHeadDict(self.dim_features,
                                        args.dataset_names,
                                        self.class_dict)

        # for name, param in self.named_parameters():
        #     print(name)

    def forward(self, x: torch.Tensor, domain) -> torch.Tensor:
        for f in self.module_list:
            if isinstance(f, MyAdapterDict):
                x = f(x, domain)
            else:
                x = f(x)
            # torchinfoで確認できないので確認用
            print(type(f))
            print(x.shape)

        x = self.head_bottom(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.head_top_dict(x, domain)
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
                UniformTemporalSubsample(args.num_frames),
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
                UniformTemporalSubsample(args.num_frames),
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
                UniformTemporalSubsample(args.num_frames),
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
                UniformTemporalSubsample(args.num_frames),
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


def make_loader(dataset, args, batch_size):
    """
    データローダーを作成

    Args:
        dataset (pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset): get_datasetメソッドで取得したdataset

    Returns:
        torch.utils.data.DataLoader: 取得したデータローダー
    """
    loader = DataLoader(LimitDataset(dataset),
                        batch_size=batch_size,
                        drop_last=True,
                        num_workers=args.num_workers,
                        shuffle=True)
    return loader


def make_named_loader(dataset_name, subset, args, batch_size):
    if dataset_name == "Kinetics":
        dataset = get_kinetics(subset, args)
    elif dataset_name == "UCF101":
        dataset = get_ucf101(subset, args)
    else:
        raise NameError("データセット名が正しくないです")
    loader = make_loader(dataset, args, batch_size)
    return loader


def loader_list(args):
    """データローダーのリストを作成"""
    train_loader_list = []
    val_loader_list = []
    dataset_list = args.dataset_names
    batch_size_dict = dict(
        zip(args.dataset_names, list(map(int, args.batch_size_list))))
    for dataset_name in dataset_list:
        train_loader_list.append(make_named_loader(
            dataset_name, "train", args, batch_size_dict[dataset_name]))
        val_loader_list.append(make_named_loader(
            dataset_name, "val", args, batch_size_dict[dataset_name]))
    return train_loader_list, val_loader_list


def loader_dict(dataset_list, args):
    train_loader_list = []
    val_loader_list = []
    for dataset_name in dataset_list:
        train_loader_list.append(make_named_loader(
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


def save_checkpoint(state, filename, dir_name, ex_name):
    dir_name = osp.join(dir_name, ex_name)
    file_path = osp.join(dir_name, filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    torch.save(state.state_dict(), file_path)


def train(args, config):
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    dataset_name_list = args.dataset_names
    train_loader_list, val_loader_list = loader_list(args)
    loader_itr_list = []
    for d in train_loader_list:
        loader_itr_list.append(iter(d))

    model = MyNet(args, config)
    model_path = "checkpoint/5000_checkpoint.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    lr = args.learning_rate
    weight_decay = args.weight_decay
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay)
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=lr,
    #     betas=(0.9, 0.999),
    #     weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.sche_list, args.lr_gamma
    )
    criterion = nn.CrossEntropyLoss()

    hyper_params = {
        "Dataset": args.dataset_names,
        "Iteration": args.iteration,
        "batch_size": args.batch_size_list,
        # "optimizer": "Adam(0.9, 0.999)",
        "learning late": lr,
        "scheuler": args.sche_list,
        "lr_gamma": args.lr_gamma,
        "weight decay": weight_decay,
        "mode": args.adp_mode,
        "adp place": args.adp_place,
        "pretrained": args.pretrained,
    }

    experiment = Experiment(
        api_key="TawRAwNJiQjPaSMvBAwk4L4pF",
        project_name="feeature-extract",
        workspace="kazukiomi",
    )

    experiment.add_tag('pytorch')
    experiment.log_parameters(hyper_params)

    step = 0

    num_iters = args.iteration

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

            if itr % 8 == 0:
                optimizer.zero_grad()
            for i, batch in enumerate(batch_list):
                inputs = batch['video'].to(device)
                labels = batch['label'].to(device)

                bs = inputs.size(0)

                outputs = model(inputs, dataset_name_list[i])
                loss = criterion(outputs, labels)
                loss.backward()

                train_loss_list[i].update(loss, bs)
                train_acc_list[i].update(top1(outputs, labels), bs)
            if itr % 8 == 7:
                optimizer.step()
            scheduler.step()

            for i, name in enumerate(dataset_name_list):
                experiment.log_metric(
                    "batch_accuracy_" + name, train_acc_list[i].val, step=step)
                experiment.log_metric(
                    "batch_loss_" + name, train_loss_list[i].val, step=step)
            step += 1

            if (itr + 1) % 1000 == 0:
                """Val mode"""
                model.eval()

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

                """save model"""
                save_checkpoint(model,
                                filename=str(itr + 1) + "_checkpoint.pth",
                                dir_name="checkpoint",
                                ex_name=osp.join(args.adp_mode, "ex0"))

    experiment.end()


def val_x3d_base(args):
    """
    学習済みモデルのKineticsの性能評価
    """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    val_dataset = get_kinetics("val", args)
    val_loader = make_loader(val_dataset, args, args.batch_size)
    model = torch.hub.load(
        'facebookresearch/pytorchvideo', "x3d_m", pretrained=True)
    model = model.to(device)
    torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()

    hyper_params = {
        "Dataset": "Kinetics val",
        "mode": "val x3d_m with Kinetics",
    }
    experiment = Experiment(
        api_key="TawRAwNJiQjPaSMvBAwk4L4pF",
        project_name="feeature-extract",
        workspace="kazukiomi",
    )
    experiment.add_tag('pytorch')
    experiment.log_parameters(hyper_params)

    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()

    with torch.no_grad():
        with tqdm(enumerate(val_loader), total=len(val_loader), leave=True) as pbar_batch:
            for batch_idx, val_batch in pbar_batch:
                inputs = val_batch['video'].to(device)
                labels = val_batch['label'].to(device)

                bs = inputs.size(0)

                val_outputs = model(inputs)
                loss = criterion(val_outputs, labels)
                val_loss.update(loss, bs)
                val_acc.update(top1(val_outputs, labels), bs)
    experiment.log_metric("val_accuracy", val_acc.avg,)

    return print(val_acc.avg)


def model_info(model):
    torchinfo.summary(
        model,
        input_size=(1, 3, 16, 224, 224),
        # model.blocks[4].res_blocks[0],
        # input_size=(1, 96, 16, 14, 14),
        depth=8,
        col_names=["input_size",
                   "output_size"],
        row_settings=("var_names",)
    )


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=16000,)
    parser.add_argument("--epoch", type=int, default=10,)
    parser.add_argument("--batch_size", type=int, default=32,)
    parser.add_argument("--batch_size_list", nargs="*", default=[32, 32])
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=32,)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--sche_list", nargs="*",
                        default=[5000, 10000, 15000])
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--pretrained", type=str, default="True",)
    parser.add_argument("--adp_place", type=str, default="stages",
                        choices=["stages", "blocks", "all", "No"])
    parser.add_argument("--adp_mode", type=str,
                        choices=["video2frame", "temporal", "space_temporal", "efficient_space_temporal"])
    parser.add_argument("--dataset_names", nargs="*",
                        default=["UCF101", "Kinetics"])
    parser.add_argument("--feature_list", nargs="*",
                        default=[[24, 112], [24, 56], [48, 28], [96, 14], [192, 7]])
    parser.add_argument("--cuda", type=str, default="cuda:2")
    return parser.parse_args()


def test_batch_process():
    """
    ローダーリストからバッチを取り出す挙動の確認用
    """
    args = get_arguments()
    dataset_name_list = ["UCF101", "Kinetics"]
    train_loader_list, _ = loader_list(dataset_name_list, args)
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


def make_class_dict(args, config):
    """
    keyがクラス名（str）でvalueがクラス数（int）のdictを作成
    """
    config.read("config.ini")
    num_class_dict = {}
    for name in args.dataset_names:
        num_class_dict[name] = int(config[name]["num_class"])
    return num_class_dict


def make_lr_list(args, config):
    model = MyNet(args, config)
    lr_list = []
    for name, model in model.named_parameters():
        # print(name)
        name = name.rsplit(".", 1)
        # print(name)
        param = "model." + name[0] + ".parameters()"
        if param not in lr_list:
            # print(param)
            lr_list.append(param)
    print(len(lr_list))
    print(lr_list[0])
    exec(lr_list[0])


def main():
    args = get_arguments()
    config = configparser.ConfigParser()
    config.read("config.ini")
    """train"""
    # train(args, config)

    """model check (torchinfo)"""
    # model = MyAdapterDict(args.adp_mode, 96, args.dataset_names)
    # model = torch.hub.load(
    # 'facebookresearch/pytorchvideo', "x3d_m", pretrained = args.pretrained)
    # # print(model)
    # model_info(model)

    """model_check (実際に入力を流す，dict使うとtorchinfoできないから)"""
    model = MyNet(args, config)
    input = torch.randn(1, 3, 16, 224, 224)
    # input = torch.randn(1, 2048)
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input = input.to(device)
    out = model(input, args.dataset_names[1])
    print(out.shape)

    # make_lr_list(args, config)


if __name__ == '__main__':
    # print(torch.cuda.get_arch_list())
    main()
