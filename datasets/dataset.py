import torch
from torch.utils.data import DataLoader, dataset
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

import itertools
import os.path as osp

from . import hmdb51


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
                clip_duration=80 / 30),
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
                clip_duration=80 / 30),
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


def get_hmdb51(subset, args):
    root_hmdb = "/mnt/dataset/HMDB51"

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

    transform = train_transform if subset == "train" else val_transform
    split_type = "train" if subset == "train" else "test"

    dataset = hmdb51.Hmdb51(
        label_name_file=osp.join(root_hmdb, "label_name.json"),
        data_path=osp.join(root_hmdb, "testTrainMulti_7030_splits"),
        video_path_prefix=osp.join(root_hmdb, 'video'),
        clip_sampler=RandomClipSampler(
            clip_duration=80 / 30),
        video_sampler=RandomSampler,
        decode_audio=False,
        transform=transform,
        split_type=split_type,
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
