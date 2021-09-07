import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DistributedSampler, RandomSampler

from torchvision import transforms

from pytorchvideo.models import x3d
from pytorchvideo.data import Ucf101, RandomClipSampler, UniformClipSampler, Kinetics

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

from tqdm import tqdm
import itertools
import os
import pickle


class Args:
    def __init__(self):
        self.metadata_path = '/mnt/NAS-TVS872XT/dataset/'
        self.root = self.metadata_path
        self.annotation_path = self.metadata_path
        self.FRAMES_PER_CLIP = 16
        self.STEP_BETWEEN_CLIPS = 16
        self.model = 'x3d_m'
        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 24

        self.clip_duration = 16/25  # 25FPSを想定して16枚
        self.video_num_subsampled = 16  # 16枚抜き出す


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


def get_kinetics(subset):  # trainかvalを指定
    args = Args()
    train_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(args.video_num_subsampled),
                transforms.Lambda(lambda x: x / 255.),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                RandomShortSideScale(min_size=256, max_size=320,),
                RandomCrop(224),
                RandomHorizontalFlip(),
            ]),
        ),
        ApplyTransformToKey(
            key="label",
            transform=transforms.Lambda(lambda x: x),
        ),
        RemoveKey("audio"),
    ])

    root_kinetics = '/mnt/NAS-TVS872XT/dataset/Kinetics400/'

    set = Kinetics(
        data_path=root_kinetics + subset,
        video_path_prefix=root_kinetics + subset,
        clip_sampler=RandomClipSampler(clip_duration=args.clip_duration),
        video_sampler=RandomSampler,
        decode_audio=False,
        transform=train_transform,
    )

    loader = DataLoader(LimitDataset(set),
                        batch_size=args.batch_size,
                        drop_last=True,
                        num_workers=args.num_workers)
    return loader


def get_ucf101(subset):  # trainかvalを指定
    subset_root_Ucf101 = 'ucfTrainTestlist/trainlist01.txt'
    if subset == "val":
        subset_root_Ucf101 = 'ucfTrainTestlist/testlist01.txt'

    args = Args()
    train_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(args.video_num_subsampled),
                transforms.Lambda(lambda x: x / 255.),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                RandomShortSideScale(min_size=256, max_size=320,),
                RandomCrop(224),
                RandomHorizontalFlip(),
            ]),
        ),
        ApplyTransformToKey(
            key="label",
            transform=transforms.Lambda(lambda x: x),
        ),
        RemoveKey("audio"),
    ])

    root_ucf101 = '/mnt/NAS-TVS872XT/dataset/UCF101/'

    set = Ucf101(
        data_path=root_ucf101 + subset_root_Ucf101,
        video_path_prefix=root_ucf101 + 'video/',
        clip_sampler=RandomClipSampler(clip_duration=args.clip_duration),
        video_sampler=RandomSampler,
        decode_audio=False,
        transform=train_transform,
    )

    loader = DataLoader(LimitDataset(set),
                        batch_size=args.batch_size,
                        drop_last=True,
                        num_workers=args.num_workers)
    return loader


def get_dataset(dataset, subset):
    """
    データローダーを取得する

    Args:
        dataset (str): "Kinetis400" or "UCF101"
        subset (str): "train" or "val"

    Returns:
        torch.utils.data.DataLoader: 取得したデータローダー
    """
    if dataset == "Kinetics400":
        return get_kinetics(subset)
    elif dataset == "UCF101":
        return get_ucf101(subset)
    return False


def main():
    loader = get_dataset("UCF101", "train")
    print("len:{}".format(len(loader)))
    for i, batch in enumerate(loader):
        if i == 0:
            print(batch.keys())
            print(batch['video'].shape)
        print(batch['label'].cpu().numpy())
        if i > 4:
            break


if __name__ == '__main__':
    main()
