
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
        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 8  # kinetics:8, ucf101:24

        self.CLIP_DURATION = 16/25  # 25FPSを想定して16枚
        self.VIDEO_NUM_SUBSAMPLED = 16  # 16枚抜き出す
        self.UCF101_NUM_CLASSES = 101
        self.KINETIC400_NUM_CLASSES = 400


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


def get_kinetics(subset):
    """
    Kinetics400のデータローダーを取得

    Args:
        subset (str): "train" or "val"

    Returns:
        torch.utils.data.DataLoader: 取得したデータローダー
    """
    args = Args()
    transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(args.VIDEO_NUM_SUBSAMPLED),
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
        clip_sampler=RandomClipSampler(clip_duration=args.CLIP_DURATION),
        video_sampler=RandomSampler,
        decode_audio=False,
        transform=transform,
    )

    loader = DataLoader(LimitDataset(set),
                        batch_size=args.BATCH_SIZE,
                        drop_last=True,
                        num_workers=args.NUM_WORKERS)
    return loader, set


def get_ucf101(subset):
    """
    ucf101のデータローダーを取得

    Args:
        subset (str): "train" or "val"

    Returns:
        torch.utils.data.DataLoader: 取得したデータローダー
    """
    subset_root_Ucf101 = 'ucfTrainTestlist/trainlist01.txt'
    if subset == "val":
        subset_root_Ucf101 = 'ucfTrainTestlist/testlist01.txt'

    args = Args()
    transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(args.VIDEO_NUM_SUBSAMPLED),
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
        clip_sampler=RandomClipSampler(clip_duration=args.CLIP_DURATION),
        video_sampler=RandomSampler,
        decode_audio=False,
        transform=transform,
    )

    loader = DataLoader(LimitDataset(set),
                        batch_size=args.BATCH_SIZE,
                        drop_last=True,
                        num_workers=args.NUM_WORKERS)
    return loader


def get_dataloader(dataset, subset):
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


def get_model(model_name, pretrained):
    """
    pytorchvideoからモデルを取得

    Args:
        model (str): "x3d_m"(UCF101用) or "slow_r50"(Kinetics400用)
        pretrained (bool): "True" or "False"

    Returns:
        model: 取得したモデル
    """
    model = torch.hub.load(
        'facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
    do_fine_tune = True
    if do_fine_tune:
        for param in model.parameters():
            param.requires_grad = False
    if model_name == "x3d_m":
        model.blocks[5].proj = nn.Linear(
            model.blocks[5].proj.in_features, 101)
    return model


def dataset_check(dataset, subset):
    """
    取得したローダーの挙動を確認
    Args:
        dataset (str): "Kinetis400" or "UCF101"
        subset (str): "train" or "val"

    """
    loader = get_dataloader("Kinetics400", "train")
    print("len:{}".format(len(loader)))
    for i, batch in enumerate(loader):
        if i == 0:
            print(batch.keys())
            print(batch['video'].shape)
        print(batch['label'].cpu().numpy())
        if i > 4:
            break


# def model_check(model_name, subset):


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model("slow_r50", True)
    model = model.to(device)
    # data = torch.randn(2, 3, 16, 224, 224).to(device)
    # print(model(data))
    loader, dataset = get_dataloader("Kinetics400", "val")
    # print(dataset.num_videos)
    # print(type(dataset))
    # print(type(dataset.video_sampler))
    # print(type(dataset.video_sampler._num_samples))
    dataset.video_sampler._num_samples = 100
    # print(dataset.num_videos)

    sample_loader = DataLoader(LimitDataset(dataset),
                               batch_size=16,
                               drop_last=True,
                               num_workers=8)
    print(len(sample_loader))
    # for i, batch in enumerate(sample_loader):
    #     print(i)
    # x = iter(sample_loader).__next__()

    with tqdm(enumerate(sample_loader),
              total=len(sample_loader),
              leave=True) as pbar_batch:
        for batch_idx, batch in pbar_batch:
            if batch_idx == 0:
                inputs = batch["video"].to(device)
                labels = batch["label"].to(device)
                # batch_size = inputs.size(0)
                outputs = model(inputs)
                # print(outputs.shape)
                preds = torch.squeeze(outputs.max(dim=1)[1])
                acc = (preds == labels).float().mean().item()
                print(acc)


if __name__ == '__main__':
    main()
