
from logging import shutdown
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DistributedSampler, RandomSampler

from torchvision import transforms

from pytorchvideo.models import x3d
from pytorchvideo.data import (
    Ucf101,
    RandomClipSampler,
    UniformClipSampler,
    Kinetics,
)

# from torchvision.transforms._transforms_video import (
#     CenterCropVideo,
#     NormalizeVideo,
# )

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
from collections import OrderedDict
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
        # self.CLIP_DURATION = 16 / 25
        self.CLIP_DURATION = (8 * 8) / 30  # (num_frames * sampling_rate)/fps
        self.VIDEO_NUM_SUBSAMPLED = 16  # 事前学習済みモデルに合わせて16→8
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
    Kinetics400のデータセットを取得

    Args:
        subset (str): "train" or "val" or "test"

    Returns:
        pytorchvideo.data.LabeledVideoDataset: 取得したデータセット
    """
    args = Args()
    transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(args.VIDEO_NUM_SUBSAMPLED),
                transforms.Lambda(lambda x: x / 255.),
                Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                ShortSideScale(size=256),
                # RandomShortSideScale(min_size=256, max_size=320,),
                # CenterCropVideo(crop_size=(256, 256)),
                CenterCrop(256),
                # RandomCrop(224),
                RandomHorizontalFlip(),
            ]),
        ),
        ApplyTransformToKey(
            key="label",
            transform=transforms.Lambda(lambda x: x),
        ),
        RemoveKey("audio"),
    ])

    root_kinetics = '/mnt/dataset/Kinetics400/'

    if subset == "test":
        dataset = Kinetics(
            data_path=root_kinetics + "test_list.txt",
            video_path_prefix=root_kinetics + 'test/',
            clip_sampler=RandomClipSampler(clip_duration=args.CLIP_DURATION),
            video_sampler=RandomSampler,
            decode_audio=False,
            transform=transform,
        )
        return dataset
    else:
        dataset = Kinetics(
            data_path=root_kinetics + subset,
            video_path_prefix=root_kinetics + subset,
            clip_sampler=RandomClipSampler(clip_duration=args.CLIP_DURATION),
            video_sampler=RandomSampler,
            decode_audio=False,
            transform=transform,
        )
        return dataset

    return False


def get_ucf101(subset):
    """
    ucf101のデータセットを取得

    Args:
        subset (str): "train" or "test"

    Returns:
        pytorchvideo.data.LabeledVideoDataset: 取得したデータセット
    """
    subset_root_Ucf101 = 'ucfTrainTestlist/trainlist01.txt'
    if subset == "test":
        subset_root_Ucf101 = 'ucfTrainTestlist/testlist.txt'

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
            transform=transforms.Lambda(lambda x: x - 1),
        ),
        RemoveKey("audio"),
    ])

    root_ucf101 = '/mnt/dataset/UCF101/'

    dataset = Ucf101(
        data_path=root_ucf101 + subset_root_Ucf101,
        video_path_prefix=root_ucf101 + 'video/',
        clip_sampler=RandomClipSampler(clip_duration=args.CLIP_DURATION),
        video_sampler=RandomSampler,
        decode_audio=False,
        transform=transform,
    )

    return dataset


def make_loader(dataset):
    """
    データローダーを作成

    Args:
        dataset (pytorchvideo.data.LabeledVideoDataset): get_datasetメソッドで取得したdataset

    Returns:
        torch.utils.data.DataLoader: 取得したデータローダー
    """
    args = Args()
    loader = DataLoader(LimitDataset(dataset),
                        batch_size=args.BATCH_SIZE,
                        drop_last=True,
                        num_workers=args.NUM_WORKERS,
                        shuffle=True)
    return loader


def get_dataset(dataset, subset):
    """
    データセットを取得
    Args:
        dataset (str): "Kinetis400" or "UCF101"
        subset (str): "train" or "val" or "test"

    Returns:
        pytorchvideo.data.LabeledVideoDataset): 取得したデータセット
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
    dataset = get_dataset("Kinetics400", "train")
    loader = make_loader(dataset)
    print("len:{}".format(len(loader)))
    for i, batch in enumerate(loader):
        if i == 0:
            print(batch.keys())
            print(batch['video'].shape)
        print(batch['label'].cpu().numpy())
        if i > 4:
            break


def sample_check():
    """学習済みモデルにサンプルデータ100個を流し込んで挙動を確認"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model("x3d_m", True)
    model = model.to(device)
    dataset = get_dataset("UCF101", "test")

    dataset.video_sampler._num_samples = 100

    sample_loader = make_loader(dataset)

    acc_list = []

    with tqdm(enumerate(sample_loader),
              total=len(sample_loader),
              leave=True) as pbar_batch:

        for batch_idx, batch in pbar_batch:
            inputs = batch["video"].to(device)
            labels = batch["label"].to(device)
            outputs = model(inputs)

            preds = torch.squeeze(outputs.max(dim=1)[1])
            acc = (preds == labels).float().mean().item()
            acc_list.append(acc)
            pbar_batch.set_postfix(OrderedDict(acc=acc))

    print("acc:{}".format(sum(acc_list) / len(acc_list)))


def main():
    # dataset_check("UCF101", "train")
    sample_check()


if __name__ == '__main__':
    main()
