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


class Args:
    def __init__(self):
        self.NUM_EPOCH = 2
        self.FRAMES_PER_CLIP = 16
        self.STEP_BETWEEN_CLIPS = 16
        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 32
        # self.CLIP_DURATION = 16 / 25
        # (num_frames * sampling_rate)/fps
        self.kinetics_clip_duration = (8 * 8) / 30
        self.ucf101_clip_duration = 16 / 25
        self.VIDEO_NUM_SUBSAMPLED = 16
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
        pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset: 取得したデータセット
    """
    args = Args()
    train_transform = Compose([
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
        RemoveKey("audio"),
    ])

    val_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(args.VIDEO_NUM_SUBSAMPLED),
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
                clip_duration=args.kinetics_clip_duration),
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
                clip_duration=args.kinetics_clip_duration),
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
        pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset: 取得したデータセット
    """
    subset_root_Ucf101 = 'ucfTrainTestlist/trainlist01.txt' if subset == "train" else 'ucfTrainTestlist/testlist.txt'
    # if subset == "test":
    #     subset_root_Ucf101 = 'ucfTrainTestlist/testlist.txt'

    args = Args()
    train_transform = Compose([
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

    val_transform = Compose([
        ApplyTransformToKey(
            key="video",
            transform=Compose([
                UniformTemporalSubsample(args.VIDEO_NUM_SUBSAMPLED),
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
            clip_duration=args.ucf101_clip_duration),
        video_sampler=RandomSampler,
        decode_audio=False,
        transform=transform,
    )

    return dataset


def make_loader(dataset):
    """
    データローダーを作成

    Args:
        dataset (pytorchvideo.data.labeled_video_dataset.LabeledVideoDataset): get_datasetメソッドで取得したdataset

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

# def get_dataset(dataset, subset):
#     """
#     データセットを取得
#     Args:
#         dataset (str): "Kinetis400" or "UCF101"
#         subset (str): "train" or "val" or "test"

#     Returns:
#         pytorchvideo.data.LabeledVideoDataset): 取得したデータセット
#     """
#     if dataset == "Kinetics400":
#         return get_kinetics(subset)
#     elif dataset == "UCF101":
#         return get_ucf101(subset)
#     return False


# def get_model(model_name, pretrained):
#     """
#     pytorchvideoからモデルを取得

#     Args:
#         model (str): "x3d_m"(UCF101用) or "slow_r50"(Kinetics400用)
#         pretrained (bool): "True" or "False"

#     Returns:
#         model: 取得したモデル
#     """
#     model = torch.hub.load(
#         'facebookresearch/pytorchvideo', model_name, pretrained=pretrained)
#     do_fine_tune = True
#     if do_fine_tune:
#         for param in model.parameters():
#             param.requires_grad = False
#     if model_name == "x3d_m":
#         model.blocks[5].proj = nn.Linear(
#             model.blocks[5].proj.in_features, 101)
#     return model


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


def save_checkpoint(state, is_best, filename, best_model_file, dir_data_name):
    file_path = osp.join(dir_data_name, filename)
    if not os.path.exists(dir_data_name):
        os.makedirs(dir_data_name)
    torch.save(state.state_dict(), file_path)
    if is_best:
        shutil.copyfile(file_path, osp.join(dir_data_name, best_model_file))


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

        # # 学習させるパラメータ名
        # self.update_param_names = ["linear.weight", "linear.bias"]
        # # 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
        # for name, param in self.named_parameters():
        #     if name in self.update_param_names:
        #         param.requires_grad = True
        #         # print(name)
        #     else:
        #         param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net_bottom(x)
        x = self.blocks4(x)
        x = self.net_top(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.linear(x)
        x = x.view(-1, self.num_class)
        return x


def train():
    args = Args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = get_ucf101("train")
    val_dataset = get_ucf101("val")
    train_loader = make_loader(train_dataset)
    val_loader = make_loader(val_dataset)

    model = ReconstructNet()
    model = model.to(device)
    # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    lr = 0.01
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    hyper_params = {
        "Dataset": "UCF101",
        "epoch": args.NUM_EPOCH,
        "batch_size": args.BATCH_SIZE,
        "num_frame": args.VIDEO_NUM_SUBSAMPLED,
        "learning late": lr,
        "mode": "finetune",
        "Adapter": "adp:0, adp:1",
    }

    experiment = Experiment(
        api_key="TawRAwNJiQjPaSMvBAwk4L4pF",
        project_name="feeature-extract",
        workspace="kazukiomi",
    )

    experiment.add_tag('pytorch')
    experiment.log_parameters(hyper_params)

    num_epochs = args.NUM_EPOCH

    step = 0
    best_acc = 0

    with tqdm(range(num_epochs)) as pbar_epoch:
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

            """save model"""
            if best_acc < val_acc.avg:
                best_acc = val_acc.avg
                is_best = True
            else:
                is_best = False

            save_checkpoint(
                model,
                is_best,
                filename="finetune_checkpoint.pth",
                best_model_file="finetune_best.pth",
                dir_data_name="adapter_2d/UCF101")

            pbar_epoch.set_postfix_str(
                ' train_loss={:6.04f} , val_loss={:6.04f}, train_acc={:6.04f}, val_acc={:6.04f}'
                ''.format(
                    train_loss.avg,
                    val_loss.avg,
                    train_acc.avg,
                    val_acc.avg)
            )

            experiment.log_metric("epoch_train_accuracy",
                                  train_acc.avg,
                                  step=epoch + 1)
            experiment.log_metric("epoch_train_loss",
                                  train_loss.avg,
                                  step=epoch + 1)
            experiment.log_metric("val_accuracy",
                                  val_acc.avg,
                                  step=epoch + 1)
            experiment.log_metric("val_loss",
                                  val_loss.avg,
                                  step=epoch + 1)


def main():
    train()


if __name__ == '__main__':
    main()
