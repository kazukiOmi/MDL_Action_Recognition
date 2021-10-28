from comet_ml import Experiment

import torch
import torch.nn as nn
import torchvision
import torchinfo
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DistributedSampler, RandomSampler

from torchvision import transforms

from pytorchvideo.models import x3d
from pytorchvideo.data import (
    Ucf101,
    RandomClipSampler,
    UniformClipSampler,
    Kinetics
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
import argparse


class Args:
    def __init__(self):
        self.metadata_path = '/mnt/NAS-TVS872XT/dataset/'
        self.root = self.metadata_path
        self.annotation_path = self.metadata_path
        self.NUM_EPOCH = 30
        self.FRAMES_PER_CLIP = 16
        self.STEP_BETWEEN_CLIPS = 16
        self.BATCH_SIZE = 32
        self.NUM_WORKERS = 32
        # self.CLIP_DURATION = 16 / 25
        # (num_frames * sampling_rate)/fps
        self.kinetics_clip_duration = (8 * 8) / 30
        self.ucf101_clip_duration = 16 / 25
        self.VIDEO_NUM_SUBSAMPLED = 8
        self.UCF101_NUM_CLASSES = 101
        self.KINETIC400_NUM_CLASSES = 400


class Adapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.conv1(out)

        out += residual
        out = self.bn2(out)

        return out


class VideoToFrame(nn.Module):
    def __init__(self):
        super().__init__()

    def make_new_inputs(self, inputs):
        """
        動画データを画像データに分割

        Args:
            inputs (torch.Tensor): inputs
        Returns:
            new_inputs torch.Tensor: new_inputs
        """

        batch_size = inputs.size(0)
        num_frame = inputs.size(2)

        inputs = inputs.permute(0, 2, 1, 3, 4)
        outputs = inputs.reshape(batch_size * num_frame,
                                 inputs.size(2),
                                 inputs.size(3),
                                 inputs.size(4))

        return outputs

    def forward(self, x):
        x = self.make_new_inputs(x)
        return x


class FrameAvg(nn.Module):
    def __init__(self, batch_size, num_frame):
        super().__init__()
        self.batch_size = batch_size
        self.num_frame = num_frame

    def frame_out_to_video_out(
            self, input: torch.Tensor, batch_size, num_frame) -> torch.Tensor:
        """
        フレームごとの出力をビデオとしての出力に変換
        Args:
            input (torch.Tensor): フレームごとの出力
            batch_size (int): バッチサイズ
            num_frame (int): フレーム数

        Returns:
            torch.Tensor: ビデオとしての出力
        """
        input = input.reshape(self.batch_size, self.num_frame, -1)
        output = torch.mean(input, dim=1)
        return output

    def forward(self, x):
        x = self.frame_out_to_video_out(x, self.batch_size, self.num_frame)
        return x


class ReconstructNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet152(pretrained=True)
        model_num_features = model.fc.in_features
        num_class = 400
        args = Args()

        self.video_to_frame = VideoToFrame()
        self.net_bottom = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

        # self.adapter_0 = Adapter(512)
        self.adapter_1 = Adapter(1024)

        self.net_top = nn.Sequential(
            nn.Linear(model_num_features, num_class)
        )

        self.frame_avg = FrameAvg(batch_size=args.BATCH_SIZE,
                                  num_frame=args.VIDEO_NUM_SUBSAMPLED)

        # 学習させるパラメータ名
        self.update_param_names = ["adapter_0.bn1.weight", "adapter_0.bn1.bias",
                                   "adapter_0.conv1.weight", "adapter_0.conv1.bias",
                                   "adapter_0.bn2.weight", "adapter_0.bn2.bias",
                                   "adapter_1.bn1.weight", "adapter_1.bn1.bias",
                                   "adapter_1.conv1.weight", "adapter_1.conv1.bias",
                                   "adapter_1.bn2.weight", "adapter_1.bn2.bias",
                                   "net_top.0.weight", "net_top.0.bias"]
        # 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
        for name, param in self.named_parameters():
            if name in self.update_param_names:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self, x):
        x = self.video_to_frame(x)
        x = self.net_bottom(x)
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.adapter_0(x)
        x = self.layer3(x)
        x = self.adapter_1(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net_top(x)
        x = self.frame_avg(x)
        return x


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


def train():
    args = Args()
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    train_dataset = get_kinetics("train")
    val_dataset = get_kinetics("val")
    train_loader = make_loader(train_dataset)
    val_loader = make_loader(val_dataset)

    model = ReconstructNet()
    model = model.to(device)
    # model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    hyper_params = {
        "Dataset": "Kinetics400",
        "epoch": args.NUM_EPOCH,
        "batch_size": args.BATCH_SIZE,
        "num_frame": args.VIDEO_NUM_SUBSAMPLED,
        "Adapter": "adp:1",
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

            pbar_epoch.set_postfix_str(
                ' train_loss={:6.04f} , val_loss={:6.04f}, train_acc={:6.04f}, val_acc={:6.04f}'
                ''.format(
                    train_loss.avg,
                    val_loss.avg,
                    train_acc.avg,
                    val_acc.avg)
            )

            # metrics = {"train_accuracy": train_acc.avg,
            #            "val_accuracy": val_acc.avg
            #            }
            # experiment.log_multiple_metrics(metrics, epoch + 1)
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-epoch", type=int, help="num_epoch")
    # parser.add_argument("-bs", type=int, help="batch_size")
    # parser.add_argument("-frame", type=int, help="num_frame from one video")
    # args = parser.parse_args()
    # print("epoch:{0}, batch_size:{1}. num_frame:{2}".format(
    #     args.epoch, args.bs, args.frame
    # ))
    train()


if __name__ == '__main__':
    main()
