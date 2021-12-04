from comet_ml import config

import torch

import models
from models import model as Model
import datasets
from datasets import dataset as Data
import train

import os.path as osp
import argparse
import configparser


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=14000,)
    parser.add_argument("--epoch", type=int, default=10,)
    parser.add_argument("--batch_size", type=int, default=32,)
    parser.add_argument("--batch_size_list", nargs="*", default=[32, 32])
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=32,)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--sche_list", nargs="*",
                        default=[8000, 12000])
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
    parser.add_argument("--ex_name", type=str)
    parser.add_argument("--api_key", type=str,
                        default="TawRAwNJiQjPaSMvBAwk4L4pF")
    return parser.parse_args()


def make_lr_list(args, config):
    model = Model.MyNet(args, config)
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
    train.train(args, config)

    """model_check (実際に入力を流す，dict使うとtorchinfoできないから)"""
    # model = Model.MyNet(args, config)
    # input = torch.randn(1, 3, 16, 224, 224)
    # # input = torch.randn(1, 2048)
    # device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # input = input.to(device)
    # out = model(input, args.dataset_names[1])
    # print(out.shape)


if __name__ == '__main__':
    # print(torch.cuda.get_arch_list())
    main()