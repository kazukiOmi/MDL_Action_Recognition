from comet_ml import config
import argparse
import configparser
import train


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", type=str, default="train",
                        choices=["train", "val", "multiview_val"])
    parser.add_argument("-i", "--iteration", type=int, default=14000,)
    parser.add_argument("-bsl", "--batch_size_list", nargs="*",
                        default=[32, 32, 32])
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=32,)
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
    parser.add_argument("--sche_list", nargs="*",
                        default=[8000, 12000])
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--scratch", action="store_false")
    parser.add_argument("-fix", "--fix_shared_params",
                        action="store_true",)
    parser.add_argument("--is_fix_in_train",
                        action="store_true",)
    parser.add_argument("-itr_fix", "--iteration_fix",
                        type=int, default=10000,)
    parser.add_argument("-ap", "--adp_place", type=str, default="stages",
                        choices=["stages", "blocks", "all", "No"])
    parser.add_argument("--adp_pos", type=str, default="all",
                        choices=["all", "top", "bottom"])
    parser.add_argument("--adp_num", type=int, default=5)
    parser.add_argument("-am", "--adp_mode", type=str, default="efficient_space_temporal",
                        choices=["video2frame", "temporal", "space_temporal", "efficient_space_temporal", "No"])
    parser.add_argument("-dn", "--dataset_names", nargs="*",
                        default=["UCF101", "Kinetics", "HMDB51"])
    parser.add_argument("--feature_list", nargs="*",
                        default=[[24, 112], [24, 56], [48, 28], [96, 14], [192, 7]])
    parser.add_argument("--cuda", type=str, default="cuda:2")
    parser.add_argument("--ex_name", type=str, default="test")
    parser.add_argument("--api_key", type=str,
                        default="TawRAwNJiQjPaSMvBAwk4L4pF")
    return parser.parse_args()


def main():
    args = get_arguments()
    config = configparser.ConfigParser()
    config.read("config.ini")

    if args.mode == "train":
        train.train(args, config)
    elif args.mode == "val":
        train.val(args, config)
    elif args.mode == "multiview_val":
        train.multiview_val(args, config)


if __name__ == '__main__':
    main()
