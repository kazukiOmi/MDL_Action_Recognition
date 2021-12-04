from comet_ml import Experiment
import torch
import torch.nn as nn

from models import model as Model
from datasets import dataset as Data

from tqdm import tqdm
import os
import os.path as osp


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
    train_loader_list, val_loader_list = Data.loader_list(args)
    loader_itr_list = []
    for d in train_loader_list:
        loader_itr_list.append(iter(d))

    model = Model.MyNet(args, config)
    # model_path = "checkpoint/5000_checkpoint.pth"
    # model.load_state_dict(torch.load(model_path))
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
        "ex_name": args.ex_name,
    }

    experiment = Experiment(
        api_key=args.api_key,
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
                                ex_name=osp.join(args.adp_mode, args.ex_name))

    experiment.end()
