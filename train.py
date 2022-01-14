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


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    https://github.com/pytorch/examples/blob/cedca7729fef11c91e28099a0e45d7e98d03b66d/imagenet/main.py#L411
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    # model_path = "checkpoint/efficient_space_temporal/ex11/3000_checkpoint.pth"
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
        # "LN": "No",
        "adp num": args.adp_num,
        "adp_pos": args.adp_pos,
    }

    experiment = Experiment(
        api_key=args.api_key,
        project_name="feature-extract",
        workspace="kazukiomi",
    )

    experiment.add_tag('pytorch')
    experiment.log_parameters(hyper_params)

    step = 0

    num_iters = args.iteration

    train_acc_list = []
    train_top5_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_top5_acc_list = []
    val_loss_list = []
    for _ in dataset_name_list:
        train_acc_list.append(AverageMeter())
        train_top5_acc_list.append(AverageMeter())
        train_loss_list.append(AverageMeter())
        val_acc_list.append(AverageMeter())
        val_top5_acc_list.append(AverageMeter())
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
                acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
                train_acc_list[i].update(acc1, bs)
                train_top5_acc_list[i].update(acc5, bs)
            if itr % 8 == 7:
                optimizer.step()
            scheduler.step()

            for i, name in enumerate(dataset_name_list):
                experiment.log_metric(
                    "batch_accuracy_" + name, train_acc_list[i].val, step=step)
                experiment.log_metric(
                    "batch_top5_accuracy_" + name, train_top5_acc_list[i].val, step=step)
                experiment.log_metric(
                    "batch_loss_" + name, train_loss_list[i].val, step=step)
            step += 1

            if (step) % 1000 == 0:
                # if (itr + 1) % 3500 == 0:
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
                            acc1, acc5 = accuracy(val_outputs, labels, (1, 5))
                            val_acc_list[i].update(acc1, bs)
                            val_top5_acc_list[i].update(acc5, bs)

                    for i, name in enumerate(dataset_name_list):
                        experiment.log_metric(
                            "train_accuracy_" + name, train_acc_list[i].avg, step=step)
                        experiment.log_metric(
                            "train_top5_accuracy_" + name, train_top5_acc_list[i].avg, step=step)
                        experiment.log_metric(
                            "train_loss_" + name, train_loss_list[i].avg, step=step)
                        experiment.log_metric(
                            "val_accuracy_" + name, val_acc_list[i].avg, step=step)
                        experiment.log_metric(
                            "val_top5_accuracy_" + name, val_top5_acc_list[i].avg, step=step)
                        experiment.log_metric(
                            "val_loss_" + name, val_loss_list[i].avg, step=step)
                        train_acc_list[i].reset()
                        train_top5_acc_list[i].reset()
                        train_loss_list[i].reset()
                        val_acc_list[i].reset()
                        val_top5_acc_list[i].reset()
                        val_loss_list[i].reset()
                """Finish Val mode"""

                """save model"""
                save_checkpoint(model,
                                filename=str(itr + 1) + "_checkpoint.pth",
                                dir_name="checkpoint",
                                ex_name=osp.join(args.adp_mode, args.ex_name))

    experiment.end()


def val(args, config):
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    dataset_name_list = args.dataset_names
    _, val_loader_list = Data.loader_list(args)

    model = Model.MyNet(args, config)
    model_path = "checkpoint/No/ex0/14000_checkpoint.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    torch.backends.cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    lr = args.learning_rate
    weight_decay = args.weight_decay

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
        project_name="feature-extract",
        workspace="kazukiomi",
    )

    experiment.add_tag('pytorch')
    experiment.log_parameters(hyper_params)
    step = 0
    val_acc_list = []
    val_top5_acc_list = []
    val_loss_list = []
    for _ in dataset_name_list:
        val_acc_list.append(AverageMeter())
        val_top5_acc_list.append(AverageMeter())
        val_loss_list.append(AverageMeter())

    model.eval()

    with torch.no_grad():
        for i, loader in enumerate(val_loader_list):
            with tqdm(loader) as pbar:
                for val_batch in pbar:
                    inputs = val_batch['video'].to(device)
                    labels = val_batch['label'].to(device)

                    bs = inputs.size(0)

                    val_outputs = model(
                        inputs, dataset_name_list[i])
                    loss = criterion(val_outputs, labels)

                    val_loss_list[i].update(loss, bs)
                    acc1, acc5 = accuracy(val_outputs, labels, (1, 5))
                    val_acc_list[i].update(acc1, bs)
                    val_top5_acc_list[i].update(acc5, bs)

                    pbar.set_postfix(
                        acc=acc1.item(), acc_avg=val_acc_list[i].avg)

        for i, name in enumerate(dataset_name_list):
            experiment.log_metric(
                "val_accuracy_" + name, val_acc_list[i].avg, step=step)
            experiment.log_metric(
                "val_top5_accuracy_" + name, val_top5_acc_list[i].avg, step=step)
            experiment.log_metric(
                "val_loss_" + name, val_loss_list[i].avg, step=step)
            val_acc_list[i].reset()
            val_top5_acc_list[i].reset()
            val_loss_list[i].reset()

    experiment.end()


def multiview_val(args, config):
    device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")

    # dataset_name_list = args.dataset_names
    # _, val_loader_list = Data.loader_list(args)
    dataset = Data.get_multiview_kinetics("val", args)
    loader = Data.make_loader(dataset, args, 1)

    model = Model.MyNet(args, config)
    model_path = "checkpoint/No/ex0/14000_checkpoint.pth"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    torch.backends.cudnn.benchmark = True

    # criterion = nn.CrossEntropyLoss()
    # lr = args.learning_rate
    # weight_decay = args.weight_decay

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
        # "LN": "No",
        "adp num": args.adp_num,
        "adp_pos": args.adp_pos,
        "multiview": True,
    }
    # experiment = Experiment(
    #     api_key=args.api_key,
    #     project_name="feature-extract",
    #     workspace="kazukiomi",
    # )

    # experiment.add_tag('pytorch')
    # experiment.log_parameters(hyper_params)
    # step = 0
    # val_loss = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        with tqdm(enumerate(loader), total=len(loader)) as pbar:
            for i, batch in pbar:
                b, v, c, t, h, w = batch['video'].shape
                batch['video'] = batch['video'].view(-1, c, t, h, w)
                batch['video'] = batch['video'].to(device)
                batch['label'] = batch['label'].to(device)
                left = batch['video'][:, :, :,
                                      (h - 224) // 2:(h + 224) // 2, 0:224]
                right = batch['video'][:, :, :,
                                       (h - 224) // 2:(h + 224) // 2, -224:]
                center = batch['video'][:, :, :,
                                        (h - 224) // 2:(h + 224) // 2, (w - 224) // 2:(w + 224) // 2]
                batch["video"] = torch.cat((left, right, center), 0)

                outputs = model(batch["video"], "Kinetics")
                outputs = torch.mean(outputs, 0, keepdim=True)
                acc1, acc5 = accuracy(outputs, batch["label"], topk=(1, 5))
                acc_top1.update(acc1, 1)
                acc_top5.update(acc5, 1)

                pbar.set_postfix(
                    acc1_avg=acc_top1.avg, acc5_avg=acc_top5.avg)

                # if i > 5:
                #     break

    # experiment.end()
