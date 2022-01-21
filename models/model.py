from importlib.metadata import requires
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorchvideo
from pytorchvideo.models import x3d
import torchinfo


class Adapter(nn.Module):
    def __init__(self, feature_list, frame):
        super().__init__()
        self.channel = feature_list[0]
        self.height = feature_list[1]
        # self.norm1 = nn.LayerNorm([channel, frame, height, height])
        self.norm1 = nn.BatchNorm3d(self.channel)
        self.act = nn.ReLU()  # TODO(omi): implement swish

    def swap(self, input: torch.Tensor) -> torch.Tensor:
        return input

    def unswap(self, input, bs, frames=16) -> torch.Tensor:
        return input

    def forward(self, x):
        batch_size, channel, frames, height, width = x.size()

        out = self.swap(x)
        out = self.conv1(out)
        out = self.unswap(out, batch_size, frames)
        out = self.norm1(out)

        out += x
        out = self.act(out)

        return out


class TestAdapter(nn.Module):
    def __init__(self):
        super().__init__()

    def ident(self, x):
        return x

    def forward(self, x):
        x = self.ident(x)
        return x


class FrameWise2dAdapter(Adapter):
    def __init__(self, feature_list, frame):
        super().__init__(feature_list, frame)
        self.conv1 = nn.Conv2d(self.channel, self.channel, 1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)

    def swap(self, input: torch.Tensor) -> torch.Tensor:
        # input: B,C,T,H,W
        batch_size, channel, frames, height, width = input.size()
        # B,C,T,H,W --> B,T,C,H,W
        output = input.permute(0, 2, 1, 3, 4)
        # B,T,C,H,W --> BT,C,H,W
        output = output.reshape(batch_size * frames, channel, height, width)
        return output

    def unswap(self, input, bs, frames=16) -> torch.Tensor:
        # input: BT,C,H,W
        batchs_frames, channel, height, width = input.size()
        frames = int(batchs_frames / bs)
        # BT,C,H,W --> B,T,C,H,W
        output = input.reshape(bs, frames, channel, height, width)
        # B,T,C,H,W --> B,C,T,H,W
        output = output.permute(0, 2, 1, 3, 4)
        return output


class ChannelWise3dAdapter(Adapter):
    def __init__(self, feature_list, frame):
        super().__init__(feature_list, frame)
        self.conv1 = nn.Conv2d(frame, frame, 1)
        torch.nn.init.kaiming_normal_(self.conv1.weight)

    def swap(self, input: torch.Tensor) -> torch.Tensor:
        # input: B,C,T,H,W
        batch_size, channel, frames, height, width = input.size()
        # B,C,T,H,W --> BC,T,H,W
        output = input.reshape(batch_size * channel, frames, width, height)
        return output

    def unswap(self, input, bs, frames=16) -> torch.Tensor:
        # input: BC,T,H,W
        batchs_channels, frames, height, width = input.size()
        channel = int(batchs_channels / bs)
        # BC,T,H,W --> B,C,T,H,W
        output = input.reshape(bs, channel, frames, height, width)
        return output


class VideoAdapter(Adapter):
    def __init__(self, feature_list, frame):
        super().__init__(feature_list, frame)
        self.conv1 = nn.Conv3d(self.channel, self.channel,
                               3, padding=(1, 1, 1))
        torch.nn.init.kaiming_normal_(self.conv1.weight)


class R2p1dConv(nn.Module):
    def __init__(self, feature_list):
        super().__init__()
        channel = feature_list[0]
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv2 = nn.Conv3d(channel, channel, (3, 1, 1), padding=(1, 0, 0))
        torch.nn.init.kaiming_normal_(self.conv1.weight)
        torch.nn.init.kaiming_normal_(self.conv2.weight)

    def swap(self, input: torch.Tensor) -> torch.Tensor:
        # input: B,C,T,H,W
        batch_size, channel, frames, height, width = input.size()
        # B,C,T,H,W --> B,T,C,H,W
        output = input.permute(0, 2, 1, 3, 4)
        # B,T,C,H,W --> BT,C,H,W
        output = output.reshape(batch_size * frames, channel, height, width)
        return output

    def unswap(self, input, bs, frames=16) -> torch.Tensor:
        # input: BT,C,H,W
        batchs_frames, channel, height, width = input.size()
        frames = int(batchs_frames / bs)
        # BT,C,H,W --> B,T,C,H,W
        output = input.reshape(bs, frames, channel, height, width)
        # B,T,C,H,W --> B,C,T,H,W
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def forward(self, x):
        batch_size, channel, frames, height, width = x.size()

        out = self.swap(x)
        out = self.conv1(out)
        out = self.unswap(out, batch_size, frames)
        out = self.conv2(out)

        return out


class EfficientVideoAdapter(Adapter):
    def __init__(self, feature_list, frame):
        super().__init__(feature_list, frame)
        self.conv1 = R2p1dConv(feature_list)


class EfficientSpaceTemporalAdapter(nn.Module):
    def __init__(self, feature_list, frame):
        super().__init__()
        self.video2frame_adapter = FrameWise2dAdapter(feature_list, frame)
        self.temporal_adapter = ChannelWise3dAdapter(feature_list, frame)

    def forward(self, x):
        out = self.video2frame_adapter(x)
        out = self.temporal_adapter(x)
        return out


class MyHeadDict(nn.Module):
    def __init__(self, in_channel, dataset_names, class_dict):
        super().__init__()
        self.head = nn.ModuleDict({})

        head_dict = {}
        for name in dataset_names:
            head = nn.Linear(in_channel, class_dict[name])
            head_dict[name] = head
        self.head.update(head_dict)

    def forward(self, x, domain):
        x = self.head[domain](x)
        return x


def select_adapter(adp_mode, feature_list, frame):
    if adp_mode == "video2frame":
        adp = FrameWise2dAdapter(feature_list, frame)
    elif adp_mode == "temporal":
        adp = ChannelWise3dAdapter(feature_list, frame)
    elif adp_mode == "space_temporal":
        adp = VideoAdapter(feature_list, frame)
    elif adp_mode == "efficient_space_temporal":
        # adp = EfficientSpaceTemporalAdapter(feature_list, frame)
        adp = EfficientVideoAdapter(feature_list, frame)
    else:
        raise NameError("invalide adapter name")
    return adp


class MyAdapterDict(nn.Module):
    def __init__(self, args, feature_list):
        super().__init__()
        channel = feature_list[0]
        height = feature_list[1]
        self.adapter = nn.ModuleDict({})

        adp_dict = {}
        for name in args.dataset_names:
            adp = select_adapter(
                args.adp_mode, feature_list, args.num_frames)
            adp_dict[name] = adp
        self.adapter.update(adp_dict)

        self.norm = nn.LayerNorm([channel, args.num_frames, height, height])

    def forward(self, x, domain):
        x = self.adapter[domain](x)
        x = self.norm(x)
        return x


def make_mod_list(model, args):
    mod_list = []
    feature_list = args.feature_list
    index = 0
    module = model.children().__next__()
    if args.adp_place == "stages":
        for child in module.children():
            if isinstance(
                    child, pytorchvideo.models.head.ResNetBasicHead) == False:
                mod_list.append(child)
                if args.adp_pos == "all":
                    mod_list.append(MyAdapterDict(args, feature_list[index]))
                    index += 1
                elif args.adp_pos == "bottom":
                    if index < args.adp_num:
                        mod_list.append(
                            MyAdapterDict(args, feature_list[index]))
                    index += 1
                elif args.adp_pos == "top":
                    if index >= 5 - args.adp_num:
                        mod_list.append(
                            MyAdapterDict(args, feature_list[index]))
                    index += 1
    elif args.adp_place == "all":
        for child in module.children():
            if isinstance(child, pytorchvideo.models.stem.ResNetBasicStem):
                mod_list.append(child)
                mod_list.append(
                    MyAdapterDict(args, feature_list[index]))
                index += 1
            elif isinstance(child, pytorchvideo.models.head.ResNetBasicHead):
                pass
            elif isinstance(child, pytorchvideo.models.resnet.ResStage):
                g = child.children()
                for g_c in g:
                    for i, g_g in enumerate(g_c):
                        mod_list.append(g_g)
                        mod_list.append(MyAdapterDict(
                            args, feature_list[index]))
                index += 1
            else:
                raise NameError("Failed to create module list.")
    elif args.adp_place == "blocks":
        for child in module.children():
            if isinstance(child, pytorchvideo.models.stem.ResNetBasicStem):
                mod_list.append(child)
            elif isinstance(child, pytorchvideo.models.head.ResNetBasicHead):
                pass
            elif isinstance(child, pytorchvideo.models.resnet.ResStage):
                g = child.children()
                for g_c in g:
                    for i, g_g in enumerate(g_c):
                        mod_list.append(g_g)
                        if i != len(g_c) - 1:
                            mod_list.append(MyAdapterDict(
                                args, feature_list[index + 1]))
                index += 1
            else:
                raise NameError("Failed to create module list.")
    elif args.adp_place == "No":
        for child in module.children():
            if isinstance(
                    child, pytorchvideo.models.head.ResNetBasicHead) == False:
                mod_list.append(child)
    else:
        raise NameError("invalide adapter place")
    return mod_list


def make_class_dict(args, config):
    """
    keyがクラス名（str）でvalueがクラス数（int）のdictを作成
    """
    config.read("config.ini")
    num_class_dict = {}
    for name in args.dataset_names:
        num_class_dict[name] = int(config[name]["num_class"])
    return num_class_dict


class MyNet(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        model = torch.hub.load(
            'facebookresearch/pytorchvideo', "x3d_m", pretrained=args.pretrained)
        self.dim_features = model.blocks[5].proj.in_features
        self.num_frames = args.num_frames
        self.class_dict = make_class_dict(args, config)

        if args.fix_shared_params == True:
            for name, param in model.named_parameters():
                param.requires_grad = False

        mod_list = make_mod_list(model, args)

        self.module_list = nn.ModuleList(mod_list)
        self.head_bottom = nn.Sequential(
            model.blocks[5].pool,
            model.blocks[5].dropout
        )

        self.head_top_dict = MyHeadDict(self.dim_features,
                                        args.dataset_names,
                                        self.class_dict)

    def forward(self, x: torch.Tensor, domain) -> torch.Tensor:
        for f in self.module_list:
            if isinstance(f, MyAdapterDict):
                x = f(x, domain)
            else:
                x = f(x)
            # torchinfoで確認できないので確認用
            # print(type(f))
            # print(x.shape)

        x = self.head_bottom(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.head_top_dict(x, domain)
        x = x.view(-1, self.class_dict[domain])
        return x

    def fix_shared_params(self, args):
        for name, param in self.named_parameters():
            param.requires_grad = False
        for name, param in self.named_parameters():
            for dataset_name in args.dataset_names:
                if dataset_name in name and param.requires_grad == False:
                    param.requires_grad = True


class TorchInfoMyNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        model = torch.hub.load(
            'facebookresearch/pytorchvideo', "x3d_m", pretrained=args.pretrained)
        self.dim_features = model.blocks[5].proj.in_features
        self.num_frames = args.num_frames
        self.num_class = 400

        self.feature_extract = nn.Sequential(
            model.blocks[0],
            # EfficientVideoAdapter([24, 112], 16),
            model.blocks[1],
            # EfficientVideoAdapter([24, 56], 16),
            model.blocks[2],
            # EfficientVideoAdapter([48, 28], 16),
            model.blocks[3],
            # EfficientVideoAdapter([96, 14], 16),
            model.blocks[4],
            # EfficientVideoAdapter([192, 7], 16),
        )
        self.head_bottom = nn.Sequential(
            model.blocks[5].pool,
            model.blocks[5].dropout
        )
        self.head_top = model.blocks[5].proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extract(x)
        x = self.head_bottom(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.head_top(x)
        x = x.view(-1, self.num_class)
        return x


def torch_info(model):
    bs = 1
    input_size = (bs, 3, 16, 224, 224)
    torchinfo.summary(
        model=model,
        input_size=input_size,
        depth=8,
        col_names=["input_size", "output_size"],
        # row_setting=("var_names")
    )
