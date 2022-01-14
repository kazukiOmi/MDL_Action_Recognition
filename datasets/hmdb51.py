import os
import os.path as osp
import json
import pathlib
from typing import Any, Callable, List, Optional, Tuple, Type, Union

import torch
import torch.utils.data
from iopath.common.file_io import g_pathmgr

from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from torch.utils.data import RandomSampler

from . import multiview as MultiView


class Hmdb51LabeledVideoPaths:

    _allowed_splits = [1, 2, 3]
    _split_type_dict = {"train": 1, "test": 2, "unused": 0}

    @classmethod
    def from_dir(
        cls, label_name_file, data_path: str, split_id: int = 1, split_type: str = "train"
    ):
        data_path = pathlib.Path(data_path)
        if not data_path.is_dir():
            return RuntimeError(
                f"{data_path} not found or is not a directory.")
        if not int(split_id) in cls._allowed_splits:
            return RuntimeError(
                f"{split_id} not found in allowed split id's {cls._allowed_splits}."
            )
        file_name_format = "_test_split" + str(int(split_id))
        file_paths = sorted(
            (
                f
                for f in data_path.iterdir()
                if f.is_file() and f.suffix == ".txt" and file_name_format in f.stem
            )
        )
        return cls.from_csvs(label_name_file, file_paths, split_type)

    @classmethod
    def from_csvs(
        cls, label_name_file, file_paths: List[Union[pathlib.Path, str]], split_type: str = "train"
    ):
        label_name_file = open(label_name_file, "r")
        label_name_file = json.load(label_name_file)
        video_paths_and_label = []
        for file_path in file_paths:
            file_path = pathlib.Path(file_path)
            assert g_pathmgr.exists(file_path), f"{file_path} not found."
            if not (file_path.suffix ==
                    ".txt" and "_test_split" in file_path.stem):
                return RuntimeError(f"Ivalid file: {file_path}")

            action_name = "_"
            action_name = action_name.join((file_path.stem).split("_")[:-2])
            action_id = label_name_file[action_name]
            with g_pathmgr.open(file_path, "r") as f:
                for path_label in f.read().splitlines():
                    line_split = path_label.rsplit(None, 1)

                    if not int(
                            line_split[1]) == cls._split_type_dict[split_type]:
                        continue

                    file_path = os.path.join(action_name, line_split[0])
                    meta_tags = line_split[0].split("_")[-6:-1]
                    video_paths_and_label.append(
                        (file_path, {"label": action_id,
                                     "meta_tags": meta_tags})
                    )
                    # print(file_path + ":" + action_name)

        assert (
            len(video_paths_and_label) > 0
        ), f"Failed to load dataset from {file_path}."
        return cls(video_paths_and_label)

    def __init__(
        self, paths_and_labels: List[Tuple[str, Optional[dict]]], path_prefix=""
    ) -> None:
        self._paths_and_labels = paths_and_labels
        self._path_prefix = path_prefix

    def path_prefix(self, prefix):
        self._path_prefix = prefix

    path_prefix = property(None, path_prefix)

    def __getitem__(self, index: int) -> Tuple[str, dict]:
        path, label = self._paths_and_labels[index]
        return (os.path.join(self._path_prefix, path), label)

    def __len__(self) -> int:
        return len(self._paths_and_labels)


def Hmdb51(
    label_name_file: str,
    data_path: pathlib.Path,
    clip_sampler: ClipSampler,
    video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
    transform: Optional[Callable[[dict], Any]] = None,
    video_path_prefix: str = "",
    split_id: int = 1,
    split_type: str = "train",
    decode_audio=True,
    decoder: str = "pyav",
) -> LabeledVideoDataset:
    torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Hmdb51")

    labeled_video_paths = Hmdb51LabeledVideoPaths.from_dir(
        label_name_file, data_path, split_id=split_id, split_type=split_type
    )
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = LabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )

    return dataset


def multiview_Hmdb51(
    label_name_file: str,
    data_path: pathlib.Path,
    transform: Optional[Callable[[dict], Any]] = None,
    video_path_prefix: str = "",
    split_id: int = 1,
    split_type: str = "test",
):
    torch._C._log_api_usage_once("PYTORCHVIDEO.dataset.Hmdb51")

    labeled_video_paths = Hmdb51LabeledVideoPaths.from_dir(
        label_name_file, data_path, split_id=split_id, split_type=split_type
    )
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = MultiView.KineticsMultiViewTest(
        labeled_video_paths,
        clip_sampler=ConstantClipsPerVideoSampler(
            clip_duration=80 / 30, clips_per_video=10),
        video_sampler=RandomSampler,
        transform=transform,
        decode_audio=False,
        decoder="pyav",
    )

    return dataset
