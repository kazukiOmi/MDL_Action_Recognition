import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DistributedSampler, RandomSampler, SequentialSampler

from torchvision import transforms

from pytorchvideo.data import Ucf101, RandomClipSampler, UniformClipSampler

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


from pytorchvideo.data.utils import MultiProcessSampler
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset
from pytorchvideo.data.clip_sampling import ClipSampler


logger = logging.getLogger(__name__)


class KineticsMultiViewTest(LabeledVideoDataset):
    def __init__(self, labeled_video_paths: List[Tuple[str, Optional[dict]]], clip_sampler: ClipSampler, video_sampler: Type[torch.utils.data.Sampler]
                 = ..., transform: Optional[Callable[[dict], Any]] = None, decode_audio: bool = True, decoder: str = "pyav") -> None:
        super().__init__(labeled_video_paths, clip_sampler, video_sampler=video_sampler,
                         transform=transform, decode_audio=decode_audio, decoder=decoder)

    def __next__(self) -> list:
        """
        Retrieves the next clip based on the clip sampling strategy and video sampler.
        Returns:
            A dictionary with the following format.
            .. code-block:: text
                {
                    'video': <video_tensor>,
                    'label': <index_label>,
                    'video_label': <index_label>
                    'video_index': <video_index>,
                    'clip_index': <clip_index>,
                    'aug_index': <aug_index>,
                }
        """
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers
            # are spawned.
            self._video_sampler_iter = iter(
                MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video, info_dict, video_index = self._loaded_video_label
            else:
                video_index = next(self._video_sampler_iter)
                try:
                    video_path, info_dict = self._labeled_videos[video_index]
                    video = self.video_path_handler.video_from_path(
                        video_path,
                        decode_audio=self._decode_audio,
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video, info_dict, video_index)
                except Exception as e:
                    logger.debug(
                        "Failed to load video with error: {}; trial {}".format(
                            e,
                            i_try,
                        )
                    )
                    continue
            frame_list = []
            for i in range(10):
                (
                    clip_start,
                    clip_end,
                    clip_index,
                    aug_index,
                    is_last_clip,
                ) = self._clip_sampler(
                    self._next_clip_start_time, video.duration, info_dict
                )

                # Only load the clip once and reuse previously stored clip if there are multiple
                # views for augmentations to perform on the same clip.
                if aug_index == 0:
                    self._loaded_clip = video.get_clip(clip_start, clip_end)

                self._next_clip_start_time = clip_end

                video_is_null = (
                    self._loaded_clip is None or self._loaded_clip["video"] is None
                )
                # print(is_last_clip)
                if is_last_clip or video_is_null:
                    # Close the loaded encoded video and reset the last sampled clip time ready
                    # to sample a new video on the next iteration.
                    self._loaded_video_label[0].close()
                    self._loaded_video_label = None
                    self._next_clip_start_time = 0.0

                    if video_is_null:
                        logger.debug(
                            "Failed to load clip {}; trial {}".format(
                                video.name, i_try)
                        )
                        continue
                frames = self._loaded_clip["video"]
                # frame_list.append(frames)
                audio_samples = self._loaded_clip["audio"]
                # frame = torch.stack(frame_list, dim=0)
                # print(frame.shape)
                # frame =
                # frame.view(-1,frame.shape(2),frame.shape(3),frame.shape(4))
                sample_dict = {
                    "video": frames,
                    "video_name": video.name,
                    "video_index": video_index,
                    "clip_index": clip_index,
                    "aug_index": aug_index,
                    **info_dict,
                    **({"audio": audio_samples} if audio_samples is not None else {}),
                }
                if self._transform is not None:
                    sample_dict = self._transform(sample_dict)
                    frame_list.append(sample_dict['video'])

                    # User can force dataset to continue by returning None in
                    # transform.
                    if sample_dict is None:
                        continue
                # print(sample_dict['video'].shape)
            sample_dict['video'] = torch.stack(frame_list, dim=0)
            # print(sample_dict['video'].shape)
            return sample_dict
        else:
            raise RuntimeError(
                f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
            )
