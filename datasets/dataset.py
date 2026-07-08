"""
dataset.py

Ultrasound Video Dataset Loader
"""

import torch
from torch.utils.data import Dataset

from utils.video_utils import read_video, sample_frames


class UltrasoundDataset(Dataset):
    def __init__(
        self,
        video_paths,
        labels,
        num_frames=16,
        transform=None,
    ):

        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def load_video(self, video_path):

        frames = read_video(video_path)

        frames = sample_frames(
            frames,
            num_frames=self.num_frames,
        )

        return frames

    def __getitem__(self, idx):

        video_path = self.video_paths[idx]
        label = self.labels[idx]

        frames = self.load_video(video_path)

        if self.transform:

            frames = self.transform(frames)

        else:

            frames = torch.tensor(
                frames,
                dtype=torch.float32,
            )

            # (T, H, W, C) -> (T, C, H, W)
            frames = frames.permute(0, 3, 1, 2)

            # Normalize to [0,1]
            frames = frames / 255.0

        label = torch.tensor(
            label,
            dtype=torch.long,
        )

        return frames, label