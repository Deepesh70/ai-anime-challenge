"""
PyTorch Dataset for Anime Video Patch Extraction (Pillar 2).

Extracts randomly cropped high-resolution patches from training video clips in data/train/train/,
and synthesizes paired degraded low-resolution inputs via AnimeBroadcastDegradation on-the-fly.
"""

import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .degradation import AnimeBroadcastDegradation


class AnimePatchDataset(Dataset):
    """
    On-the-fly paired patch dataset for anime domain adaptation fine-tuning.
    Draws patches from local MP4 video clips, degrades with broadcast simulator.
    """

    def __init__(
        self,
        data_dir: str = "data/train/train",
        hr_patch_size: int = 256,
        scale: int = 2,
        samples_per_epoch: int = 1000,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.hr_patch_size = hr_patch_size
        self.scale = scale
        self.lr_patch_size = hr_patch_size // scale
        self.samples_per_epoch = samples_per_epoch

        self.video_paths = sorted(glob.glob(os.path.join(data_dir, "*.mp4")))
        if not self.video_paths:
            # Fallback to val directory if train not available
            self.video_paths = sorted(glob.glob("val/*.mp4"))

        if not self.video_paths:
            raise FileNotFoundError(f"No MP4 video clips found in {data_dir} or val/.")

        self.degrader = AnimeBroadcastDegradation(scale=scale)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _sample_random_frame(self) -> np.ndarray:
        """Picks a random video and extracts a random frame."""
        for _ in range(5):  # retry up to 5 times
            vid_path = random.choice(self.video_paths)
            cap = cv2.VideoCapture(vid_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                continue

            frame_idx = random.randint(0, max(0, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret and frame is not None and frame.shape[0] >= self.hr_patch_size and frame.shape[1] >= self.hr_patch_size:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Fallback dummy frame if video decoding fails
        return np.zeros((self.hr_patch_size, self.hr_patch_size, 3), dtype=np.uint8)

    def __getitem__(self, idx: int):
        frame = self._sample_random_frame()
        h, w = frame.shape[:2]

        # 1. Extract random HR crop
        top = random.randint(0, h - self.hr_patch_size)
        left = random.randint(0, w - self.hr_patch_size)
        hr_patch = frame[top : top + self.hr_patch_size, left : left + self.hr_patch_size]

        # Random horizontal flip
        if random.random() < 0.5:
            hr_patch = np.fliplr(hr_patch).copy()

        # 2. Synthesize degraded LR patch
        lr_patch = self.degrader.degrade(hr_patch)

        # Ensure exact dimensional match for LR
        if lr_patch.shape[:2] != (self.lr_patch_size, self.lr_patch_size):
            lr_patch = cv2.resize(lr_patch, (self.lr_patch_size, self.lr_patch_size), interpolation=cv2.INTER_AREA)

        # 3. Convert to PyTorch tensors normalized in [0.0, 1.0]
        # (H, W, C) uint8 -> (C, H, W) float32
        hr_tensor = torch.from_numpy(hr_patch.transpose(2, 0, 1)).float() / 255.0
        lr_tensor = torch.from_numpy(lr_patch.transpose(2, 0, 1)).float() / 255.0

        return {"lr": lr_tensor, "hr": hr_tensor}
