"""Anime Motion-Gated Temporal Consistency Filter (Pillar 3).

Exploits anime production properties:
- Locks static painted backgrounds across frames (eliminating sub-pixel line shimmer).
- Selectively passes moving character line art (eliminating motion blur and ghosting).
- Detects scene cuts to instantly reset temporal state.
"""
import cv2
import numpy as np


class AnimeTemporalFilter:
    """Streaming motion-gated temporal filter for anime video sequences."""

    def __init__(self, tau_static: float = 2.5, tau_motion: float = 9.0, scene_cut_thresh: float = 28.0):
        """
        Args:
            tau_static: Differences below this threshold are treated as static background noise.
            tau_motion: Differences above this threshold are treated as moving character lines.
            tau_static to tau_motion: Soft transition zone preventing hard boundary seams.
            scene_cut_thresh: Mean frame difference threshold triggering state reset.
        """
        self.tau_static = tau_static
        self.tau_motion = tau_motion
        self.scene_cut_thresh = scene_cut_thresh
        
        self.prev_input = None
        self.prev_restored = None

    def reset(self):
        """Reset temporal state (e.g., between different video clips)."""
        self.prev_input = None
        self.prev_restored = None

    def process(self, input_frame: np.ndarray, restored_frame: np.ndarray) -> np.ndarray:
        """
        Args:
            input_frame: Raw RGB uint8 frame (H, W, 3).
            restored_frame: Model-restored RGB uint8 frame (H, W, 3).
        Returns:
            Temporally stabilized RGB uint8 frame (H, W, 3).
        """
        if self.prev_input is None or self.prev_restored is None:
            self.prev_input = input_frame.copy()
            self.prev_restored = restored_frame.astype(np.float32)
            return restored_frame

        # 1. Compute pixel-wise absolute difference on input frames (float32)
        diff = np.abs(input_frame.astype(np.float32) - self.prev_input.astype(np.float32))
        max_diff = np.max(diff, axis=-1)  # (H, W) maximum channel difference

        # 2. Check for global scene cut
        mean_diff = np.mean(max_diff)
        if mean_diff > self.scene_cut_thresh:
            # Hard scene transition: reset history to prevent ghosting
            self.prev_input = input_frame.copy()
            self.prev_restored = restored_frame.astype(np.float32)
            return restored_frame

        # 3. Soft spatial smoothing on difference map to bridge thin line gaps
        diff_smooth = cv2.GaussianBlur(max_diff, (5, 5), sigmaX=1.2)

        # 4. Compute continuous motion gating alpha mask in [0.0, 1.0]
        # alpha = 0.0 -> static background (lock to temporal history)
        # alpha = 1.0 -> dynamic moving character (pass sharp current model frame)
        alpha = np.clip(
            (diff_smooth - self.tau_static) / max(self.tau_motion - self.tau_static, 1e-5),
            0.0,
            1.0
        )
        alpha = np.expand_dims(alpha, axis=-1)  # (H, W, 1) for broadcasting

        # 5. Motion-gated temporal blend
        curr_restored_f32 = restored_frame.astype(np.float32)
        smoothed_f32 = alpha * curr_restored_f32 + (1.0 - alpha) * self.prev_restored

        # 6. Update internal state
        self.prev_input = input_frame.copy()
        self.prev_restored = smoothed_f32

        # 7. Format output uint8
        return np.clip(np.round(smoothed_f32), 0, 255).astype(np.uint8)
