"""Luma-Guided Chroma Bleed Filter (Pillar 1).

Uses the full-resolution luminance (Y) channel as a structural guidance image
to refine and de-blur subsampled chroma channels (Cb, Cr) using He et al. Guided Filtering.
Prevents vibrant cel colors from bleeding across sharp dark ink contours.
"""
import cv2
import numpy as np


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 3, eps: float = 1e-3) -> np.ndarray:
    """
    Fast O(N) Guided Filter (He et al., ECCV 2010 / TPAMI 2013).
    Transfers edge structure of 'guide' to 'src'.

    Args:
        guide: Normalized float32 guidance image (H, W) in [0.0, 1.0].
        src: Float32 source image to filter (H, W).
        radius: Filter window radius (ksize = 2*radius + 1).
        eps: Regularization parameter penalizing large gradients.
    Returns:
        Filtered float32 image (H, W).
    """
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I

    # Linear coefficients a and b
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    q = mean_a * guide + mean_b
    return q


class LumaGuidedChromaFilter:
    """Edge-constrained chroma filter for broadcast anime video frames."""

    def __init__(self, radius: int = 3, eps: float = 2e-3, blend: float = 0.85):
        """
        Args:
            radius: Kernel radius for guided filtering.
            eps: Edge-preservation penalty (smaller = tighter edge preservation).
            blend: Weight for filtered chroma (1.0 = fully filtered, 0.0 = original).
        """
        self.radius = radius
        self.eps = eps
        self.blend = blend

    def process(self, rgb_frame: np.ndarray) -> np.ndarray:
        """
        Args:
            rgb_frame: RGB uint8 frame (H, W, 3).
        Returns:
            Chroma-refined RGB uint8 frame (H, W, 3).
        """
        # 1. Convert RGB to YCrCb
        ycrcb = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float32) / 255.0  # Guide image: normalized luma
        cr = ycrcb[:, :, 1].astype(np.float32)
        cb = ycrcb[:, :, 2].astype(np.float32)

        # 2. Filter chroma channels using luma as guidance
        cr_filtered = guided_filter(y, cr, radius=self.radius, eps=self.eps)
        cb_filtered = guided_filter(y, cb, radius=self.radius, eps=self.eps)

        # 3. Controlled blend with original chroma to preserve artistic color shifts
        cr_final = self.blend * cr_filtered + (1.0 - self.blend) * cr
        cb_final = self.blend * cb_filtered + (1.0 - self.blend) * cb

        # 4. Reconstruct YCrCb and convert back to RGB
        ycrcb[:, :, 0] = np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8)
        ycrcb[:, :, 1] = np.clip(np.round(cr_final), 0, 255).astype(np.uint8)
        ycrcb[:, :, 2] = np.clip(np.round(cb_final), 0, 255).astype(np.uint8)

        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
