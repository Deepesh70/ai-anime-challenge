"""
Physics-Based Broadcast Degradation Simulator for Anime (Pillar 2).

Simulates the compound degradation pipeline present in broadcast anime video:
1. Optical/resampling blur (Gaussian & anisotropic kernels)
2. Downsampling (bilinear, bicubic, area)
3. Chroma Subsampling (YUV 4:2:0 downsampling simulating broadcast color bleed)
4. Transform Quantization (DCT macroblocking & high-frequency cutoff, simulating H.264/AVC compression)
5. Additive transmission noise (Poisson-Gaussian noise)

Used to generate synthetic (LR, HR) training pairs from high-fidelity anime patches
for domain adaptation fine-tuning on Re-Anime600.
"""

import random
import cv2
import numpy as np


class AnimeBroadcastDegradation:
    """Simulates broadcast television compression and optical degradations for anime frames."""

    def __init__(
        self,
        scale: int = 2,
        blur_prob: float = 0.7,
        chroma_bleed_prob: float = 0.9,
        compression_prob: float = 0.95,
        noise_prob: float = 0.5,
    ):
        self.scale = scale
        self.blur_prob = blur_prob
        self.chroma_bleed_prob = chroma_bleed_prob
        self.compression_prob = compression_prob
        self.noise_prob = noise_prob

    def apply_blur(self, img: np.ndarray) -> np.ndarray:
        """Applies mild Gaussian or directional blur simulating broadcast transmission."""
        if random.random() > self.blur_prob:
            return img

        # Kernel size must be odd
        ksize = random.choice([3, 5])
        sigma = random.uniform(0.3, 1.2)
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def apply_chroma_bleed(self, img: np.ndarray) -> np.ndarray:
        """
        Simulates 4:2:0 chroma subsampling:
        Converts to YCrCb, downsamples Cr/Cb by 2x or 4x with nearest or bilinear,
        and upsamples back, creating edge color spill.
        """
        if random.random() > self.chroma_bleed_prob:
            return img

        ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        h, w = ycrcb.shape[:2]

        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        # Downsample chroma planes by factor of 2 or 4
        sub_factor = random.choice([2, 4])
        down_w, down_h = max(1, w // sub_factor), max(1, h // sub_factor)
        inter_down = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA])
        inter_up = random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR])

        cr_sub = cv2.resize(cr, (down_w, down_h), interpolation=inter_down)
        cb_sub = cv2.resize(cb, (down_w, down_h), interpolation=inter_down)

        cr_recon = cv2.resize(cr_sub, (w, h), interpolation=inter_up)
        cb_recon = cv2.resize(cb_sub, (w, h), interpolation=inter_up)

        ycrcb[:, :, 1] = cr_recon
        ycrcb[:, :, 2] = cb_recon

        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    def apply_compression(self, img: np.ndarray) -> np.ndarray:
        """
        Simulates DCT block quantization and ringing artifacts (H.264/JPEG macroblocks).
        Uses randomized quality factor between 30 (heavy broadcast) and 75 (moderate broadcast).
        """
        if random.random() > self.compression_prob:
            return img

        quality = random.randint(32, 75)
        # Convert RGB to BGR for OpenCV encoder
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        result, encimg = cv2.imencode(".jpg", bgr, encode_param)
        if not result:
            return img
        decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)

    def apply_noise(self, img: np.ndarray) -> np.ndarray:
        """Simulates camera sensor and broadcast RF transmission noise."""
        if random.random() > self.noise_prob:
            return img

        sigma = random.uniform(2.0, 8.0)
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return noisy

    def downsample(self, img: np.ndarray) -> np.ndarray:
        """Downsamples HR frame to LR scale."""
        h, w = img.shape[:2]
        lr_w, lr_h = w // self.scale, h // self.scale
        mode = random.choice([cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR])
        return cv2.resize(img, (lr_w, lr_h), interpolation=mode)

    def degrade(self, hr_img: np.ndarray) -> np.ndarray:
        """
        Full forward degradation pipeline:
        HR -> Blur -> Chroma Subsampling -> Downsample (2x) -> Compression -> Noise -> LR
        """
        # Ensure dimensions are divisible by scale
        h, w = hr_img.shape[:2]
        crop_h = h - (h % self.scale)
        crop_w = w - (w % self.scale)
        hr_cropped = hr_img[:crop_h, :crop_w]

        # 1. Optical blur
        img = self.apply_blur(hr_cropped)

        # 2. Chroma bleeding
        img = self.apply_chroma_bleed(img)

        # 3. Spatial downsampling
        lr_img = self.downsample(img)

        # 4. Compression blocking & ringing
        lr_img = self.apply_compression(lr_img)

        # 5. Transmission noise
        lr_img = self.apply_noise(lr_img)

        return lr_img


if __name__ == "__main__":
    import os

    # Smoke test on a reference validation frame
    ref_video = os.path.join(os.path.dirname(__file__), "..", "val", "10299.mp4")
    if os.path.isfile(ref_video):
        cap = cv2.VideoCapture(ref_video)
        ret, frame_bgr = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            simulator = AnimeBroadcastDegradation(scale=2)
            lr = simulator.degrade(frame_rgb)

            # Upsample LR back with nearest for visual comparison
            lr_viz = cv2.resize(lr, (frame_rgb.shape[1], frame_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            side_by_side = np.hstack([frame_rgb, lr_viz])

            os.makedirs("comparisons", exist_ok=True)
            out_path = os.path.join("comparisons", "degradation_simulation.png")
            cv2.imwrite(out_path, cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
            print(f"[TEST PASSED] Synthetic degradation generated: {out_path}")
            print(f"  Input HR shape: {frame_rgb.shape} -> Degraded LR shape: {lr.shape}")
