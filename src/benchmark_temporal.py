"""Benchmark temporal stability and measure inter-frame line shimmer."""
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import cv2
import numpy as np
import torch
from apisr_arch import RRDBNet
from src.temporal_filter import AnimeTemporalFilter

def benchmark(num_frames=120):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Temporal Consistency Benchmark on: {device} ({num_frames} frames)")

    # 1. Load APISR model
    model = RRDBNet(3, 3, scale=2)
    ckpt = torch.load("model_zoo/2x_APISR_RRDB_GAN_generator.pth", map_location='cpu', weights_only=False)['model_state_dict']
    model.load_state_dict(ckpt, strict=True)
    model.eval().to(device)

    # 2. Read consecutive frames from val/10299.mp4
    cap = cv2.VideoCapture("val/10299.mp4")
    frames = []
    for _ in range(num_frames):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Loaded {len(frames)} frames for temporal evaluation.")

    # 3. Restore frames with APISR (Raw single-frame)
    raw_restored = []
    with torch.no_grad():
        for f in frames:
            h, w, _ = f.shape
            x = torch.from_numpy(f).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            with torch.amp.autocast('cuda'):
                out = model(x)
            out_np = out.squeeze(0).clamp(0, 1).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
            raw_restored.append(cv2.resize(out_np, (w, h), interpolation=cv2.INTER_AREA))

    # 4. Apply Temporal Filter
    temp_filter = AnimeTemporalFilter()
    filtered_restored = []
    for orig_f, rest_f in zip(frames, raw_restored):
        filtered_restored.append(temp_filter.process(orig_f, rest_f))

    # 5. Quantify Temporal Inconsistency (Static Background Shimmer Variance)
    # Identify static pixels (|orig_t - orig_{t-1}| <= 2) and measure variance in restored
    raw_shimmer_errors = []
    filtered_shimmer_errors = []
    raw_sharpness = []
    filtered_sharpness = []

    for t in range(1, len(frames)):
        orig_diff = np.max(np.abs(frames[t].astype(np.float32) - frames[t-1].astype(np.float32)), axis=-1)
        static_mask = orig_diff <= 2.0  # Pixels that should be completely motionless

        if np.sum(static_mask) > 100:
            # Measure mean squared error on static regions
            raw_diff = (raw_restored[t].astype(np.float32) - raw_restored[t-1].astype(np.float32)) ** 2
            raw_err = np.mean(raw_diff[static_mask])
            raw_shimmer_errors.append(raw_err)

            filt_diff = (filtered_restored[t].astype(np.float32) - filtered_restored[t-1].astype(np.float32)) ** 2
            filt_err = np.mean(filt_diff[static_mask])
            filtered_shimmer_errors.append(filt_err)

        # Measure edge sharpness via Laplacian
        raw_sharpness.append(cv2.Laplacian(cv2.cvtColor(raw_restored[t], cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
        filtered_sharpness.append(cv2.Laplacian(cv2.cvtColor(filtered_restored[t], cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())

    mean_raw_shimmer = np.mean(raw_shimmer_errors)
    mean_filt_shimmer = np.mean(filtered_shimmer_errors)
    reduction = (mean_raw_shimmer - mean_filt_shimmer) / mean_raw_shimmer * 100.0

    print("\n--- Temporal Stability Results ---")
    print(f"Static Region Flicker (Raw APISR):      {mean_raw_shimmer:.2f} MSE")
    print(f"Static Region Flicker (APISR + Filter): {mean_filt_shimmer:.2f} MSE")
    print(f"Flicker Reduction:                      {reduction:.1f}% less shimmer")
    print(f"Mean Edge Sharpness (Raw APISR):        {np.mean(raw_sharpness):.1f}")
    print(f"Mean Edge Sharpness (APISR + Filter):   {np.mean(filtered_sharpness):.1f}")

    return mean_raw_shimmer, mean_filt_shimmer, reduction

if __name__ == "__main__":
    benchmark()
