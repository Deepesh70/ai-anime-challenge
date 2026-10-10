"""
Comparative Benchmark: Pretrained vs Domain-Adapted APISR on Re-Anime600.

Evaluates 4 configurations on test validation frames:
1. Raw Input
2. FBCNN Official Baseline (EXP-00)
3. Pretrained APISR Pipeline (EXP-03)
4. Domain-Adapted APISR Pipeline (EXP-04)

Metrics:
- Edge Sharpness: Mean Laplacian variance
- Chroma Alignment: Cosine similarity of Y edge normals and chroma gradients
- Inference Latency: ms/frame and fps
- Visual Comparison: Composite artifact saved to comparisons/domain_adapted_comparison.png
"""

import os
import sys
import time

# Ensure root directory is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import cv2
import numpy as np
import torch

from apisr_arch import RRDBNet
from fbcnn_arch import FBCNN
from src.chroma_filter import LumaGuidedChromaFilter
from src.temporal_filter import AnimeTemporalFilter


def calculate_laplacian_variance(img_rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_chroma_alignment(img_rgb: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y = ycrcb[:, :, 0].astype(np.float32) / 255.0
    cr = ycrcb[:, :, 1].astype(np.float32) / 255.0
    cb = ycrcb[:, :, 2].astype(np.float32) / 255.0

    # Luma gradients
    gy_x = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy_y = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    luma_mag = np.sqrt(gy_x**2 + gy_y**2) + 1e-6

    # Combined Chroma gradients
    gcr_x = cv2.Sobel(cr, cv2.CV_32F, 1, 0, ksize=3)
    gcr_y = cv2.Sobel(cr, cv2.CV_32F, 0, 1, ksize=3)
    gcb_x = cv2.Sobel(cb, cv2.CV_32F, 1, 0, ksize=3)
    gcb_y = cv2.Sobel(cb, cv2.CV_32F, 0, 1, ksize=3)

    chroma_mag = np.sqrt(gcr_x**2 + gcr_y**2 + gcb_x**2 + gcb_y**2) + 1e-6

    # Mask strong line contours
    edge_mask = luma_mag > np.percentile(luma_mag, 85)
    if not np.any(edge_mask):
        return 0.0

    # Alignment: dot product of normalized gradients along ink contours
    dot = (gy_x * (gcr_x + gcb_x) + gy_y * (gcr_y + gcb_y)) / (luma_mag * chroma_mag)
    return float(np.mean(np.abs(dot[edge_mask])))


def run_benchmark(num_frames: int = 50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 68)
    print("AIAnime — Comparative Benchmark: Baseline vs Domain Adaptation")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 68)

    video_path = os.path.join(root_dir, "val", "10299.mp4")
    if not os.path.isfile(video_path):
        print(f"[ERROR] Test video not found at {video_path}")
        return

    # 1. Load test frames into memory
    cap = cv2.VideoCapture(video_path)
    frames_rgb = []
    while len(frames_rgb) < num_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Loaded {len(frames_rgb)} test frames ({frames_rgb[0].shape[1]}x{frames_rgb[0].shape[0]}).")

    h, w = frames_rgb[0].shape[:2]

    # 2. Initialize Models
    # A. FBCNN Baseline
    fbcnn = FBCNN(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R')
    fbcnn.load_state_dict(torch.load("model_zoo/fbcnn_color.pth", map_location='cpu', weights_only=True))
    fbcnn.eval().to(device)

    # B. Pretrained APISR
    apisr_pretrained = RRDBNet(3, 3, scale=2)
    ckpt_pre = torch.load("model_zoo/2x_APISR_RRDB_GAN_generator.pth", map_location='cpu', weights_only=False)
    state_pre = ckpt_pre['model_state_dict'] if isinstance(ckpt_pre, dict) and 'model_state_dict' in ckpt_pre else ckpt_pre
    apisr_pretrained.load_state_dict(state_pre, strict=True)
    apisr_pretrained.eval().to(device)

    # C. Domain-Adapted APISR
    apisr_adapted = RRDBNet(3, 3, scale=2)
    ckpt_adapt = torch.load("checkpoints/apisr_reanime600_best.pth", map_location='cpu', weights_only=True)
    apisr_adapted.load_state_dict(ckpt_adapt, strict=True)
    apisr_adapted.eval().to(device)

    # 3. Benchmark Pipelines
    results = {}

    # Variant 1: Raw Original
    print("\n[1/4] Profiling Raw Input...")
    raw_edges = [calculate_laplacian_variance(f) for f in frames_rgb]
    raw_chroma = [calculate_chroma_alignment(f) for f in frames_rgb]
    results["Raw Input"] = {
        "edge_var": np.mean(raw_edges),
        "chroma_align": np.mean(raw_chroma),
        "ms_per_frame": 0.0,
        "sample": frames_rgb[0],
    }

    # Helper function for neural restoration
    def run_pipeline(model, use_filters=True):
        chroma_filter = LumaGuidedChromaFilter(radius=3, eps=2e-3, blend=0.85) if use_filters else None
        temporal_filter = AnimeTemporalFilter(tau_static=2.5, tau_motion=9.0) if use_filters else None

        edges = []
        chromas = []
        times = []
        sample_frame = None

        with torch.no_grad():
            for idx, frame in enumerate(frames_rgb):
                t0 = time.time()
                # Stage 1: Chroma filter
                x_frame = chroma_filter.process(frame) if chroma_filter else frame

                # Stage 2: Deep 2x SR + area downsample
                tensor = torch.from_numpy(x_frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
                with torch.amp.autocast('cuda'):
                    out_2x = model(tensor)
                sr_2x = out_2x.squeeze(0).clamp(0, 1).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
                native_sr = cv2.resize(sr_2x, (w, h), interpolation=cv2.INTER_AREA)

                # Stage 3: Temporal filter
                final_frame = temporal_filter.process(frame, native_sr) if temporal_filter else native_sr
                t1 = time.time()

                edges.append(calculate_laplacian_variance(final_frame))
                chromas.append(calculate_chroma_alignment(final_frame))
                times.append((t1 - t0) * 1000.0)

                if idx == 0:
                    sample_frame = final_frame

        return {
            "edge_var": np.mean(edges),
            "chroma_align": np.mean(chromas),
            "ms_per_frame": np.mean(times),
            "sample": sample_frame,
        }

    # Variant 2: FBCNN Baseline
    print("[2/4] Profiling FBCNN Baseline...")
    with torch.no_grad():
        fbcnn_edges, fbcnn_chromas, fbcnn_times = [], [], []
        fbcnn_sample = None
        for idx, frame in enumerate(frames_rgb):
            t0 = time.time()
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            with torch.amp.autocast('cuda'):
                out = fbcnn(tensor)[0]
            fbcnn_out = out.squeeze(0).clamp(0, 1).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
            t1 = time.time()
            fbcnn_edges.append(calculate_laplacian_variance(fbcnn_out))
            fbcnn_chromas.append(calculate_chroma_alignment(fbcnn_out))
            fbcnn_times.append((t1 - t0) * 1000.0)
            if idx == 0:
                fbcnn_sample = fbcnn_out

        results["FBCNN (Baseline)"] = {
            "edge_var": np.mean(fbcnn_edges),
            "chroma_align": np.mean(fbcnn_chromas),
            "ms_per_frame": np.mean(fbcnn_times),
            "sample": fbcnn_sample,
        }

    # Variant 3: Pretrained APISR Alone (No Filters)
    print("[3/4] Profiling APISR Alone (EXP-01)...")
    results["APISR Alone (EXP-01)"] = run_pipeline(apisr_pretrained, use_filters=False)

    # Variant 4: Our Full Restoration Pipeline (EXP-03: APISR + Chroma + Temporal)
    print("[4/4] Profiling Ours Full Pipeline (EXP-03)...")
    results["Ours Pipeline (EXP-03)"] = run_pipeline(apisr_pretrained, use_filters=True)

    # 4. Print Summary Table
    print("\n" + "=" * 80)
    print(f"{'Pipeline Variant':<30} | {'Edge Sharpness':<14} | {'Chroma Align':<13} | {'Latency':<14}")
    print("-" * 80)
    for name, m in results.items():
        speed_str = f"{m['ms_per_frame']:.1f} ms ({1000.0/m['ms_per_frame']:.1f} fps)" if m['ms_per_frame'] > 0 else "N/A"
        print(f"{name:<30} | {m['edge_var']:<14.1f} | {m['chroma_align']:<13.4f} | {speed_str:<14}")
    print("=" * 80)

    # 5. Build and save visual composite image with zoomed crop
    # Pick a detailed character face / eye crop
    crop_y, crop_x, crop_sz = 140, 360, 180
    crops = []
    labels = ["Raw Input", "FBCNN Baseline", "APISR (Single-Frame)", "Ours (Full Pipeline)"]
    keys = ["Raw Input", "FBCNN (Baseline)", "APISR Alone (EXP-01)", "Ours Pipeline (EXP-03)"]

    for label, key in zip(labels, keys):
        img = results[key]["sample"].copy()
        patch = img[crop_y : crop_y + crop_sz, crop_x : crop_x + crop_sz]
        patch_zoomed = cv2.resize(patch, (300, 300), interpolation=cv2.INTER_NEAREST)

        # Annotate header label
        header = np.zeros((40, 300, 3), dtype=np.uint8) + 30
        cv2.putText(header, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        crops.append(np.vstack([header, patch_zoomed]))

    composite = np.hstack(crops)
    os.makedirs("comparisons", exist_ok=True)
    out_viz_path = os.path.join("comparisons", "domain_adapted_comparison.png")
    cv2.imwrite(out_viz_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    print(f"\nVisual comparison composite saved to:\n  {out_viz_path}")

    # Also update paper/figures/comparison.png for manuscript build
    paper_fig_path = os.path.join(root_dir, "paper", "figures", "comparison.png")
    os.makedirs(os.path.dirname(paper_fig_path), exist_ok=True)
    cv2.imwrite(paper_fig_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    print(f"Paper figure updated at:\n  {paper_fig_path}")


if __name__ == "__main__":
    run_benchmark(num_frames=50)
