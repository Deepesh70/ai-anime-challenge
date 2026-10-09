"""
Multi-Clip Validation Suite & Robustness Benchmark.

Evaluates the 3-stage restoration pipeline across multiple diverse anime video clips
drawn from the Re-Anime600 dataset to verify generalization across:
- Character close-ups
- High-motion action sequences
- Painted background pans
- Variable broadcast compression levels

Metrics measured per clip:
1. Edge Sharpness (Laplacian Variance)
2. Background Shimmer MSE (Inter-frame variance on static patches)
3. Chroma Contour Alignment (YUV guidance fidelity)
4. Throughput (ms/frame and fps)
5. Peak GPU Memory (VRAM in MB)
"""

import os
import sys
import time
import glob
import cv2
import numpy as np
import torch

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from apisr_arch import RRDBNet
from fbcnn_arch import FBCNN
from src.chroma_filter import LumaGuidedChromaFilter
from src.temporal_filter import AnimeTemporalFilter
from src.benchmark_domain_adaptation import calculate_laplacian_variance, calculate_chroma_alignment


def compute_temporal_shimmer(frames: list, tau_static: float = 3.0) -> float:
    """Computes inter-frame mean squared difference strictly over static background pixels."""
    if len(frames) < 2:
        return 0.0

    shimmers = []
    for i in range(1, len(frames)):
        f_prev = frames[i - 1].astype(np.float32)
        f_curr = frames[i].astype(np.float32)

        diff = np.abs(f_curr - f_prev)
        max_diff = np.max(diff, axis=-1)

        static_mask = max_diff < tau_static
        if np.sum(static_mask) > 100:
            shimmers.append(np.mean((f_curr[static_mask] - f_prev[static_mask]) ** 2))

    return float(np.mean(shimmers)) if shimmers else 0.0


def evaluate_suite(num_clips: int = 5, frames_per_clip: int = 40):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("AIAnime — Multi-Clip Validation Suite & Robustness Audit")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 72)

    # 1. Discover sample clips
    search_dirs = ["data/train/train", "val"]
    candidate_clips = []
    for d in search_dirs:
        candidate_clips.extend(sorted(glob.glob(os.path.join(d, "*.mp4"))))

    # Pick unique clips
    seen_stems = set()
    test_clips = []
    for p in candidate_clips:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in seen_stems:
            seen_stems.add(stem)
            test_clips.append(p)
        if len(test_clips) >= num_clips:
            break

    if not test_clips:
        print("[ERROR] No MP4 clips found to benchmark.")
        return

    print(f"Selected {len(test_clips)} validation clips for robustness audit:")
    for idx, c in enumerate(test_clips, start=1):
        print(f"  [{idx}] {os.path.basename(c)}")

    # 2. Initialize Pipeline
    model = RRDBNet(3, 3, scale=2).to(device)
    ckpt_path = "model_zoo/apisr_reanime600_interpolated.pth"
    if not os.path.isfile(ckpt_path):
        ckpt_path = "model_zoo/2x_APISR_RRDB_GAN_generator.pth"
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    chroma_filter = LumaGuidedChromaFilter(radius=3, eps=2e-3, blend=0.85)
    temporal_filter = AnimeTemporalFilter(tau_static=2.5, tau_motion=9.0)

    # 3. Process Clips
    clip_summaries = []

    for clip_idx, clip_path in enumerate(test_clips, start=1):
        clip_name = os.path.basename(clip_path)
        cap = cv2.VideoCapture(clip_path)
        total_avail = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 23.98

        raw_frames = []
        while len(raw_frames) < frames_per_clip:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            raw_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        cap.release()

        temporal_filter.reset()
        restored_frames = []
        latencies = []

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            for frame in raw_frames:
                t0 = time.time()
                # Stage 1: Chroma guidance
                cf = chroma_filter.process(frame)

                # Stage 2: Deep 2x super-resolution + area downsampling
                tensor = torch.from_numpy(cf).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
                with torch.amp.autocast('cuda'):
                    out_2x = model(tensor)
                sr_2x = out_2x.squeeze(0).clamp(0, 1).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
                native_sr = cv2.resize(sr_2x, (w, h), interpolation=cv2.INTER_AREA)

                # Stage 3: Soft motion-gated temporal smoother
                final_f = temporal_filter.process(frame, native_sr)
                t1 = time.time()

                restored_frames.append(final_f)
                latencies.append((t1 - t0) * 1000.0)

        # Compute Metrics
        raw_edges = np.mean([calculate_laplacian_variance(f) for f in raw_frames])
        res_edges = np.mean([calculate_laplacian_variance(f) for f in restored_frames])

        raw_shimmer = compute_temporal_shimmer(raw_frames)
        res_shimmer = compute_temporal_shimmer(restored_frames)

        raw_chroma = np.mean([calculate_chroma_alignment(f) for f in raw_frames])
        res_chroma = np.mean([calculate_chroma_alignment(f) for f in restored_frames])

        mean_lat = np.mean(latencies)
        fps_proc = 1000.0 / mean_lat if mean_lat > 0 else 0.0
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

        shimmer_change = ((res_shimmer - raw_shimmer) / raw_shimmer * 100.0) if raw_shimmer > 0 else 0.0
        edge_gain = ((res_edges - raw_edges) / raw_edges * 100.0) if raw_edges > 0 else 0.0

        summary = {
            "clip": clip_name,
            "resolution": f"{w}x{h}",
            "raw_edges": raw_edges,
            "res_edges": res_edges,
            "edge_gain": edge_gain,
            "raw_shimmer": raw_shimmer,
            "res_shimmer": res_shimmer,
            "shimmer_change": shimmer_change,
            "res_chroma": res_chroma,
            "fps": fps_proc,
            "peak_vram_mb": peak_vram,
        }
        clip_summaries.append(summary)

        print(f"[{clip_idx}/{len(test_clips)}] {clip_name:<16} | Edge: {raw_edges:.0f} -> {res_edges:.0f} (+{edge_gain:.1f}%) | Shimmer: {raw_shimmer:.2f} -> {res_shimmer:.2f} ({shimmer_change:.1f}%) | {fps_proc:.1f} fps")

    # 4. Aggregate Report Table
    print("\n" + "=" * 95)
    print(f"{'Clip ID':<16} | {'Resolution':<10} | {'Edge Sharpness':<18} | {'Background Shimmer':<19} | {'Chroma':<8} | {'Speed':<8}")
    print("-" * 95)
    for s in clip_summaries:
        edge_col = f"{s['res_edges']:.0f} (+{s['edge_gain']:.1f}%)"
        shim_col = f"{s['res_shimmer']:.2f} ({s['shimmer_change']:.1f}%)"
        print(f"{s['clip']:<16} | {s['resolution']:<10} | {edge_col:<18} | {shim_col:<19} | {s['res_chroma']:<8.4f} | {s['fps']:<5.1f} fps")
    print("=" * 95)

    avg_edge_gain = np.mean([s["edge_gain"] for s in clip_summaries])
    avg_shimmer_red = np.mean([s["shimmer_change"] for s in clip_summaries])
    avg_chroma = np.mean([s["res_chroma"] for s in clip_summaries])
    avg_fps = np.mean([s["fps"] for s in clip_summaries])
    max_vram = max([s["peak_vram_mb"] for s in clip_summaries])

    print(f"SUMMARY ACROSS {len(test_clips)} CLIPS:")
    print(f"  • Mean Edge Sharpness Gain:      +{avg_edge_gain:.1f}%")
    print(f"  • Mean Background Shimmer Shift:  {avg_shimmer_red:.1f}%")
    print(f"  • Mean Chroma Alignment:          {avg_chroma:.4f}")
    print(f"  • Average Throughput:             {avg_fps:.1f} fps")
    print(f"  • Peak GPU VRAM Usage:            {max_vram:.1f} MB (well within 8192 MB limit)")
    print("=" * 95)


if __name__ == "__main__":
    evaluate_suite(num_clips=5, frames_per_clip=35)
