"""Benchmark Luma-Guided Chroma Filter and measure color bleed correction."""
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import time
import cv2
import numpy as np
import torch
from apisr_arch import RRDBNet
from src.chroma_filter import LumaGuidedChromaFilter

def benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running Chroma Bleed Benchmark on: {device}")

    # 1. Load test frame from val/10299.mp4
    cap = cv2.VideoCapture("val/10299.mp4")
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read frame from val/10299.mp4")
    
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = rgb.shape

    # 2. Benchmark Chroma Filter speed
    filter_obj = LumaGuidedChromaFilter(radius=3, eps=2e-3, blend=0.85)
    
    # Warmup
    _ = filter_obj.process(rgb)
    
    t0 = time.time()
    num_runs = 30
    for _ in range(num_runs):
        chroma_cleaned_rgb = filter_obj.process(rgb)
    dt = (time.time() - t0) / num_runs
    print(f"Chroma Filter Latency: {dt * 1000.0:.2f} ms/frame ({1.0/dt:.1f} fps)")

    # 3. Measure Chroma Edge Alignment along Line Contours
    # Convert to YCrCb to analyze chroma channels
    def compute_edge_chroma_coherence(img_rgb):
        ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        y = ycrcb[:, :, 0]
        cb = ycrcb[:, :, 2]
        cr = ycrcb[:, :, 1]

        # Sobel gradients
        gy_x = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        gy_y = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
        mag_y = np.sqrt(gy_x**2 + gy_y**2)

        # Detect ink line edges (top 15% gradient)
        edge_mask = mag_y > np.percentile(mag_y, 85)

        gcb_x = cv2.Sobel(cb, cv2.CV_32F, 1, 0, ksize=3)
        gcb_y = cv2.Sobel(cb, cv2.CV_32F, 0, 1, ksize=3)
        mag_cb = np.sqrt(gcb_x**2 + gcb_y**2)

        # Normalized cosine similarity between luma edge normal and chroma gradient
        dot = gy_x * gcb_x + gy_y * gcb_y
        denom = (mag_y * mag_cb) + 1e-5
        alignment = np.abs(dot / denom)
        
        # Mean alignment along ink lines (higher = color strictly conforms to line contours)
        return float(np.mean(alignment[edge_mask]))

    raw_alignment = compute_edge_chroma_coherence(rgb)
    clean_alignment = compute_edge_chroma_coherence(chroma_cleaned_rgb)
    improvement = (clean_alignment - raw_alignment) / raw_alignment * 100.0

    print(f"\n--- Chroma Edge Alignment (Higher = Less Color Bleed) ---")
    print(f"Raw Input Alignment:            {raw_alignment:.4f}")
    print(f"Chroma-Filtered Alignment:      {clean_alignment:.4f}")
    print(f"Contour Alignment Improvement:  +{improvement:.1f}% tighter color boundaries")

    # 4. End-to-end test with APISR
    model = RRDBNet(3, 3, scale=2)
    ckpt = torch.load("model_zoo/2x_APISR_RRDB_GAN_generator.pth", map_location='cpu', weights_only=False)['model_state_dict']
    model.load_state_dict(ckpt, strict=True)
    model.eval().to(device)

    def restore_frame(inp_rgb):
        x = torch.from_numpy(inp_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = model(x)
        out_np = out.squeeze(0).clamp(0, 1).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        return cv2.resize(out_np, (w, h), interpolation=cv2.INTER_AREA)

    apisr_only = restore_frame(rgb)
    apisr_chroma = restore_frame(chroma_cleaned_rgb)

    # Save visual comparison
    os.makedirs("comparisons", exist_ok=True)
    cv2.imwrite("comparisons/frame0_chroma_filtered.png", cv2.cvtColor(chroma_cleaned_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite("comparisons/frame0_apisr_with_chroma.png", cv2.cvtColor(apisr_chroma, cv2.COLOR_RGB2BGR))
    print("Saved comparison frames to comparisons/")

    return dt * 1000.0, raw_alignment, clean_alignment, improvement

if __name__ == "__main__":
    benchmark()
