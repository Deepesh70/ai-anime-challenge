"""Generate Spatio-Temporal (x-t) slice visualization for WACV 2027 paper.

Demonstrates temporal consistency across consecutive video frames:
- Raw Input: Shows compression noise across frames.
- APISR Alone (Single-Frame): Shows jagged inter-frame line shimmering in static backgrounds.
- Ours (Motion-Gated Temporal Filter): Shows smooth, temporally locked bands in static regions while preserving character kinematics.
Saves to paper/figures/temporal_slice.png
"""
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from apisr_arch import RRDBNet
from src.chroma_filter import LumaGuidedChromaFilter
from src.temporal_filter import AnimeTemporalFilter

def generate_temporal_slice(num_frames=60, scanline_y=260, x_start=200, x_width=450):
    video_path = os.path.join(root_dir, "val", "10299.mp4")
    out_path = os.path.join(root_dir, "paper", "figures", "temporal_slice.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating Spatio-Temporal slice on: {device}")

    # 1. Read frames
    cap = cv2.VideoCapture(video_path)
    raw_frames = []
    while len(raw_frames) < num_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        raw_frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Loaded {len(raw_frames)} frames.")
    h, w = raw_frames[0].shape[:2]

    # 2. Load APISR model
    model = RRDBNet(3, 3, scale=2).to(device)
    ckpt = torch.load(os.path.join(root_dir, "model_zoo", "2x_APISR_RRDB_GAN_generator.pth"), map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # 3. Process with APISR Alone (Single-frame, no temporal filter)
    print("Processing with Single-Frame APISR (EXP-01)...")
    apisr_frames = []
    with torch.no_grad():
        for frame in raw_frames:
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda"):
                out = model(tensor)
            sr_2x = out.squeeze(0).clamp(0, 1).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
            native = cv2.resize(sr_2x, (w, h), interpolation=cv2.INTER_AREA)
            apisr_frames.append(native)

    # 4. Process with Ours (Chroma + APISR + Temporal filter)
    print("Processing with Ours Pipeline (EXP-03)...")
    chroma_filter = LumaGuidedChromaFilter(radius=3, eps=2e-3, blend=0.85)
    temporal_filter = AnimeTemporalFilter(tau_static=2.5, tau_motion=9.0)
    ours_frames = []
    with torch.no_grad():
        for frame in raw_frames:
            # Stage 1: Chroma
            cleaned = chroma_filter.process(frame)
            # Stage 2: APISR
            tensor = torch.from_numpy(cleaned).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
            with torch.amp.autocast("cuda"):
                out = model(tensor)
            sr_2x = out.squeeze(0).clamp(0, 1).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
            native = cv2.resize(sr_2x, (w, h), interpolation=cv2.INTER_AREA)
            # Stage 3: Temporal
            final = temporal_filter.process(frame, native)
            ours_frames.append(final)

    # 5. Extract x-t slices
    # Dimensions: (T, x_width, 3)
    def extract_slice(frames):
        rows = [f[scanline_y, x_start : x_start + x_width] for f in frames]
        slice_img = np.array(rows, dtype=np.uint8)  # (T, x_width, 3)
        # Resize vertically to make time evolution clearly visible: (T -> 300 px)
        resized = cv2.resize(slice_img, (x_width, 300), interpolation=cv2.INTER_NEAREST)
        return resized

    slice_raw = extract_slice(raw_frames)
    slice_apisr = extract_slice(apisr_frames)
    slice_ours = extract_slice(ours_frames)

    # 6. Build annotated composite figure
    panel_w = x_width
    panel_h = 300
    banner_h = 45
    pad = 20

    total_w = 3 * panel_w + 4 * pad
    total_h = panel_h + banner_h + 90

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font_path_bold = "C:\\Windows\\Fonts\\segoeuib.ttf"
    font_path_reg = "C:\\Windows\\Fonts\\segoeui.ttf"
    font_title = ImageFont.truetype(font_path_bold, 20)
    font_sub = ImageFont.truetype(font_path_reg, 15)
    font_label = ImageFont.truetype(font_path_bold, 17)
    font_axis = ImageFont.truetype(font_path_reg, 14)

    # Title
    draw.text((pad, 15), "Spatio-Temporal (x-t) Slice Analysis: Inter-Frame Stability Verification", fill=(15, 23, 42), font=font_title)
    draw.text((pad, 42), f"Horizontal scanline at y = {scanline_y} tracked across {num_frames} consecutive frames (Vertical axis = Time t, Horizontal axis = Spatial x)", fill=(71, 85, 105), font=font_sub)
    draw.line([(pad, 70), (total_w - pad, 70)], fill=(226, 232, 240), width=1)

    panels = [
        ("Raw Input (H.264 Broadcast)", "High compression jitter", slice_raw, (220, 38, 38)),
        ("APISR Alone (Single-Frame)", "Severe line flickering / shimmer", slice_apisr, (234, 88, 12)),
        ("Ours Pipeline (EXP-03)", "Temporally locked background cel", slice_ours, (22, 101, 52)),
    ]

    cur_x = pad
    y_panel_top = 85

    for title, subtitle, img_np, color in panels:
        # Header banner
        draw.rounded_rectangle((cur_x, y_panel_top, cur_x + panel_w, y_panel_top + banner_h), radius=4, fill=(248, 250, 252), outline=(203, 213, 225), width=1)
        draw.text((cur_x + 12, y_panel_top + 6), title, fill=color, font=font_label)
        draw.text((cur_x + 12, y_panel_top + 26), subtitle, fill=(100, 116, 139), font=font_axis)

        # Image panel
        panel_pil = Image.fromarray(img_np)
        canvas.paste(panel_pil, (cur_x, y_panel_top + banner_h + 5))

        # Thin border around panel
        draw.rectangle((cur_x, y_panel_top + banner_h + 5, cur_x + panel_w, y_panel_top + banner_h + 5 + panel_h), outline=(203, 213, 225), width=1)
        cur_x += panel_w + pad

    canvas.save(out_path, "PNG", dpi=(300, 300))
    print(f"Spatio-Temporal slice figure saved to:\n  {out_path}")

if __name__ == "__main__":
    generate_temporal_slice()
