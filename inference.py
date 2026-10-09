"""AINIME @ WACV 2027 Challenge: Official Submission Interface.

Usage:
    python inference.py

Reads every clip (.mp4 or directory of frames) in ./val
and writes ./val_output/<clip_id>/000000.png, 000001.png, ...
Output frames strictly match input resolution, count, and order.
Lossless PNG compression is enforced.
"""
import argparse
import os
import re
import time
import cv2
import torch
import torch.multiprocessing as mp

from fbcnn_arch import FBCNN
from apisr_arch import RRDBNet
from src.temporal_filter import AnimeTemporalFilter
from src.chroma_filter import LumaGuidedChromaFilter

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
VID_EXT = ('.mp4', '.mkv', '.avi', '.mov', '.webm')


# ----------------------------------------------------------------------------------------
# Model-specific: load weights and perform single-frame / sequence restoration
# ----------------------------------------------------------------------------------------
def load_model(checkpoint, device, model_type='apisr'):
    """Load neural network onto target device."""
    if model_type == 'apisr':
        model = RRDBNet(3, 3, scale=2)
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
    elif model_type == 'fbcnn':
        model = FBCNN(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R')
        state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    return model.eval().to(device)


@torch.no_grad()
def restore(model, img, device, model_type='apisr'):
    """RGB uint8 (H, W, 3) -> RGB uint8 (H, W, 3) with identical dimensions."""
    h, w, _ = img.shape
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    
    # Use FP16 autocast on CUDA for lower memory and faster inference
    if device.type == 'cuda':
        with torch.amp.autocast('cuda'):
            out = model(x)
    else:
        out = model(x)

    if isinstance(out, (tuple, list)):
        out = out[0]

    out_np = out.squeeze(0).clamp(0, 1).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    
    # If 2x super-resolution was performed, downsample back to native resolution
    if model_type == 'apisr' and (out_np.shape[0] != h or out_np.shape[1] != w):
        out_np = cv2.resize(out_np, (w, h), interpolation=cv2.INTER_AREA)

    return out_np


# ----------------------------------------------------------------------------------------
# I/O and frame handling
# ----------------------------------------------------------------------------------------
def natural_key(name):
    """Sort strings with embedded numbers naturally (e.g. 1, 2, 10)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', name)]


def list_clips(root):
    """Find [(clip_id, path)] for every sub-folder or video file in root."""
    clips = []
    for name in sorted(os.listdir(root), key=natural_key):
        path = os.path.join(root, name)
        stem, ext = os.path.splitext(name)
        if os.path.isdir(path):
            clips.append((name, path))
        elif ext.lower() in VID_EXT:
            clips.append((stem, path))
    return clips


def read_frames(path):
    """Yield RGB uint8 frames sequentially from a video file or folder of frames."""
    if os.path.isdir(path):
        for name in sorted((f for f in os.listdir(path) if f.lower().endswith(IMG_EXT)), key=natural_key):
            frame = cv2.imread(os.path.join(path, name), cv2.IMREAD_COLOR)
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f'Failed to open video stream: {path}')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()


def worker(rank, devices, clips, checkpoint, output, model_type='apisr', use_temporal=True, use_chroma=True):
    """Worker process assigned to a specific GPU/CPU device."""
    device = devices[rank]
    model = load_model(checkpoint, device, model_type=model_type)
    temp_filter = AnimeTemporalFilter() if use_temporal else None
    chroma_filter = LumaGuidedChromaFilter() if use_chroma else None
    
    for cid, path in clips[rank::len(devices)]:
        t0, count = time.time(), 0
        if temp_filter is not None:
            temp_filter.reset()

        try:
            out_clip_dir = os.path.join(output, cid)
            os.makedirs(out_clip_dir, exist_ok=True)
            
            for i, frame in enumerate(read_frames(path)):
                # 1. Pre-filter broadcast chroma bleed across line contours
                if chroma_filter is not None:
                    frame = chroma_filter.process(frame)

                # 2. Reconstruct high-frequency lines & details
                restored = restore(model, frame, device, model_type=model_type)
                
                # 3. Apply motion-gated temporal filter to kill background line shimmer
                if temp_filter is not None:
                    restored = temp_filter.process(frame, restored)

                if restored.shape != frame.shape:
                    raise ValueError(f'Dimension mismatch: output {restored.shape} != input {frame.shape}')
                
                # Write lossless PNG with fastest compression level (level 1)
                bgr = cv2.cvtColor(restored, cv2.COLOR_RGB2BGR)
                out_path = os.path.join(out_clip_dir, f'{i:06d}.png')
                cv2.imwrite(out_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                count += 1
                
        except Exception as e:
            print(f'[{device}] {cid}: FAILED ({e})', flush=True)
            continue
            
        dt = time.time() - t0
        speed = dt / max(count, 1)
        fps = count / max(dt, 1e-4)
        tags = []
        if use_chroma:
            tags.append("ChromaClean")
        if use_temporal:
            tags.append("TemporalLock")
        tag_str = f" + {'+'.join(tags)}" if tags else ""
        print(f'[{device}] {cid} [{model_type.upper()}{tag_str}]: {count} frames processed in {dt:.1f}s ({speed:.3f} s/frame, {fps:.1f} fps)', flush=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="AINIME Video Restoration Inference")
    ap.add_argument('--model', choices=['apisr', 'fbcnn'], default='apisr', help="Model backbone to run")
    ap.add_argument('--checkpoint', default=None, help="Path to checkpoint file")
    ap.add_argument('--input', default=os.path.join(here, 'val'))
    ap.add_argument('--output', default=os.path.join(here, 'val_output'))
    ap.add_argument('--no-temporal', action='store_true', help="Disable motion-gated temporal consistency filter")
    ap.add_argument('--no-chroma', action='store_true', help="Disable luma-guided chroma bleed pre-filter")
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    use_temporal = not args.no_temporal
    use_chroma = not args.no_chroma

    # Determine default checkpoint based on model type
    if args.checkpoint is None:
        if args.model == 'apisr':
            interpolated_ckpt = os.path.join(here, 'model_zoo', 'apisr_reanime600_interpolated.pth')
            if os.path.isfile(interpolated_ckpt):
                args.checkpoint = interpolated_ckpt
            else:
                args.checkpoint = os.path.join(here, 'model_zoo', '2x_APISR_RRDB_GAN_generator.pth')
        else:
            args.checkpoint = os.path.join(here, 'model_zoo', 'fbcnn_color.pth')

    if not os.path.isfile(args.checkpoint):
        raise SystemExit(f'Checkpoint not found at: {args.checkpoint}')
    if not os.path.isdir(args.input):
        raise SystemExit(f'Input folder not found at: {args.input}')
        
    clips = list_clips(args.input)
    if not clips:
        raise SystemExit(f'No clips found in input folder: {args.input}')

    n_gpu = 0 if args.cpu else torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(n_gpu)] or [torch.device('cpu')]
    devices = devices[:len(clips)]
    temporal_str = "Enabled" if use_temporal else "Disabled"
    chroma_str = "Enabled" if use_chroma else "Disabled"
    print(f'Starting inference [{args.model.upper()} | Chroma: {chroma_str} | Temporal: {temporal_str}]: {len(clips)} clip(s) on device(s): {", ".join(map(str, devices))}')

    if len(devices) == 1:
        worker(0, devices, clips, args.checkpoint, args.output, model_type=args.model, use_temporal=use_temporal, use_chroma=use_chroma)
    else:
        mp.spawn(worker, args=(devices, clips, args.checkpoint, args.output, args.model, use_temporal, use_chroma), nprocs=len(devices))


if __name__ == '__main__':
    main()
