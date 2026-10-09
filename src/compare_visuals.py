"""Generate side-by-side visual comparisons between Original, FBCNN, and APISR."""
import os
import sys

# Ensure root directory is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import cv2
import torch
from fbcnn_arch import FBCNN
from apisr_arch import RRDBNet

def compare():
    os.makedirs("comparisons", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running visual comparison on: {device}")

    # 1. Read first frame from val/10299.mp4
    cap = cv2.VideoCapture("val/10299.mp4")
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read frame from val/10299.mp4")
    
    h, w, _ = frame_bgr.shape
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite("comparisons/frame0_original.png", frame_bgr)
    print(f"Saved original frame: {w}x{h}")

    # 2. Run FBCNN
    fbcnn = FBCNN(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R')
    fbcnn.load_state_dict(torch.load("model_zoo/fbcnn_color.pth", map_location='cpu', weights_only=True))
    fbcnn.eval().to(device)

    with torch.no_grad():
        x = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        with torch.amp.autocast('cuda'):
            out_fbcnn = fbcnn(x)[0]
        fbcnn_np = out_fbcnn.squeeze(0).clamp(0, 1).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
    
    cv2.imwrite("comparisons/frame0_fbcnn.png", cv2.cvtColor(fbcnn_np, cv2.COLOR_RGB2BGR))
    print("Saved FBCNN restored frame")

    # 3. Run APISR (2x + INTER_AREA downsample)
    apisr = RRDBNet(3, 3, scale=2)
    ckpt = torch.load("model_zoo/2x_APISR_RRDB_GAN_generator.pth", map_location='cpu', weights_only=False)['model_state_dict']
    apisr.load_state_dict(ckpt, strict=True)
    apisr.eval().to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            out_apisr = apisr(x)
        apisr_2x = out_apisr.squeeze(0).clamp(0, 1).mul(255.0).round().byte().permute(1, 2, 0).cpu().numpy()
        # Downsample back to native resolution using area averaging
        apisr_native = cv2.resize(apisr_2x, (w, h), interpolation=cv2.INTER_AREA)

    cv2.imwrite("comparisons/frame0_apisr.png", cv2.cvtColor(apisr_native, cv2.COLOR_RGB2BGR))
    print("Saved APISR restored frame")

    # 4. Create annotated side-by-side composite
    composite = cv2.hconcat([frame_bgr, cv2.cvtColor(fbcnn_np, cv2.COLOR_RGB2BGR), cv2.cvtColor(apisr_native, cv2.COLOR_RGB2BGR)])
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(composite, "Original (480p)", (30, 40), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(composite, "FBCNN (Baseline)", (w + 30, 40), font, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(composite, "APISR (Anime SR)", (2 * w + 30, 40), font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imwrite("comparisons/frame0_side_by_side.png", composite)
    print("Saved side-by-side comparison: comparisons/frame0_side_by_side.png")

if __name__ == "__main__":
    compare()
