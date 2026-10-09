# 🏗️ 03 - System Architecture & Pipeline

> End-to-end dataflow for training, local validation, and sandboxed inference.

---

## 🔄 Inference Pipeline Architecture

```
[Input Clip] -> (.mp4 or image directory)
      │
      ▼
[read_frames()] -> OpenCV frame decode (RGB uint8)
      │
      ▼
[Pre-Processor] -> Dimensions check & Tile Splitter (if frame > VRAM threshold)
      │
      ▼
[Model Engine] -> APISR / AnimeSR Backbone (FP16 Autocast)
      │
      ▼
[Post-Processor] -> Temporal Stabilizer & Dynamic Range Clamp (0-255)
      │
      ▼
[write_frames()] -> Lossless PNG (cv2.IMWRITE_PNG_COMPRESSION=1) to val_output/<clip_id>/
```

---

## 💻 Hardware Constraints & Memory Budget
* **Hardware:** NVIDIA GeForce RTX 4060 Laptop (8GB VRAM).
* **Target Memory Footprint:** Keep peak inference VRAM under **6.5 GB** to prevent CUDA Out-of-Memory (OOM) crashes in shared GPU runner environments.
* **Optimization Techniques:**
  * **FP16 Inference:** Halves tensor memory with negligible perceptual loss.
  * **Tiled Processing:** For resolutions > 1080p, process overlapping tiles with feathered blend windows.
  * **Fast PNG Encoding:** Set `[cv2.IMWRITE_PNG_COMPRESSION, 1]` to reduce disk I/O latency while retaining bit-for-bit lossless output.