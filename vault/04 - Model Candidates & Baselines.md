# 🔬 04 - Model Candidates & Baselines

> Comparative analysis of state-of-the-art anime restoration backbones against challenge requirements.

---

## 🥊 Candidate Comparison & Measured Benchmarks

| Model | Venue | Weights Size | Measured Latency | Edge Clarity (Laplacian Var) | Perceptual Character |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FBCNN (Baseline)** | ICCV 2021 | 274 MB | 0.216 s/frame (4.6 fps) | 1230.2 | Smooths JPEG blocks, but blurs fine anime line art. |
| **APISR (RRDB-6B)** | CVPR 2024 | **17.9 MB** | 0.232 s/frame (4.3 fps) | **2616.5** (+112%) | Reconstructs dark ink contours, eliminates chroma bleed, cel-preserving. |
| **AnimeSR** | NeurIPS 2022 | ~60 MB | Queued | Pending | Video-specific degradation model. |
| **Real-ESRGAN Anime** | ICCV 2021 | ~17 MB | Queued | Pending | Heavy sharpening; potential ringing. |

---

## 🔍 APISR Implementation Mechanics
* **Input Unshuffle:** Input $(B, 3, H, W)$ is pixel-unshuffled to $(B, 12, H/2, W/2)$, halving spatial dimensions and quadrupling channels for efficient feature extraction.
* **Deep Feature Extraction:** Processed by 6 Residual-in-Residual Dense Blocks (RRDB) with LeakyReLU activations and residual scaling ($0.2$).
* **Upsampling:** Progressively upsampled with two $2x$ nearest interpolations followed by convolutions to reach $(B, 3, 2H, 2W)$.
* **Native 1x Downsampling:** Using `cv2.INTER_AREA`, the 2x reconstructed frame is anti-aliased and downsampled back to $(H, W, 3)$ matching the exact input resolution.