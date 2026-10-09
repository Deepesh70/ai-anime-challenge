# 📐 02 - Evaluation Metrics & NR-IQA

> The benchmark evaluates models using **No-Reference Image/Video Quality Assessment (NR-IQA)** because native anime source material lacks pristine ground-truth paired degradation.

---

## 🔍 The Four Core NR-IQA Axes

### 1. MANIQA (Multi-dimension Attention Network for IQA)
* **What it measures:** Patch-level perceptual quality using multi-dimension ViT attention.
* **Why it matters:** Highly sensitive to structural blurring, loss of fine anime linework, and unnatural smoothing.
* **Optimization target:** Higher score is better.

### 2. CLIPIQA (Contrastive Language-Image Pre-Training IQA)
* **What it measures:** Alignment of visual features against aesthetic and perceptual text prompts in CLIP latent space.
* **Why it matters:** Evaluates whether the generated output looks like a clean, high-grade production anime cel rather than a noisy, low-bitrate stream.
* **Optimization target:** Higher score is better.

### 3. HyperIQA (Self-Adaptive Hyper-Network IQA)
* **What it measures:** Evaluates perceptual quality by splitting image understanding into content classification and distortion estimation via dynamic hypernetworks.
* **Why it matters:** Catches compression artifacts, color quantization bands, and ringing without relying on reference frames.
* **Optimization target:** Higher score is better.

### 4. NIQE (Natural Image Quality Evaluator)
* **What it measures:** Statistical distance between natural scene statistics (NSS) features and the restored frame.
* **Anime adaptation note:** Traditional NIQE penalizes flat cel colors as "unnatural", so the Re-Anime600 benchmark uses domain-calibrated filtering.
* **Optimization target:** Lower score is better.

---

## ⚡ Temporal Coherence & Artifact Constraints
* **Flickering Penalty:** Single-frame super-resolution models often hallucinate slightly different line contours per frame, causing jitter.
* **Ringing & Halos:** Over-sharpened kernels produce dark or bright halos along hand-drawn contours.
* **Color Bleeding:** Compression algorithms (H.264 chroma subsampling 4:2:0) smudge vibrant colors into adjacent lines.