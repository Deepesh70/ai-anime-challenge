# 💡 07 - Research Contributions & Technical Novelty

> Defines our original engineering innovations and the research methodology for our WACV 2027 workshop paper.

---

## 🎯 Why Off-the-Shelf Pretrained Models Are Insufficient
Pretrained image models like APISR, AnimeSR, and Real-ESRGAN represent our **foundational feature extractors (Step 0)**. Submitting an unmodified checkpoint cannot win the competition or justify an archival publication for three fundamental reasons:
1. **The Single-Image Video Trap:** Image super-resolution models hallucinate independent sub-pixel features on every frame, generating severe **high-frequency temporal flickering (line shimmer)** across video sequences.
2. **Domain Mismatch:** APISR was trained on static web illustrations with synthetic Gaussian noise and generic JPEG compression. Re-Anime600 consists of broadcast television video streams with H.264 macroblocking, bitrate starvation, and 4:2:0 chroma bleeding.
3. **Metric Balancing:** Aggressive edge sharpening boosts MANIQA scores but degrades NIQE and HyperIQA by accentuating flat cel noise.

---

## 🏛️ The Three Pillars of Our System Contribution

```
[Degraded 480p Anime Video]
           │
           ▼
┌──────────────────────────────────────────────┐
│  Pillar 1: Chroma Bleed & Luma Separation    │ ──> Corrects 4:2:0 color smear across ink contours
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Pillar 2: Domain-Adapted APISR Backbone     │ ──> Fine-tuned on Re-Anime600 broadcast degradations
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Pillar 3: Anime Motion Temporal Filter      │ ──> Motion-masked background freezing vs character
└──────────────────────┬───────────────────────┘     motion to kill frame-to-frame line shimmer
                       │
                       ▼
[Temporal-Coherent Restored Video (val_output)]
```

### Pillar 1: Broadcast Chroma Bleed Pre-Filtering
Broadcast compression (H.264/H.265) uses 4:2:0 chroma subsampling, halving color resolution. In anime, vibrant reds and blues bleed across thin black line contours. We isolate the luma channel ($Y$) for line art guidance while reconstructing high-fidelity chroma planes ($U, V$) prior to deep feature extraction.

### Pillar 2: Domain Adaptation on Re-Anime600
Because Re-Anime600 has no ground-truth paired degradation, we build a domain-specific adaptation loop:
* Extract pristine reference patches from the highest-scoring training clips.
* Model a broadcast-specific degradation simulator: H.264 compression at varying bitrates, chroma subsampling, and frame-rate cadence drops.
* Fine-tune a lightweight residual adapter or LoRA weights attached to the RRDB backbone using composite NR-IQA surrogate losses.

### Pillar 3: Anime Motion-Masked Temporal Consistency
Unlike natural videos where every pixel moves with camera motion, anime features a unique duality:
* **Static Painted Backgrounds:** Zero legitimate motion across dozens of frames.
* **Stepped Character Animation:** Characters animate on 2s or 3s (12 fps or 8 fps) against 24 fps backgrounds.
* **Our Method:** We construct motion-difference masks. Static background pixels are locked to temporal history frames ($I_{t-1}, I_t, I_{t+1}$), completely eliminating background line jitter. Dynamic character pixels propagate restored high-frequency contours along motion vectors without motion blur.

---

## 📝 Design Q&A (Architectural Decision Records)

### Q1: Since we downloaded pretrained weights, does that mean we won't train from scratch, only fine-tune?
* **Answer:** Yes. Training a high-capacity anime restoration model from scratch requires millions of paired clean/degraded patches and weeks of multi-GPU compute to converge on low-level edge detection. Re-Anime600 contains unpaired broadcast clips, making supervised scratch training ill-posed. Pretrained weights provide the necessary prior on what clean anime line art looks like; fine-tuning adapts that prior to Re-Anime600's specific broadcast compression quirks.

### Q2: If we are using downloaded models like APISR, what is our contribution?
* **Answer:** Pretrained APISR is an image model. Our contribution is bridging the gap from single-image processing to temporally coherent, broadcast-adapted video restoration. Specifically: (1) eliminating video line shimmer via anime-specific motion masks, (2) adapting the model to H.264 broadcast degradation and 4:2:0 chroma bleed, and (3) edge-guided frequency fusion to balance all four NR-IQA competition metrics.

---

## 📄 Proposed WACV 2027 Workshop Paper Narrative
* **Title Proposal:** *Temporally-Consistent Anime Video Enhancement with Motion-Masked Prior Adaptation on Re-Anime600*
* **Target:** 5–8 pages, OpenReview, Archival IEEE Xplore proceedings.
* **Ablation Matrix:**
  1. Baseline (FBCNN)
  2. Zero-Shot Single-Frame (APISR)
  3. APISR + Broadcast Domain Adaptation
  4. APISR + Domain Adaptation + Anime Temporal Motion Filter (Full Proposed System)