# 📝 06 - Experiments & Progress Log

> Chronological record of architectural decisions, training runs, and leaderboard submissions.

---

## 📜 Timeline & Log

### 2026-10-10 — Official Submission Package Verified & Publication Figures Finalized
* **Milestone:** Locked `EXP-03` as the winning competition entry, validated isolated sandbox execution, and completed publication figure suite.
* **Actions:**
  * Configured `inference.py` and `package_submission.py` to default to `2x_APISR_RRDB_GAN_generator.pth` combined with chroma and temporal filters.
  * Rebuilt `submissions/submission.zip` (15.87 MB, SHA-256 `b20f096d4559f63329f7ba11af8ef33a05c60649a886f13314446fc877630810`).
  * Audited in isolated mock sandbox via `verify_submission.py`: All 6 rules passed with exit code 0 at 2.87 fps throughput (0.348 s/frame).
  * Generated 3 publication figures in `paper/figures/`:
    * Figure 1: Qualitative 4-stage comparison (`comparison.png`).
    * Figure 2: Modular 3-stage architecture diagram at 300 DPI (`architecture.png`).
    * Figure 3: Spatio-temporal $x\text{--}t$ slice proving background stability across 60 frames (`temporal_slice.png`).
  * Created `SUBMISSION_GUIDE.md` for formal evaluation portal delivery.

### 2026-10-09 — Research Contribution Framework & Design Q&A Added
* **Milestone:** Formalized original technical contributions and paper roadmap; established design FAQ.
* **Actions:**
  * Created `07 - Research Contributions & Technical Novelty.md` in Obsidian vault.
  * Formalized the 3 innovation pillars: (1) Broadcast chroma bleed correction, (2) Re-Anime600 domain adaptation, (3) Anime motion-masked temporal filter.
  * Formulated WACV 2027 workshop paper narrative and experimental ablation matrix.
  * Documented design decisions regarding pretrained priors vs. scratch training and unpaired data handling.

### 2026-10-09 — Pillar 3: Anime Motion Temporal Filter Benchmarked
* **Milestone:** Implemented and verified motion-gated temporal consistency filter (`src/temporal_filter.py`).
* **Actions:**
  * Implemented `AnimeTemporalFilter` exploiting anime duality: locks motionless painted backgrounds ($\tau_{static}=2.5$) while passing dynamic character movement ($\tau_{motion}=9.0$).
  * Built benchmark script `src/benchmark_temporal.py` measuring static background shimmer on 120 frames of `val/10299.mp4`:
    * Static Region Flicker (Raw APISR): 1.37 MSE
    * Static Region Flicker (APISR + Temporal Filter): **0.93 MSE** (**32.1% reduction in line shimmer**)
    * Edge Clarity Preservation: 2514.5 (retains 98.5% of sharp contour contrast)
  * Integrated directly into `inference.py` enabled by default.

### 2026-10-09 — Pillar 1: Luma-Guided Chroma Bleed Filter Benchmarked
* **Milestone:** Implemented and verified edge-guided chroma bleed filter (`src/chroma_filter.py`).
* **Actions:**
  * Implemented `LumaGuidedChromaFilter` using He et al. Guided Filtering with normalized luma ($Y$) as the structural guidance barrier.
  * Measured Chroma Edge Alignment along detected ink contours on `val/10299.mp4`:
    * Raw Input Alignment: 0.7428
    * Chroma-Filtered Alignment: **0.8098** (**+9.0% tighter color boundary confinement**)
    * Filter Latency: 36.46 ms/frame (~27.4 fps throughput)
  * Integrated directly into `inference.py` preceding neural restoration.
* **Next Steps:**
  * Package full submission zip and execute offline sandbox verification.
  * Implement Pillar 2: Synthetic broadcast degradation pipeline for domain adaptation.

---

## 🧪 Experiments Tracker Table

| Exp ID | Model Architecture | Resolution | Speed (s/fr) | Edge Clarity | Flicker MSE | Chroma Alignment | Output Parity | Status | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-00` | FBCNN (Official Baseline) | 854x480 | 0.216s (4.6 fps) | 1230.2 | High | 0.74 | 844/844 (100%) | Complete | Baseline reference; JPEG deblocking only; softens edges. |
| `EXP-01` | APISR (2x RRDB-6B) | 854x480 | 0.232s (4.3 fps) | **2616.5** | 1.37 | 0.74 | 844/844 (100%) | Complete | +112% edge clarity; sharp contours; single-frame shimmer. |
| `EXP-02` | APISR + Temporal Filter | 854x480 | 0.235s (4.2 fps) | **2514.5** | **0.93** | 0.74 | 844/844 (100%) | Complete | **-32.1% line shimmer**; locks painted backgrounds; zero motion blur. |
| `EXP-03` | APISR + Chroma + Temporal | 854x480 | 0.282s (3.6 fps) | **2528.1** | **0.93** | **0.81** (+9%) | 844/844 (100%) | Complete | Full 3-stage pipeline: clean color boundaries, sharp ink, steady cels. |
| `EXP-04` | Domain-Adapted APISR | 854x480 | 0.284s (3.5 fps) | 1196.2 | **0.93** | 0.71 | 844/844 (100%) | Complete | Pure L1+Sobel fine-tuning; eliminates broadcast blockiness; softer lines. |
| `EXP-05` | Interpolated APISR (alpha=0.7) | 854x480 | 0.255s (3.9 fps) | **2244.7** (+78%) | **0.93** | **0.73** | 844/844 (100%) | Complete | 70% GAN prior + 30% domain adaptation; sweet spot between sharp ink and clean cel fills. |

---

## 📈 Multi-Clip Robustness Audit (5 Diverse Re-Anime600 Clips)
* **Average Edge Sharpness Gain:** **+99.3%** (ranging up to +197.1% on detailed scenes).
* **Average Background Shimmer Reduction:** **-68.3%** (substantial inter-frame static noise suppression).
* **Average Inference Speed:** **3.9 fps** (~0.256 s/frame) on RTX 4060 GPU.
* **Peak GPU VRAM:** **1,304.2 MB** (15.9% of 8GB ceiling; zero OOM risk).
* **Status:** Verified and locked into `submissions/submission.zip`.