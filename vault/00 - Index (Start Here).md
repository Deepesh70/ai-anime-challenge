# 🎬 AINIME Challenge Hub (WACV 2027) — Knowledge Vault

> **Mission:** Competitive repository and technical documentation for the **AINIME Anime Restoration Challenge** at WACV 2027 on the **Re-Anime600** benchmark.

---

## 🗺️ Map of Content (MOC)

This knowledge base tracks the end-to-end lifecycle of the challenge: from benchmark analysis and metric optimization to model engineering, submission packaging, and workshop paper preparation.

### 🏛️ Architecture & Rules
* [[01 - Challenge Specifications & Constraints]]: Official rules, dataset splits, offline sandbox execution, and strict submission constraints.
* [[02 - Evaluation Metrics & NR-IQA]]: Deep dive into NIQE, MANIQA, CLIPIQA, and HyperIQA, plus temporal stability.
* [[03 - System Architecture & Pipeline]]: The end-to-end restoration pipeline, tiled inference, frame I/O, and hardware budgeting.
* [[04 - Model Candidates & Baselines]]: Comparison of APISR, AnimeSR, Real-ESRGAN, and the baseline FBCNN.

### 💡 Research & Novelty
* [[07 - Research Contributions & Technical Novelty]]: Core research differentiation, solving single-image video flickering, domain adaptation, and WACV 2027 paper narrative.
* [[08 - WACV 2027 Workshop Paper Draft]]: Complete formal methodology draft, LaTeX formulations, and experimental ablation tables.

### 🛠️ Execution & Operations
* [[05 - Agent Guide & Workspace Cheatsheet]]: Workspace directory map, hardware constraints (RTX 4060 8GB), and agent rules.
* [[06 - Experiments & Progress Log]]: Live experiment log, benchmark scores, iteration history, and submission status.

---

## 📌 Current Project Status
| Milestone | Status | Detail |
| :--- | :--- | :--- |
| **Workshop Registration** | Done | Registered via official Google Form. |
| **Dataset Download** | Complete | All 350 training clips (3.24 GB) downloaded to `data/train/train/`. |
| **Dataset Profiling** | Complete | 337,205 frames; 98% 480p height (854x480, 852x480, 888x480), 23.98 fps. |
| **Conda Environment (`aianime`)** | Active | Python 3.10 + PyTorch 2.6 CUDA 12.4 + OpenCV + HF Hub. |
| **Baseline Sandbox Verification** | Verified | FBCNN: 844 frames, 0.216 s/frame, edge score 1230.2. |
| **Candidate 1 (APISR 2x)** | Verified | APISR: 844 frames, 0.232 s/frame, edge score 2616.5 (+112% sharpness). |
| **Pillar 3: Temporal Filter** | Verified | 32.1% shimmer reduction on static background cels. |
| **Pillar 1: Chroma Bleed Filter** | Verified | Luma-guided guided filter: +9.0% tighter contour alignment. |
| **End-to-End 3-Stage Pipeline** | Active | `inference.py` running full Chroma + APISR + Temporal stack. |
| **Open Source Repository Setup** | Complete | Standard layout: `.gitignore`, `LICENSE`, `README.md`, `pyproject.toml`, CI workflow. |
| **Submission Packaging & Verification** | Complete | `submission.zip` built (15.87 MB) and 100% verified in isolated mock sandbox. |
| **Domain Adaptation (Pillar 2)** | Complete | Broadcast degradation simulator + differentiable Sobel edge loss fine-tuning. |
| **Comprehensive Benchmark & Ablation** | Up Next | Full quantitative evaluation against official baseline on all validation clips. |