# AINIME @ WACV 2027 Challenge: Official Submission Guide

This document details the certified submission package for the **AINIME Anime Video Restoration Challenge (WACV 2027 Workshop on Anime Computer Vision)** evaluated on the **Re-Anime600** benchmark.

---

## 1. Submission Package Metadata

- **Archive File:** `submissions/submission.zip`
- **Archive Size:** 15.87 MB (compressed) / 17.16 MB (uncompressed)
- **SHA-256 Checksum:** `b20f096d4559f63329f7ba11af8ef33a05c60649a886f13314446fc877630810`
- **Primary Pipeline:** `EXP-03` (Adversarial APISR GAN Prior + Luma-Guided Chroma Bleed Filter + Motion-Gated Temporal Consistency Filter)

---

## 2. Bundled File Manifest

The ZIP bundle contains strictly the 8 required offline execution components:

```
submission.zip
├── inference.py                          # Official zero-argument runner
├── apisr_arch.py                         # 6-block RRDBNet neural architecture
├── fbcnn_arch.py                         # Baseline reference architecture
├── requirements.txt                      # Minimum package dependencies
├── src/
│   ├── __init__.py                       # Package initialization
│   ├── chroma_filter.py                  # Stage 1: Luma-guided chroma de-bleeding
│   └── temporal_filter.py                # Stage 3: Motion-gated temporal recurrence
└── model_zoo/
    └── 2x_APISR_RRDB_GAN_generator.pth  # 17.13 MB pretrained adversarial anime prior
```

---

## 3. Competition Evaluation Protocol & Rules Audit

The submission has been audited in a clean, isolated mock sandbox (`verify_submission.py`) against all 6 competition rules:

| Rule | Requirement | Sandbox Audit Result |
| :--- | :--- | :--- |
| **Rule 1** | Zero-argument execution (`python inference.py`) | **PASSED** (exit code 0, completely offline) |
| **Rule 2** | Output directory structure: `val_output/<clip_id>/` | **PASSED** (exact hierarchy preserved) |
| **Rule 3** | Exact frame count parity | **PASSED** (1:1 input to output frame count) |
| **Rule 4** | Exact dimensional parity | **PASSED** (input resolution = output resolution) |
| **Rule 5** | Strict frame naming convention | **PASSED** (`%06d.png` starting at `000000.png`) |
| **Rule 6** | Image format integrity | **PASSED** (lossless uncompressed PNG) |

---

## 4. Hardware & Resource Profile

Audited on an NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) running PyTorch 2.6 with CUDA 12.4:

- **Peak VRAM Consumption:** 1,304.2 MB (well under the 8,192 MB competition ceiling)
- **Runtime Throughput:** 2.87 to 4.2 fps (0.24 to 0.35 s/frame in FP16 mixed precision)
- **Network Access:** Zero external socket or HTTP calls; operates 100% offline

---

## 5. Verification Commands

To reproduce the submission bundle and verify sandbox compliance locally:

```bash
# 1. Package the zip bundle
python package_submission.py

# 2. Execute the isolated mock sandbox evaluation
python verify_submission.py
```
