# 📋 01 - Challenge Specifications & Constraints

> **Workshop:** AINIME (AI for Animation · 1st Edition)  
> **Conference:** WACV 2027 (Disney Springs, Florida · Jan 4–5, 2027)  
> **Official Hub:** [https://mv-lab.github.io/ainime/](https://mv-lab.github.io/ainime/)

---

## 🎯 The Objective
Enhance the **perceptual quality** of anime video clips from **Re-Anime600** while strictly preserving:
1. **Original Content & Composition:** Character identity, line fidelity, line width, and background art.
2. **Artistic Style:** Flat cel-shading, characteristic color palettes, and anime drawing conventions.
3. **Temporal Consistency:** Zero flickering, temporal coherence across moving camera pans and character movements.

---

## 📊 Dataset Structure: Re-Anime600
Curated from the Sakuga-42M anime corpus and filtered using four NR-IQA quality metrics.

| Split | Count | Availability | Notes |
| :--- | :--- | :--- | :--- |
| **Train** | 350 clips | Public on Hugging Face | Requires signed data agreement (`ainime-challenge/re-anime600-train`). |
| **Validation** | 100 clips | Validation phase | Used for leaderboard validation runs. |
| **Test** | 150 clips | Hidden | Used for final competition scoring and prize awards. |

* **Clip properties:** MP4 video containers or image sequences, 23.98 fps, variable standard anime broadcast resolutions.

---

## ⏱️ Critical Timeline
* **Oct 2, 2026:** Challenge opens & train data released.
* **Oct 2 – Oct 20, 2026:** Validation Phase.
* **Oct 20, 2026 (AoE):** Challenge Paper Submission Deadline (5–8 pages, OpenReview, archival proceedings in IEEE Xplore).
* **Oct 21 – Oct 25, 2026:** Test Phase.
* **Oct 28, 2026:** Final winners & results announced.
* **Jan 4–5, 2027:** Workshop session at WACV Disney Springs.

---

## 🔒 Submission Constraints & Sandboxed Environment
Submissions are evaluated in a strict offline sandbox. Violating any of these results in immediate failure:

1. **Packaging:** Single `.zip` containing:
   * `inference.py` (entry point)
   * Model architecture definitions (e.g. `model_arch.py`)
   * Checkpoint files (e.g. `model_zoo/checkpoint.pth`)
   * Pinned `requirements.txt`
2. **Zero Internet Access:** The evaluation runner executes `python inference.py` with no flags and disconnected from the internet. Pretrained weights must be bundled inside the zip.
3. **I/O Folder Contract:**
   * Input: `./val/` folder placed next to `inference.py` containing input video files or frame subdirectories.
   * Output: `./val_output/<clip_id>/000000.png, 000001.png, ...` (lossless PNG frames).
4. **Dimension & Timing Preservation:** Output frames **must exactly match** input resolution, frame count, and frame rate.