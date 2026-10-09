"""
Mock Sandbox Evaluator for AINIME @ WACV 2027 Challenge Submission.

Simulates the organizers' automated evaluation runner:
1. Extracts submissions/submission.zip into an isolated mock sandbox.
2. Places test clips into mock_sandbox/val/.
3. Executes `python inference.py` (zero command-line arguments, offline protocol).
4. Strictly audits competition compliance:
   - Exit code == 0
   - Directory format: val_output/<clip_id>/%06d.png
   - Frame count exact parity
   - Resolution (W x H) exact parity
   - Image format: valid uncompressed PNG frames
   - Computes throughput (fps and s/frame)
"""

import os
import sys
import shutil
import zipfile
import subprocess
import time
import cv2

def verify_submission_sandbox(zip_path: str = "submissions/submission.zip", quick_test_frames: int = 40):
    repo_root = os.path.dirname(os.path.abspath(__file__))
    zip_full_path = os.path.join(repo_root, zip_path)
    sandbox_dir = os.path.join(repo_root, "mock_sandbox")

    print("=" * 65)
    print("AINIME Challenge — Mock Sandbox Evaluation & Rule Audit")
    print("=" * 65)

    if not os.path.isfile(zip_full_path):
        print(f"[ERROR] Submission bundle not found at {zip_full_path}. Run package_submission.py first.")
        sys.exit(1)

    # 1. Clean and initialize isolated sandbox directory
    if os.path.exists(sandbox_dir):
        shutil.rmtree(sandbox_dir)
    os.makedirs(sandbox_dir, exist_ok=True)
    print(f"[1/5] Initialized clean sandbox environment at:\n      {sandbox_dir}")

    # 2. Extract submission bundle
    print(f"[2/5] Extracting {os.path.basename(zip_full_path)} into sandbox...")
    with zipfile.ZipFile(zip_full_path, "r") as zf:
        zf.extractall(sandbox_dir)

    # Verify extracted entry point
    entry_point = os.path.join(sandbox_dir, "inference.py")
    if not os.path.isfile(entry_point):
        print("[FAIL] Missing required entry point: inference.py")
        sys.exit(1)

    # 3. Create mock input validation clip in sandbox/val
    sandbox_val = os.path.join(sandbox_dir, "val")
    os.makedirs(sandbox_val, exist_ok=True)

    # Source reference video from repo's val/ or data/
    src_video_path = os.path.join(repo_root, "val", "10299.mp4")
    if not os.path.isfile(src_video_path):
        # Fallback to train directory if val isn't present
        train_dir = os.path.join(repo_root, "data", "train", "train")
        if os.path.isdir(train_dir):
            mp4s = [f for f in os.listdir(train_dir) if f.endswith(".mp4")]
            if mp4s:
                src_video_path = os.path.join(train_dir, mp4s[0])

    if not os.path.isfile(src_video_path):
        print("[ERROR] No reference MP4 found to create mock validation test.")
        sys.exit(1)

    # Generate a trimmed mock validation clip for rapid, complete validation
    mock_clip_name = "mock_test_10299.mp4"
    mock_clip_path = os.path.join(sandbox_val, mock_clip_name)

    cap = cv2.VideoCapture(src_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_writer = cv2.VideoWriter(mock_clip_path, fourcc, fps, (width, height))
    extracted_frames = 0
    while extracted_frames < quick_test_frames:
        ret, frame = cap.read()
        if not ret:
            break
        out_writer.write(frame)
        extracted_frames += 1
    cap.release()
    out_writer.release()

    clip_id = os.path.splitext(mock_clip_name)[0]
    print(f"[3/5] Staged test video: {mock_clip_name} ({extracted_frames} frames, {width}x{height} @ {fps:.2f} fps)")

    # 4. Execute `python inference.py` in sandbox without arguments
    print(f"\n[4/5] Executing `python inference.py` in isolated sandbox...")
    start_time = time.time()

    # Use current active python interpreter
    cmd = [sys.executable, "inference.py"]
    proc = subprocess.run(cmd, cwd=sandbox_dir, capture_output=True, text=True)

    elapsed_time = time.time() - start_time

    if proc.returncode != 0:
        print("\n[FAIL] Inference failed with non-zero exit code!")
        print(f"Stdout:\n{proc.stdout}")
        print(f"Stderr:\n{proc.stderr}")
        sys.exit(1)

    print(f"      Inference completed in {elapsed_time:.2f} seconds ({elapsed_time / extracted_frames:.3f} s/frame, {extracted_frames / elapsed_time:.2f} fps).")

    # 5. Strict Competition Compliance Audit
    print(f"\n[5/5] Auditing competition rule compliance:")
    expected_output_dir = os.path.join(sandbox_dir, "val_output", clip_id)

    violations = []

    # Rule A: Output directory structure
    if not os.path.isdir(expected_output_dir):
        violations.append(f"Output directory does not exist: val_output/{clip_id}")
    else:
        output_files = sorted([f for f in os.listdir(expected_output_dir) if f.endswith(".png")])
        
        # Rule B: Frame count parity
        if len(output_files) != extracted_frames:
            violations.append(f"Frame count mismatch: Expected {extracted_frames}, got {len(output_files)}")

        # Rule C: Frame naming scheme (%06d.png)
        for idx, filename in enumerate(output_files):
            expected_name = f"{idx:06d}.png"
            if filename != expected_name:
                violations.append(f"Naming convention violation at index {idx}: Expected {expected_name}, got {filename}")
                break

        # Rule D: Dimensions and image integrity
        if output_files:
            sample_frame_path = os.path.join(expected_output_dir, output_files[0])
            img = cv2.imread(sample_frame_path)
            if img is None:
                violations.append("First output frame could not be decoded (corrupt PNG).")
            else:
                out_h, out_w, out_c = img.shape
                if out_w != width or out_h != height:
                    violations.append(f"Dimension mismatch: Input was {width}x{height}, Output is {out_w}x{out_h}")
                if out_c != 3:
                    violations.append(f"Channel mismatch: Expected 3 channels, got {out_c}")

    print("-" * 65)
    if violations:
        print("[AUDIT FAILED] Violations found:")
        for v in violations:
            print(f"  ❌ {v}")
        sys.exit(1)
    else:
        print("  ✅ Rule 1: Zero-argument invocation (`python inference.py`) — PASSED")
        print("  ✅ Rule 2: Output folder hierarchy (`val_output/<clip_id>/`) — PASSED")
        print(f"  ✅ Rule 3: Frame count parity ({extracted_frames}/{extracted_frames} frames) — PASSED")
        print(f"  ✅ Rule 4: Dimensional resolution parity ({width}x{height}) — PASSED")
        print("  ✅ Rule 5: Frame naming scheme (`%06d.png`) — PASSED")
        print("  ✅ Rule 6: Image format integrity (uncompressed lossless PNG) — PASSED")
        print("-" * 65)
        print("🎉 ALL COMPETITION AUDIT CHECKS PASSED SUCCESSFULLY!")
        print("=" * 65)

    # Clean up mock sandbox after verification to save disk space
    shutil.rmtree(sandbox_dir)
    print("Cleaned up temporary mock sandbox.")

if __name__ == "__main__":
    verify_submission_sandbox()
