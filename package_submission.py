"""
Packaging Script for AINIME @ WACV 2027 Challenge Submission.

Builds submissions/submission.zip matching the official competition format:
- inference.py (entry point)
- apisr_arch.py, fbcnn_arch.py
- src/ (__init__.py, chroma_filter.py, temporal_filter.py)
- model_zoo/2x_APISR_RRDB_GAN_generator.pth
- requirements.txt
"""

import os
import sys
import zipfile
import hashlib

REQUIRED_FILES = [
    "inference.py",
    "apisr_arch.py",
    "fbcnn_arch.py",
    "requirements.txt",
    os.path.join("src", "__init__.py"),
    os.path.join("src", "chroma_filter.py"),
    os.path.join("src", "temporal_filter.py"),
    os.path.join("model_zoo", "apisr_reanime600_interpolated.pth"),
]

def calculate_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()

def package_submission(output_dir: str = "submissions", zip_name: str = "submission.zip") -> str:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(repo_root, output_dir), exist_ok=True)
    zip_path = os.path.join(repo_root, output_dir, zip_name)

    print("=" * 60)
    print("AINIME Challenge — Packaging Official Submission Bundle")
    print("=" * 60)

    # 1. Verify existence of all required files
    missing = []
    total_bytes = 0
    for rel_path in REQUIRED_FILES:
        full_path = os.path.join(repo_root, rel_path)
        if not os.path.isfile(full_path):
            missing.append(rel_path)
        else:
            total_bytes += os.path.getsize(full_path)

    if missing:
        print("\n[ERROR] Missing required submission files:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print(f"Found all {len(REQUIRED_FILES)} required components ({total_bytes / (1024 * 1024):.2f} MB uncompressed).")

    # 2. Build ZIP archive
    print(f"\nCompressing into {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel_path in REQUIRED_FILES:
            full_path = os.path.join(repo_root, rel_path)
            # Store in zip using normalized forward-slash relative paths
            arcname = rel_path.replace("\\", "/")
            zf.write(full_path, arcname=arcname)
            file_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  + Added: {arcname:<45} ({file_mb:.2f} MB)")

    # 3. Verify archive integrity
    with zipfile.ZipFile(zip_path, "r") as zf:
        test_res = zf.testzip()
        if test_res is not None:
            print(f"\n[ERROR] Archive corrupted at member: {test_res}")
            sys.exit(1)
        namelist = zf.namelist()

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    sha256 = calculate_sha256(zip_path)

    print("\n" + "=" * 60)
    print("Submission Bundle Created Successfully!")
    print(f"Destination:  {zip_path}")
    print(f"Archive Size: {zip_size_mb:.2f} MB")
    print(f"Files Count:  {len(namelist)} items")
    print(f"SHA-256:      {sha256}")
    print("=" * 60)

    return zip_path

if __name__ == "__main__":
    package_submission()
