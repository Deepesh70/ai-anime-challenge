"""Dataset downloader for the AINIME @ WACV 2027 Challenge (Re-Anime600).

Downloads the 350 training MP4 clips into ./data/train/.
Usage:
    python download_dataset.py
"""
import os
import sys
from huggingface_hub import snapshot_download

REPO_ID = "ainime-challenge/re-anime600-train"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train")
def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Note: HF_TOKEN environment variable not set. For gated datasets, set HF_TOKEN=<your_token>.")
    print(f"Connecting to Hugging Face dataset: {REPO_ID}")
    print(f"Destination folder: {LOCAL_DIR}")

    os.makedirs(LOCAL_DIR, exist_ok=True)

    try:
        download_path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DIR,
            token=token,
            allow_patterns=["train/*.mp4", "*.md", "*.json"],
            max_workers=4,
        )
        print("\nDownload finished successfully.")
        
        # Verify file count
        train_folder = os.path.join(LOCAL_DIR, "train") if os.path.isdir(os.path.join(LOCAL_DIR, "train")) else LOCAL_DIR
        mp4_files = [f for f in os.listdir(train_folder) if f.lower().endswith(".mp4")]
        print(f"Total training clips found: {len(mp4_files)} / 350")
    except Exception as e:
        print(f"\nDownload failed with error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
