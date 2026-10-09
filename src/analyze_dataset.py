"""Dataset profiler for Re-Anime600 training clips."""
import glob
from collections import Counter
import cv2

def profile():
    clips = glob.glob("data/train/**/*.mp4", recursive=True)
    print(f"Total clips analyzed: {len(clips)}")
    
    resolutions = Counter()
    frame_counts = []
    fps_list = []

    for path in clips:
        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = round(cap.get(cv2.CAP_PROP_FPS), 2)
        
        resolutions[(w, h)] += 1
        frame_counts.append(frames)
        fps_list.append(fps)
        cap.release()

    print("\nResolution breakdown (Width x Height: Count):")
    for (w, h), count in resolutions.most_common():
        print(f"  {w}x{h}: {count} clips ({count / len(clips) * 100:.1f}%)")

    print(f"\nFrame counts: min={min(frame_counts)}, max={max(frame_counts)}, mean={sum(frame_counts)/len(frame_counts):.1f}")
    print(f"Total frames across all clips: {sum(frame_counts)}")
    print(f"Common FPS values: {Counter(fps_list).most_common(5)}")

if __name__ == "__main__":
    profile()
