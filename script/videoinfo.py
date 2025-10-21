import argparse
from pathlib import Path

import cv2

def iter_video_files(root: Path):
    video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".flv", ".wmv"}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in video_extensions:
            yield path


def print_video_fps(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"{video_path}: unable to open video")
        return
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if fps:
        print(f"{video_path}: {fps:.2f} fps")
    else:
        print(f"{video_path}: fps unavailable")


def main():
    parser = argparse.ArgumentParser(description="Print frame rates for videos in a directory.")
    parser.add_argument(
        "--input",
        type=str,
        default="../data/video",
        help="Path to directory containing video files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input path '{input_dir}' is not a directory.")
        return

    found_any = False
    for video_file in iter_video_files(input_dir):
        found_any = True
        print_video_fps(video_file)

    if not found_any:
        print(f"No video files found in '{input_dir}'.")


if __name__ == "__main__":
    main()