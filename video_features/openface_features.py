import argparse
import subprocess

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Sequence


def find_videos(root: Path, suffixes: Sequence[str]) -> list[Path]:
    return [path for path in root.glob("**/*") if path.suffix.lower() in suffixes and path.is_file()]


def run_feature_extraction(
    video_path: Path,
    executable: Path,
    output_dir: Path,
) -> Path:

    if Path(output_dir / video_path.with_suffix('.csv').name).exists():
        return video_path

    command = [
        str(executable),
        "-f",
        str(video_path),
        "-gaze",
        "-aus",
        "-pose",
        "-au_static",
        "-out_dir",
        str(output_dir),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"FeatureExtraction failed for {video_path} (exit {result.returncode}):\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return video_path


def process_videos(
    videos: Iterable[Path],
    executable: Path,
    output_root: Path,
    max_workers: int,
) -> None:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_video = {
            executor.submit(run_feature_extraction, video, executable, output_root): video
            for video in videos
        }
        for future in as_completed(future_to_video):
            video = future_to_video[future]
            future.result()
            print(f"Processed {video}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, default=Path("../data/video"))
    parser.add_argument("--output-root", type=Path, default=Path("../data/video_features/csv"))
    parser.add_argument(
        "--feature-extraction",
        type=Path,
        default=Path("FeatureExtraction"),
        help="Path to OpenFace FeatureExtraction executable",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--suffixes",
        nargs="*",
        default=[".mp4", ".avi", ".mov", ".mkv"],
        help="Video file suffixes to include",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    videos = find_videos(args.video_root, tuple(args.suffixes))

    if not videos:
        raise SystemExit(f"No videos found in {args.video_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    process_videos(videos, args.feature_extraction, args.output_root, args.max_workers)


if __name__ == "__main__":
    main()
# ...existing code...