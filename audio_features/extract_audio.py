import argparse
import subprocess
import multiprocessing

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

def extract_wav_from_video(input_file: Path, output_dir: Path) -> Tuple[Path, Path]:
    """
    Extract raw and normalized WAV files from a video file.

    Args:
        input_file: Path to the input video file
        output_dir: Path to the output directory for WAV files
    """
    name = input_file.stem

    raw_wav  = output_dir / "raw"  / f"{name}.wav"
    norm_wav = output_dir / "norm" / f"{name}.wav"

    # Skip if both WAV files already exist
    if raw_wav.exists() and norm_wav.exists():
        return raw_wav, norm_wav

    raw_wav.parent.mkdir(parents=True, exist_ok=True)
    norm_wav.parent.mkdir(parents=True, exist_ok=True)

    # Extract raw WAV using ffmpeg
    if not raw_wav.exists():
        subprocess.run([
            "ffmpeg",
            "-i", str(input_file),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            str(raw_wav)
        ], check=True, capture_output=True)

    # Normalize and compress using sox
    if not norm_wav.exists():
        subprocess.run([
            "sox",
            str(raw_wav),
            "-r", "16k",
            str(norm_wav),
            "norm", "-0.5",
            "compand", "0.3,1", "-90,-90,-70,-70,-60,-20,0,0", "-5", "0", "0.2"
        ], check=True, capture_output=True)

    return raw_wav, norm_wav

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/video/"))
    parser.add_argument("--output", type=Path, default=Path("data/audio/"))

    args = parser.parse_args()

    (args.output / "raw").mkdir(parents=True, exist_ok=True)
    (args.output / "norm").mkdir(parents=True, exist_ok=True)

    # Check if input directory exists
    if not args.input.exists():
        print(f"Error: Input path {args.input} does not exist")
        return

    # Process all video files in the input directory
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}

    if args.input.is_file():
        if args.input.suffix.lower() not in video_extensions:
            print(f"Error: File {args.input} has unsupported extension.")
            return
        video_files = [args.input]
    else:
        # Process all video files in the input directory
        video_files = [f for f in args.input.iterdir()
                       if f.is_file() and f.suffix.lower() in video_extensions]

    if not video_files:
        print(f"No video files found in {args.input}")
        return

    print(f"Found {len(video_files)} video file(s) to process")

    # Determine number of workers (use CPU count, but cap at number of files)
    max_workers = min(multiprocessing.cpu_count(), len(video_files))
    print(f"Processing with {max_workers} parallel workers...\n")

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(extract_wav_from_video, video_file, args.output): video_file
            for video_file in video_files
        }

        # Process results as they complete
        for future in as_completed(futures):
            video_file = futures[future]
            try:
                result = future.result()
                print(result)
            except subprocess.CalledProcessError as e:
                print(f"Error processing {video_file.name}: {e}")
            except Exception as e:
                print(f"Unexpected error processing {video_file.name}: {e}")

    print("\nAll files processed!")

if __name__ == "__main__":
    main()