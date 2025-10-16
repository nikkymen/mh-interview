import docker
import docker.errors
import multiprocessing

from typing import List, Tuple
from pathlib import Path

# --- Configuration ---
VIDEO_DIR = "../data/video"
OUTPUT_DIR = "output/csv"
DOCKER_IMAGE = "nikkymen/openface"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
# Adjust the number of parallel processes based on your CPU cores and memory
NUM_PROCESSES = multiprocessing.cpu_count() // 2 or 1

# -------------------

def get_video_files(directory: str) -> List[Path]:
    """Finds all video files in a directory."""
    path = Path(directory)
    if not path.is_dir():
        print(f"Error: Input directory not found at '{directory}'")
        return []

    video_files = [
        f for f in path.rglob('*')
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    ]
    print(f"Found {len(video_files)} video files in '{directory}'.")
    return video_files

def process_video(args: Tuple[Path, Path, Path]):
    """
    Runs OpenFace feature extraction in a Docker container for a single video file.
    """
    video_file, base_input_dir, base_output_dir = args

    # We mount the parent directories and use paths relative to the container mounts
    # Host Path: /home/user/project/data/video/my_vid.mp4
    # Mount: /home/user/project/data -> /data
    # Path inside container: /data/video/my_vid.mp4

    container_input_dir = "/data_in"
    container_output_dir = "/data_out"

    relative_video_path = video_file.relative_to(base_input_dir)

    # Construct paths as they will be inside the container
    container_video_path = Path(container_input_dir) / relative_video_path

    command = [
        "FeatureExtraction",
        "-f", str(container_video_path),
        "-gaze",
        "-aus",
        "-pose",
        "-au_static",
        "-out_dir", container_output_dir
    ]

    try:
        client = docker.from_env()

        # Ensure the output directory for this specific file exists on the host
        # OpenFace will create a file with the video's basename inside this dir
        host_output_path = base_output_dir / relative_video_path.parent
        host_output_path.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {video_file.name}")

        container = client.containers.run(
            image=DOCKER_IMAGE,
            command=command,
            volumes={
                str(base_input_dir.resolve()): {'bind': str(container_input_dir), 'mode': 'ro'},
                str(base_output_dir.resolve()): {'bind': str(container_output_dir), 'mode': 'rw'}
            },
            remove=True,  # Automatically remove the container when it exits
            detach=False, # Run in foreground and wait for completion
            stdout=True,
            stderr=True
        #    log_config={'type': 'none'} # Suppress container logs to stdout
        )

        print(f"Finished: {video_file.name}")
        return None
    except docker.errors.ImageNotFound:
        return f"Error: Docker image '{DOCKER_IMAGE}' not found. Please pull it first: `docker pull {DOCKER_IMAGE}`"
    except Exception as e:
        return f"Error processing {video_file.name}: {e}"

def main():
    """Main function to set up and run the parallel processing."""
    base_input_path = Path(VIDEO_DIR)
    base_output_path = Path(OUTPUT_DIR)

    video_files = get_video_files(str(base_input_path))
    if not video_files:
        return

    video_files = [video_files[0]]

    # Prepare arguments for the worker pool
    tasks = [(video_file, base_input_path, base_output_path) for video_file in video_files]

    process_video(tasks[0])

    # print(f"\nStarting feature extraction with {NUM_PROCESSES} parallel processes...")

    # with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
    #     results = pool.map(process_video, tasks)

    # print("\n--- Processing Complete ---")
    # errors = [res for res in results if res is not None]
    # if errors:
    #     print("The following errors occurred:")
    #     for error in errors:
    #         print(f"- {error}")
    # else:
    #     print("All videos processed successfully.")
    # print(f"Output features are located in: '{base_output_path.resolve()}'")


if __name__ == "__main__":
    main()