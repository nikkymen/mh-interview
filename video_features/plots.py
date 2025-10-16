import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import concurrent.futures

from functools import partial
from typing import List

matplotlib.use('Agg') # Do not use TCL

COLUMNS = ['success',
           'gaze_angle_x', 'gaze_angle_y',
           'pose_Rx', 'pose_Ry', 'pose_Rz',
           'AU01_r', 'AU02_r', 'AU04_r',
           'AU05_r', 'AU06_r', 'AU07_r',
           'AU09_r', 'AU10_r', 'AU12_r',
           'AU14_r', 'AU15_r', 'AU17_r',
           'AU20_r', 'AU23_r', 'AU25_r',
           'AU26_r', 'AU45_r']


def load_dataframe(file_path: str) -> pd.DataFrame:
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.parquet', '.pq']:
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Normalize AU intensity columns by dividing by 5
    au_columns = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]
    for col in au_columns:
        df[col] = df[col] / 5.0

    df.reset_index(drop=True, inplace=True)
    return df


def plot_data(data: pd.DataFrame,
              video_id: str,
              output_dir: str,
              columns: List[str],
              crop_start: int,
              crop_end: int) -> bool:
    """
    Create both consolidated and individual column plots for a video

    Args:
        video_data: DataFrame containing the video data
        video_id: Video identifier
        output_dir: Directory to save plots
        columns: List of columns to plot

    Returns:
        bool: True if new plots were created, False if they already existed
    """
    consolidated_path = os.path.join(output_dir, f"{video_id}.png")
    created_consolidated = False

    if not os.path.exists(consolidated_path):
        num_cols = 6
        num_sublots = len(columns)
        num_rows = (num_sublots + num_cols - 1) // num_cols

        fig = plt.figure(figsize=(19.2, 10.8))

        for i, column in enumerate(columns):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            sns.lineplot(x='timestamp', y=column, data=data, linewidth=1.5, ax=ax)

            ax.set_xlabel('')
            ax.set_ylabel('')

            if column == 'success':
                ax.set_ylim(0, 2)
            elif column.startswith('AU'):
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(-0.6, 0.6)

            ax.set_title(column)
            ax.set_xlim(crop_start, crop_end)

        plt.suptitle(f'{video_id}', fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(consolidated_path, dpi=100)
        plt.close()
        created_consolidated = True

    # Create individual column plots
    video_dir = os.path.join(output_dir, video_id)
    created_individual = False

    if not os.path.exists(video_dir):
        os.makedirs(video_dir, exist_ok=True)

        # Plot each column separately
        for column in columns:
            plt.figure(figsize=(12, 6))

            # Create the line plot
            sns.lineplot(x='timestamp', y=column, data=data, linewidth=1.5)

            # Set y-axis limits based on column type
            if column == 'success':
                plt.ylim(0, 2)
            elif column.startswith('AU'):
                plt.ylim(0, 1)
            else:
                plt.ylim(-0.6, 0.6)

            # Add labels and title
            plt.xlabel('Time (seconds)')
            plt.ylabel('Intensity')
            plt.title(f'{column} - {video_id}')

            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(video_dir, f"{column}.png"), dpi=100)
            plt.close()

        created_individual = True

    return created_consolidated or created_individual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input/parquet')
    parser.add_argument('--output', type=str, default='output/plots')
    parser.add_argument('--crop-start', type=int, default=60)
    parser.add_argument('--crop-end', type=str, default=80)

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    data_files = glob.glob(os.path.join(args.input, "*"))

    if not data_files:
        raise ValueError(f"No data files found in {args.input}")

    data_dict = {}

    # Load dataframes

    for file_path in data_files:
        video_id = os.path.splitext(os.path.basename(file_path))[0]
        df = load_dataframe(file_path)
        data_dict[video_id] = df

    # Start plotting in parallel processes

    plot_func = partial(plot_data)
    max_workers = min(os.cpu_count() or 4, len(data_dict))

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [(df, video_id) for video_id, df in data_dict.items()]
        futures = {
            executor.submit(plot_func, data, vid_id, args.output, COLUMNS, args.crop_start, args.crop_end): vid_id
            for data, vid_id in tasks
        }

        for future in concurrent.futures.as_completed(futures):
            video_id = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")



if __name__ == "__main__":
    main()