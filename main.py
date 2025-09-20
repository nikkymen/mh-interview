import os
from pathlib import Path
from sys import base_prefix

os.environ["NUMBA_DISABLE_CUDA"] = "1"
os.environ["TCL_LIBRARY"] = str(Path(base_prefix) / "lib" / "tcl8.6")
os.environ["TK_LIBRARY"] = str(Path(base_prefix) / "lib" / "tk8.6")

#/home/nik/.local/share/uv/python/cpython-3.9.5-linux-x86_64-gnu/lib/tcl8.6/
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection import select_features
#from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import MinimalFCParameters

from typing import Dict, Optional, List

def main():

    parser = argparse.ArgumentParser(description='Facial Expression Analysis')

    parser.add_argument('--data', type=str, default='data/parquet', help='')
    parser.add_argument('--output', type=str, default='output', help='')
    parser.add_argument('--plots', action=argparse.BooleanOptionalAction, default=False, help='')
    parser.add_argument('--features', action=argparse.BooleanOptionalAction, default=True, help='')

    processor = DataProcessor(parser.parse_args())

    processor.process_videos()

def plot_single_video(video_data: pd.DataFrame,
                      video_id: str,
                      video_dir: str,
                      columns: List[str]):

    image_path = os.path.join(video_dir, f"{video_id}.png")

    if os.path.exists(image_path):
        return False  # Skip if already exists

    df = video_data

    # Get all AU columns with intensity values (_r suffix)
    #au_intensity_cols = [col for col in df.columns if col.endswith('_r')]

    num_cols = 6

    # Calculate grid dimensions
    num_sublots = len(columns)
    num_rows = (num_sublots + num_cols - 1) // num_cols  # Ceiling division

    # Create a figure with FullHD dimensions (1920x1080)
    fig = plt.figure(figsize=(19.2, 10.8))  # 1920x1080 pixels at 100 DPI

    # Create subplots for each column
    for i, column in enumerate(columns):
        ax = fig.add_subplot(num_rows, num_cols, i + 1)

        # Create the line plot
        sns.lineplot(x='timestamp', y=column, data=df, linewidth=1.5, ax=ax)

        # Add labels and title
        ax.set_xlabel('')
        ax.set_ylabel('')

        if column == 'success':
            ax.set_ylim(0, 2)
        elif column.startswith('AU'):
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(-0.6, 0.6)

        ax.set_title(column)
        ax.set_xlim(60, 80)

    # Add a main title
    plt.suptitle(f'{video_id}', fontsize=12)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for the suptitle

    # Save the figure to the video directory
    plt.savefig(os.path.join(video_dir, f"{video_id}.png"), dpi=100)
    plt.close()

    return True  # Successfully created

class DataProcessor:
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the facial expression analyzer

        Args:
            data_dir: Directory containing CSV or Parquet files
            output_dir: Directory to save output files
        """
        self.args = args

        # The table columns gaze_angle_x, gaze_angle_y it is eye gaze direction in radians in world coordinates
        # averaged for both eyes and converted into more easy to use format than gaze vectors.
        # If a person is looking left-right this will results in the change of gaze_angle_x(from positive to negative) and,
        # if a person is looking up-down this will result in change of gaze_angle_y (from negative to positive),
        # if a person is looking straight ahead both of the angles will be close to 0 (within measurement error).

        self.columns = ['success',
                        'gaze_angle_x', 'gaze_angle_y',
                        'pose_Rx', 'pose_Ry', 'pose_Rz',
                        'AU01_r', 'AU02_r', 'AU04_r',
                        'AU05_r', 'AU06_r', 'AU07_r',
                        'AU09_r', 'AU10_r', 'AU12_r',
                        'AU14_r', 'AU15_r', 'AU17_r',
                        'AU20_r', 'AU23_r', 'AU25_r',
                        'AU26_r', 'AU45_r']

        # Create output directories
        os.makedirs(self.args.output, exist_ok=True)
        os.makedirs(os.path.join(self.args.output, "graphs"), exist_ok=True)

    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and preprocess data from a CSV or Parquet file

        Args:
            file_path: Path to the file

        Returns:
            Preprocessed DataFrame
        """

        # Determine file type by extension
        file_extension = os.path.splitext(file_path)[1].lower()

        # Load data based on file type
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.parquet', '.pq']:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Only .csv and .parquet files are supported.")

        # Filter out rows where success is 0
        #df = df[df['success'] == 1]

        # Normalize AU intensity columns by dividing by 5
        au_columns = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]

        for col in au_columns:
            df[col] = df[col] / 5.0

        # Reset index after filtering
        df.reset_index(drop=True, inplace=True)

        return df

    def extract_features_for_video(self, df: pd.DataFrame, video_id: str) -> pd.DataFrame:
        """
        Extract time series features for a single video

        Args:
            df: DataFrame containing the video data
            video_id: ID of the video

        Returns:
            DataFrame with extracted features
        """
        # Prepare data for tsfresh

        # Create a dataframe suitable for tsfresh
        tsfresh_df = pd.DataFrame()

        #common_settings = {} # empty
        #common_settings = MinimalFCParameters().data
        common_settings = EfficientFCParameters().data

        settings = {}

        for col in self.columns:
            if col == 'success':
                continue
            temp_df = df[['timestamp', col]].copy()
            temp_df['id'] = f"{video_id}"         # ID for the video
            temp_df['kind'] = col                 # The kind of feature
            temp_df.rename(columns={col: 'value'}, inplace=True)

            # Append to the tsfresh dataframe
            tsfresh_df = pd.concat([tsfresh_df, temp_df[['id', 'timestamp', 'kind', 'value']]])

            col_settings = common_settings.copy()

            if col.startswith('AU'):
               col_settings['number_crossing_m'] = [{"m": 0.1}, {"m": 0.3}, {"m": 0.5}, {"m": 0.7}, {"m": 0.9}]

            # if col.startswith('AU'):
            #    col_settings['number_crossing_m'] = [{"m": 0.1}, {"m": 0.3}, {"m": 0.5}, {"m": 0.7}, {"m": 0.9}]
            # else:
            #     del col_settings['number_crossing_m']

            settings[col] = col_settings

        # Extract features using tsfresh
        features = extract_features(
            tsfresh_df,
            kind_to_fc_parameters=settings,
            column_id='id',
            column_sort='timestamp',
            column_kind='kind',
            column_value='value',
            impute_function=impute
        )

        # Ensure we return a DataFrame
        if isinstance(features, pd.DataFrame):
            return features
        else:
            # If features is not a DataFrame, return an empty DataFrame
            return pd.DataFrame()

    def create_columns_graphs(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Create intensity change graphs for each AU in each video

        Args:
            data_dict: Dictionary containing DataFrames for each video
        """
        # For each video
        for video_id, df in data_dict.items():
            # Create a directory for this video
            video_dir = os.path.join(self.args.output, "graphs", f"{video_id}")

            if os.path.exists(video_dir):
                continue

            os.makedirs(video_dir, exist_ok=True)

            # Plot each column
            for column in self.columns:
                plt.figure(figsize=(12, 6))

                # Create the line plot
                sns.lineplot(x='timestamp', y=column, data=df)

                if column == 'success':
                    plt.ylim(0, 2)
                elif column.startswith('AU'):
                    plt.ylim(0, 1)
                else:
                    plt.ylim(-0.6, 0.6)

                plt.ylim(0, 1)

                # Add labels and title
                plt.xlabel('Time (seconds)')
                plt.ylabel('Intensity')
                plt.title(f'{column} {video_id}')

                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(video_dir, f"{column}.png"))
                plt.close()

    def create_consolidated_graphs(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Create a consolidated image for each video showing all AU intensity graphs
        arranged in a grid with 5 graphs per row, processing videos in parallel.

        Args:
            data_dict: Dictionary containing DataFrames for each video
        """
        import concurrent.futures
        from functools import partial

        video_dir = os.path.join(self.args.output, "graphs")

        plot_func = partial(plot_single_video)

        # Determine optimal number of workers based on CPU cores
        max_workers = min(os.cpu_count() or 4, len(data_dict))

        # Process videos in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Create a list of (data, id) tuples for processing
            tasks = [(df, video_id) for video_id, df in data_dict.items()]

            # Submit all tasks and process results as they complete
            futures = {executor.submit(plot_func, data, vid_id, video_dir, self.columns): vid_id for data, vid_id in tasks}

            for future in concurrent.futures.as_completed(futures):
                video_id = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing video {video_id}: {e}")

    def process_videos(self, target_values: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Process all videos in the data directory

        Args:
            target_values: Target values for feature filtering (optional)

        Returns:
            DataFrame with extracted and filtered features
        """

        data_files = glob.glob(os.path.join(self.args.data, "*"))

        if not data_files:
            raise ValueError(f"No data files found in {self.args.data}")

        # Dictionary to store dataframes
        data_dict = {}
        all_features = []

        # Process each file
        for file_path in data_files:
            # Get video ID from filename
            video_id = os.path.splitext(os.path.basename(file_path))[0]

            # Load and preprocess the data
            df = self.load_and_preprocess_data(file_path)

            # Store the preprocessed data
            data_dict[video_id] = df

            # Extract features for this video
            if self.args.features:
                video_features = self.extract_features_for_video(df, video_id)
                video_features['video_id'] = video_id
                all_features.append(video_features)

        if self.args.plots:
            self.create_columns_graphs(data_dict)
            self.create_consolidated_graphs(data_dict)

        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features)

            combined_features.to_parquet(os.path.join(self.args.output, "extracted_features.parquet"))

            # Filter relevant features if target values are provided
            # if target_values is not None:
            #     # Ensure target_values align with features
            #     if len(target_values) != len(combined_features):
            #         raise ValueError("Target values must have the same length as the number of videos")

            #     # Select relevant features
            #     filtered_features = select_features(
            #         combined_features.drop(columns=['video_id']),
            #         target_values,
            #         fdr_level=0.05
            #     )

            #     # Add video_id back
            #     filtered_features['video_id'] = combined_features['video_id']

            #     # Save the filtered features
            #     filtered_features.to_csv(os.path.join(self.output_dir, "filtered_features.csv"))

            #     return filtered_features

            return combined_features
        else:
            return pd.DataFrame()

if __name__ == "__main__":
    main()