import os
os.environ["NUMBA_DISABLE_CUDA"] = "1"

import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection import select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

from typing import Dict, Optional

class FacialExpressionAnalyzer:
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize the facial expression analyzer

        Args:
            data_dir: Directory containing CSV or Parquet files
            output_dir: Directory to save output files
        """
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "graphs"), exist_ok=True)

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
        df = df[df['success'] == 1]

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
        # For each AU column, we'll create a separate time series
        au_columns = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]

        # Create a dataframe suitable for tsfresh
        tsfresh_df = pd.DataFrame()

        for au_col in au_columns:
            # Create a temporary dataframe for this AU
            temp_df = df[['timestamp', au_col]].copy()
            temp_df['id'] = f"{video_id}"  # ID for the video
            temp_df['kind'] = au_col  # The kind of feature (which AU)
            temp_df.rename(columns={au_col: 'value'}, inplace=True)

            # Append to the tsfresh dataframe
            tsfresh_df = pd.concat([tsfresh_df, temp_df[['id', 'timestamp', 'kind', 'value']]])

        settings = ComprehensiveFCParameters()

    #     fc_parameters = {
    #    #     "number_cwt_peaks": [{"n": 1}, {"n": 3}, {"n": 5}, {"n": 10}, {"n": 50}]
    #         "number_crossing_m": [{"m": 1.5}]
    #     }

        # Extract features using tsfresh
        features = extract_features(
            tsfresh_df,
            default_fc_parameters=settings,
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

    def create_au_intensity_graphs(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Create intensity change graphs for each AU in each video

        Args:
            data_dict: Dictionary containing DataFrames for each video
        """
        # For each video
        for video_id, df in data_dict.items():
            # Create a directory for this video
            video_dir = os.path.join(self.output_dir, "graphs", f"{video_id}")

            if os.path.exists(video_dir):
                continue

            os.makedirs(video_dir, exist_ok=True)

            # Get all AU columns with intensity values (_r suffix)
            au_intensity_cols = [col for col in df.columns if col.endswith('_r')]

            # Plot each AU
            for au_col in au_intensity_cols:
                plt.figure(figsize=(12, 6))

                # Create the line plot
                sns.lineplot(x='timestamp', y=au_col, data=df)

                plt.ylim(0, 5)

                # Add labels and title
                plt.xlabel('Time (seconds)')
                plt.ylabel('Intensity')
                plt.title(f'{au_col} {video_id}')

                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(video_dir, f"{au_col}.png"))
                plt.close()

    def create_consolidated_graphs(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Create a consolidated image for each video showing all AU intensity graphs
        arranged in a grid with 5 graphs per row.

        Args:
            data_dict: Dictionary containing DataFrames for each video
        """
        video_dir = os.path.join(self.output_dir, "graphs")

        # For each video
        for video_id, df in data_dict.items():
            image_path = os.path.join(video_dir, f"{video_id}.png")

            if os.path.exists(image_path):
                continue

            # Get all AU columns with intensity values (_r suffix)
            au_intensity_cols = [col for col in df.columns if col.endswith('_r')]

            # Calculate grid dimensions
            num_aus = len(au_intensity_cols)
            num_cols = 5  # 5 graphs per row
            num_rows = (num_aus + num_cols - 1) // num_cols  # Ceiling division

            # Create a figure with FullHD dimensions (1920x1080)
            fig = plt.figure(figsize=(19.2, 10.8))  # 1920x1080 pixels at 100 DPI

            # Create subplots for each AU
            for i, au_col in enumerate(au_intensity_cols):
                ax = fig.add_subplot(num_rows, num_cols, i + 1)

                # Create the line plot
                sns.lineplot(x='timestamp', y=au_col, data=df, linewidth=1.5, ax=ax)

                # Add labels and title
                ax.set_xlabel('')
                ax.set_ylabel('')

                ax.set_title(au_col)
                ax.set_ylim(0, 5)

            # Add a main title
            plt.suptitle(f'{video_id}', fontsize=12)

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)  # Make room for the suptitle

            # Save the figure to the video directory
            plt.savefig(os.path.join(video_dir, f"{video_id}.png"), dpi=100)
            plt.close()

    def process_videos(self, target_values: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Process all videos in the data directory

        Args:
            target_values: Target values for feature filtering (optional)

        Returns:
            DataFrame with extracted and filtered features
        """
        # Get all CSV files in the data directory
        data_files = glob.glob(os.path.join(self.data_dir, "*"))

        if not data_files:
            raise ValueError(f"No data files found in {self.data_dir}")

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
            if False:
                video_features = self.extract_features_for_video(df, video_id)
                video_features['video_id'] = video_id
                all_features.append(video_features)

        # Create AU intensity graphs
        self.create_au_intensity_graphs(data_dict)

        # Create consolidated graphs
        self.create_consolidated_graphs(data_dict)

        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features)

            # Save the extracted features
            combined_features.to_csv(os.path.join(self.output_dir, "extracted_features.csv"))
            combined_features.to_parquet(os.path.join(self.output_dir, "extracted_features.parquet"),
                                         compression='snappy')

            # Filter relevant features if target values are provided
            if target_values is not None:
                # Ensure target_values align with features
                if len(target_values) != len(combined_features):
                    raise ValueError("Target values must have the same length as the number of videos")

                # Select relevant features
                filtered_features = select_features(
                    combined_features.drop(columns=['video_id']),
                    target_values,
                    fdr_level=0.05
                )

                # Add video_id back
                filtered_features['video_id'] = combined_features['video_id']

                # Save the filtered features
                filtered_features.to_csv(os.path.join(self.output_dir, "filtered_features.csv"))

                return filtered_features

            return combined_features
        else:
            return pd.DataFrame()


def main():

    """
    Main function to orchestrate the process
    """
    # Define data and output directories
    data_dir = "data/parquet"
    output_dir = "output"

    # Initialize the analyzer
    analyzer = FacialExpressionAnalyzer(data_dir, output_dir)

    # Process videos
    features = analyzer.process_videos()

    print(f"Processed videos and extracted {features.shape[1] - 1} features.")  # -1 for video_id
    print(f"Features saved to {os.path.join(output_dir, 'extracted_features.csv')}")
    print(f"Visualization graphs saved to {os.path.join(output_dir, 'graphs')}")

if __name__ == "__main__":
    main()