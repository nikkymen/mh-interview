#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import itertools

from typing import Dict, List, Tuple, Optional

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file and filter out unsuccessful frames."""
    print(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    # Skip frames with success == 0
    df = df[df['success'] == 1]
    return df

def identify_au_columns(df: pd.DataFrame) -> List[str]:
    """Identify Action Unit columns in the dataframe."""
    au_pattern = re.compile(r'AU\d+_r')
    au_columns = [col for col in df.columns if au_pattern.match(col)]
    return au_columns

def detect_spikes(series: pd.Series, threshold: float = 0.5, min_drop: float = 0.2) -> int:
    """
    Detect complete spikes (rise and fall patterns) in a time series.
    A spike is counted when signal rises above threshold and then drops below (threshold - min_drop).
    """
    # For very short series, return 0
    if len(series) < 3:
        return 0

    # Normalize the series to 0-1 range for consistent threshold application
    if series.std() > 0 and series.max() > series.min():
        normalized = (series - series.min()) / (series.max() - series.min())
    else:
        normalized = series - series.min()

    # Set the lower threshold for detecting when a spike has ended
    lower_threshold = max(0, threshold - min_drop)

    spike_count = 0
    in_spike = False

    for i in range(len(normalized)):
        if not in_spike and normalized.iloc[i] >= threshold:
            # Start of a spike
            in_spike = True
        elif in_spike and normalized.iloc[i] <= lower_threshold:
            # End of a spike
            in_spike = False
            spike_count += 1

    # If we're still in a spike at the end of the series, count it if it was long enough
    if in_spike:
        spike_count += 1

    return spike_count

def extract_binned_features(series: pd.Series, num_bins: int = 10) -> Dict[str, float]:
    """Extract features by dividing the time series into fixed number of bins."""
    features = {}
    # Create equal-sized bins regardless of video length
    bin_indices = np.array_split(np.arange(len(series)), num_bins)

    for i, indices in enumerate(bin_indices):
        if len(indices) > 0:
            bin_values = series.iloc[indices]
            features[f"bin_{i}_mean"] = float(bin_values.mean())
            features[f"bin_{i}_max"] = float(bin_values.max())
            features[f"bin_{i}_std"] = float(bin_values.std()) if len(bin_values) > 1 else 0.0

    return features

def extract_frequency_features(series: pd.Series, num_components: int = 10) -> Dict[str, float]:
    """Extract frequency domain features using FFT."""
    features = {}

    # Apply FFT and get magnitudes
    if len(series) > 1:
        fft_values = np.abs(np.fft.fft(series - series.mean()))
        # Get the most significant frequencies
        significant_freqs = fft_values[:len(fft_values)//2]

        # Take top N components
        top_n = min(num_components, len(significant_freqs))
        if top_n > 0:
            for i in range(top_n):
                if i < len(significant_freqs):
                    features[f"fft_{i}"] = float(significant_freqs[i])

    return features

def extract_pattern_features(series: pd.Series) -> Dict[str, float]:
    """Extract features related to temporal patterns."""
    features = {}

    if len(series) > 2:
        # Direction changes (zero crossings of the derivative)
        diff_series = np.diff(series)
        direction_changes = ((diff_series[:-1] * diff_series[1:]) < 0).sum()
        features["direction_changes"] = float(direction_changes)

        # Time ascending/descending
        features["time_ascending"] = float((diff_series > 0).sum() / len(diff_series))

        # Longest streak above mean
        above_mean = series > series.mean()
        streaks = [sum(1 for _ in group) for val, group in itertools.groupby(above_mean) if val]
        features["longest_above_mean"] = float(max(streaks)) if streaks else 0.0

    return features

def aggregate_features(df: pd.DataFrame, au_columns: List[str]) -> Dict[str, float]:
    """Aggregate features for each Action Unit."""
    features: Dict[str, float] = {}
    video_duration = df['timestamp'].max() - df['timestamp'].min()

    for au in au_columns:
        series = df[au]

        # Basic statistics
        features[f"{au}_mean"] = float(series.mean())
        features[f"{au}_std"] = float(series.std())
        features[f"{au}_max"] = float(series.max())
        features[f"{au}_min"] = float(series.min())
        features[f"{au}_median"] = float(series.median())

        # Detect spikes (complete events like blinks)
        # Use different thresholds depending on the AU
        if au == "AU45_r":  # For blinking
            spikes = detect_spikes(series, threshold=0.5, min_drop=0.3)
        else:
            spikes = detect_spikes(series, threshold=0.4, min_drop=0.2)

        features[f"{au}_spike_count"] = spikes

        # Frequency of spikes per minute
        if video_duration > 0:
            features[f"{au}_spikes_per_minute"] = spikes / (video_duration / 60)
        else:
            features[f"{au}_spikes_per_minute"] = 0.0

        # Time above thresholds
        for threshold in [0.2, 0.5, 0.8]:
            time_above = (series > threshold).sum() / len(series)
            features[f"{au}_time_above_{threshold}"] = float(time_above)

        # Variability measures
        features[f"{au}_iqr"] = float(series.quantile(0.75) - series.quantile(0.25))
        if len(series) > 1:
            features[f"{au}_rate_of_change"] = float(np.abs(np.diff(series)).mean())
        else:
            features[f"{au}_rate_of_change"] = 0.0

        # Add new fixed-length feature vectors
        bin_features = extract_binned_features(series, num_bins=10)
        freq_features = extract_frequency_features(series, num_components=8)
        pattern_features = extract_pattern_features(series)

        # Add prefixes to identify features
        for k, v in bin_features.items():
            features[f"{au}_{k}"] = v
        for k, v in freq_features.items():
            features[f"{au}_{k}"] = v
        for k, v in pattern_features.items():
            features[f"{au}_{k}"] = v

    # Add general video statistics
    features["video_duration"] = float(video_duration)

    return features

def plot_au_intensity(df: pd.DataFrame, au_column: str, output_dir: str, file_name: str) -> str:
    """Plot intensity change of an Action Unit and save the plot."""
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df[au_column], label=au_column)

    # Visualize spikes with colored regions
    if len(df) > 3:
        # Normalize for spike detection
        series = df[au_column]
        if series.std() > 0 and series.max() > series.min():
            normalized = (series - series.min()) / (series.max() - series.min())
        else:
            normalized = series - series.min()

        threshold = 0.5 if au_column == "AU45_r" else 0.4
        lower_threshold = threshold - (0.3 if au_column == "AU45_r" else 0.2)

        in_spike = False
        spike_regions: List[Tuple[int, int]] = []
        start_idx = 0

        for i in range(len(normalized)):
            if not in_spike and normalized.iloc[i] >= threshold:
                # Start of spike
                in_spike = True
                start_idx = i
            elif in_spike and normalized.iloc[i] <= lower_threshold:
                # End of spike
                in_spike = False
                spike_regions.append((start_idx, i))

        # If still in a spike at the end
        if in_spike:
            spike_regions.append((start_idx, len(normalized)-1))

        # Add colored regions for spikes
        for start, end in spike_regions:
            plt.axvspan(df['timestamp'].iloc[start], df['timestamp'].iloc[end],
                       alpha=0.2, color='green', label='_spike')

    # Add annotations and labels
    plt.title(f"{au_column} Intensity Over Time - {file_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Intensity")
    plt.grid(True, alpha=0.3)

    # Add a single "Spike" label to the legend
    if len(df) > 3:
        handles, labels = plt.gca().get_legend_handles_labels()
        if 'Spike' not in labels and '_spike' in labels:
            for i, label in enumerate(labels):
                if label == '_spike':
                    labels[i] = 'Spike'
                    break
            plt.legend(handles, labels)
        else:
            plt.legend()
    else:
        plt.legend()

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the plot to a file
    plot_path = os.path.join(output_dir, f"{au_column}_{file_name}.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path

def process_video_file(file_path: str, plots_dir: str) -> Optional[Dict[str, float]]:
    """Process a single video file."""
    # Load and filter data
    df = load_data(file_path)
    if len(df) == 0:
        print(f"No valid frames found in {file_path}")
        return None

    # Get file name without extension
    file_name = os.path.basename(file_path).split('.')[0]

    # Identify Action Unit columns
    au_columns = identify_au_columns(df)
    if not au_columns:
        print(f"No Action Unit columns found in {file_path}")
        return None

    print(f"Found {len(au_columns)} Action Units")

    # Aggregate features
    features = aggregate_features(df, au_columns)

    # Create plots for each Action Unit
    for au in au_columns:
        plot_path = plot_au_intensity(df, au, plots_dir + '/' + file_name, file_name)
        print(f"Saved plot to {plot_path}")

    return features

def main() -> None:
    # Directory containing CSV files
    data_dir: str = 'data'
    plots_dir: str = 'plots'
    output_file: str = 'aggregated_features.csv'

    # Create output directory
    os.makedirs(plots_dir, exist_ok=True)

    # List available CSV files
    try:
        csv_files: List[str] = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    except FileNotFoundError:
        print(f"Error: '{data_dir}' directory not found.")
        return

    if not csv_files:
        print(f"No CSV files found in '{data_dir}' directory.")
        return

    print(f"Found {len(csv_files)} CSV files to process.")

    # Process each file and collect features
    all_features: Dict[str, Dict[str, float]] = {}
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        print(f"\nProcessing {file}...")

        features = process_video_file(file_path, plots_dir)
        if features:
            file_name = os.path.basename(file).split('.')[0]
            all_features[file_name] = features

    # Convert features to DataFrame and save
    if all_features:
        features_df = pd.DataFrame.from_dict(all_features, orient='index')
        features_df.to_csv(output_file)
        print(f"\nAggregated features saved to {output_file}")
        print(f"Features extracted: {features_df.shape[1]}")
        print(f"Videos processed: {features_df.shape[0]}")
    else:
        print("No features were extracted.")

if __name__ == "__main__":
    main()