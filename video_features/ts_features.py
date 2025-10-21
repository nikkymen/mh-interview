import os

os.environ["NUMBA_DISABLE_CUDA"] = "1"

import glob
import pandas as pd
import argparse
import pathlib

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_selection import select_features
from tsfresh.feature_selection import significance_tests

#from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import MinimalFCParameters

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

def extract_features_dataframe(df: pd.DataFrame, video_id: str) -> pd.DataFrame:

    # Create a dataframe suitable for tsfresh
    tsfresh_df = pd.DataFrame()

    #common_settings = {} # empty
    #common_settings = MinimalFCParameters().data
    common_settings = EfficientFCParameters().data

    # del common_settings['value_count']

    # for col in common_settings:
    #     print(col)

    settings = {}

    for col in COLUMNS:
        if col == 'success':
            continue
        temp_df = df[['timestamp', col]].copy()
        temp_df['id'] = f"{video_id}"         # ID for the video
        temp_df['kind'] = col                 # The kind of feature
        temp_df.rename(columns={col: 'value'}, inplace=True)

        # Append to the tsfresh dataframe
        tsfresh_df = pd.concat([tsfresh_df, temp_df[['id', 'timestamp', 'kind', 'value']]])

        col_settings = common_settings.copy()

        # Adapt some parameters

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input/parquet')
    parser.add_argument('--output', type=str, default='output/features.parquet')

    args = parser.parse_args()

    features = []

    os.makedirs(pathlib.Path(args.output).parent, exist_ok=True)

    data_files = glob.glob(os.path.join(args.input, "*"))

    if not data_files:
        raise ValueError(f"No data files found in {args.input}")

    for file_path in data_files:
        video_id = os.path.splitext(os.path.basename(file_path))[0]
        df = load_dataframe(file_path)
        df_features = extract_features_dataframe(df, video_id)
        features.append(df_features)

    # Save result
    pd.concat(features).to_parquet(args.output)

if __name__ == "__main__":
    main()