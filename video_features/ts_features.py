import os

#os.environ["NUMBA_DISABLE_CUDA"] = "1"

import glob
import pandas as pd
import argparse
import pathlib

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute

COLUMNS = ['success',
           'gaze_angle_x', 'gaze_angle_y',
           'pose_Rx', 'pose_Ry', 'pose_Rz',
           'AU01_r', 'AU02_r', 'AU04_r',
           'AU05_r', 'AU06_r', 'AU07_r',
           'AU09_r', 'AU10_r', 'AU12_r',
           'AU14_r', 'AU15_r', 'AU17_r',
           'AU20_r', 'AU23_r', 'AU25_r',
           'AU26_r', 'AU45_r']

GAZE_COLS = ["gaze_angle_x", "gaze_angle_y"]
POSE_COLS = ["pose_Rx", "pose_Ry", "pose_Rz"]
AU_COLS = ['AU01_r', 'AU02_r', 'AU04_r',
           'AU05_r', 'AU06_r', 'AU07_r',
           'AU09_r', 'AU10_r', 'AU12_r',
           'AU14_r', 'AU15_r', 'AU17_r',
           'AU20_r', 'AU23_r', 'AU25_r',
           'AU26_r', 'AU45_r']

COLUMNS = GAZE_COLS + POSE_COLS + AU_COLS

test_stats = {
    "mean": None
}

base_stats = {
    "mean": None,
    "median": None,
    "variance": None,
    "skewness": None,
    "kurtosis": None,
    "quantile": [{"q":q} for q in [0.1,0.25,0.75,0.9]]
}

dyn_stats = {
    "absolute_sum_of_changes": None,
    "mean_abs_change": None,
    "mean_second_derivative_central": None,
    "change_quantiles": [{"f_agg":"mean","isabs":True,"qh":0.9,"ql":0.1}],
    "autocorrelation": [{"lag":l} for l in [10,20,30]],
    "spkt_welch_density": [{"coeff":c} for c in [5,10,20,30]],
    "fft_aggregated": [{"aggtype":a} for a in ["centroid","variance","skew","kurtosis"]],
    "approximate_entropy": [{"m":2,"r":0.2}],
    "sample_entropy": None,
    "time_reversal_asymmetry_statistic": [{"lag":l} for l in [1,2,3]],
}

pose_extras = {
    "linear_trend": [{"attr":a} for a in ["slope","pvalue","rvalue"]],
    "ratio_beyond_r_sigma": [{"r":r} for r in [1.0,1.5,2.0]],
    "number_peaks": [{"n":n} for n in [1,3,5]],
}

au_common = {
    **base_stats,
    **{
      "absolute_sum_of_changes": None,
      "mean_abs_change": None,
      "change_quantiles": [{"f_agg":"mean","isabs":True,"qh":0.9,"ql":0.1}],
      "longest_strike_above_mean": None,
      "longest_strike_below_mean": None,
      "number_peaks": [{"n":n} for n in [1,3,5]],
      "spkt_welch_density": [{"coeff":c} for c in [5,10,20,30]],
      "fft_aggregated": [{"aggtype":a} for a in ["centroid","variance"]],
      "approximate_entropy": [{"m":2,"r":0.2}],
      "sample_entropy": None,
    }
}

au_specific = {
    "AU45": {
        **au_common,
        "range_count": [{"min":2.0,"max":5.0}],
        "number_crossing_m": [{"m":1.0},{"m":2.0},{"m":3.0}, {"m":4.0}]
    },

    "AU12": { **au_common, "range_count": [{"min":1.5,"max":5.0}] },
    "AU06": { **au_common, "range_count": [{"min":1.5,"max":5.0}] },
    "AU04": { **au_common, "range_count": [{"min":1.0,"max":5.0}] },
    "AU15": { **au_common, "range_count": [{"min":1.0,"max":5.0}] },
}

normalize = [
    'number_peaks',
    'number_crossing_m',
    'count_above_mean',
    'count_below_mean',
    'range_count',
    'absolute_sum_of_changes',
    'sum_values'
]

kind_to_fc_parameters = {}

# for c in GAZE_COLS:
#     kind_to_fc_parameters[c] = {**test_stats}

# for c in POSE_COLS:
#     kind_to_fc_parameters[c] = {**test_stats}

# for c in AU_COLS:
#     kind_to_fc_parameters[c] = {**test_stats}

for c in GAZE_COLS:
    kind_to_fc_parameters[c] = {**base_stats, **dyn_stats}

for c in POSE_COLS:
    kind_to_fc_parameters[c] = {**base_stats, **dyn_stats, **pose_extras}

for c in AU_COLS:
    kind_to_fc_parameters[c] = au_specific.get(c, au_common)

def load_dataframe(file_path: str) -> pd.DataFrame:
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.parquet', '.pq']:
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    df.reset_index(drop=True, inplace=True)
    return df

def extract_features_dataframe(df: pd.DataFrame, id: str) -> pd.DataFrame:

    # Create a dataframe suitable for tsfresh
    tsfresh_df = pd.DataFrame()

    for col in COLUMNS:
        temp_df = df[['timestamp', col]].copy()
        temp_df['id'] = f"{id}"               # ID for the video
        temp_df['kind'] = col                 # The kind of feature
        temp_df.rename(columns={col: 'value'}, inplace=True)

        tsfresh_df = pd.concat([tsfresh_df, temp_df[['id', 'timestamp', 'kind', 'value']]])

    features = extract_features(
        tsfresh_df,
        kind_to_fc_parameters=kind_to_fc_parameters,
        column_id='id',
        column_sort='timestamp',
        column_kind='kind',
        column_value='value',
        impute_function=impute
    )

    if not isinstance(features, pd.DataFrame):
        return pd.DataFrame()

    # Add video length as feature
    video_length = df['timestamp'].iloc[-1]
    features['video_length'] = video_length

    # Normalize features by video length
    for col in features.columns:
        for norm_feature in normalize:
            if norm_feature in col:
                features[col] = features[col] / video_length
                break

    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/video_features/parquet/')
    parser.add_argument('--output', type=str, default='data/video_features/video_features.parquet')

    args = parser.parse_args()

    features = []

    os.makedirs(pathlib.Path(args.output).parent, exist_ok=True)

    data_files = glob.glob(os.path.join(args.input, "*"))

    if not data_files:
        raise ValueError(f"No data files found in {args.input}")

    for file_path in data_files:
        video_id = os.path.splitext(os.path.basename(file_path))[0]
        df = load_dataframe(file_path)

        print(video_id)
        df_features = extract_features_dataframe(df, video_id)
        features.append(df_features)

    # Save result
    pd.concat(features).to_parquet(args.output)

if __name__ == "__main__":
    main()