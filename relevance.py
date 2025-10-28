import os
os.environ["NUMBA_DISABLE_CUDA"] = "1"

import pandas as pd
import argparse

from pathlib import Path
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.feature_selection.relevance import combine_relevance_tables


def main():
    parser = argparse.ArgumentParser(description='Visualize features from parquet file')

    parser.add_argument('--features', type=str,
                        default='data/features/af_opensmile.parquet',
                        help='Path to the parquet file with features')

    parser.add_argument('--actuals', type=str,
                        default='data/actuals.csv',
                        help='Path to the parquet file with features')

    args = parser.parse_args()

    features_path = Path(args.features)

    if features_path.suffix.lower() in {".parquet", ".pq"}:
        features_df = pd.read_parquet(features_path)
    elif features_path.suffix.lower() == ".csv":
        features_df = pd.read_csv(features_path, index_col = 0)
    else:
        raise ValueError(f"Unsupported features file type: {features_path.suffix}")

    actuals_df = pd.read_csv(args.actuals)
    actuals_df = actuals_df.set_index('id')

    # Align features_df with actuals_df by index
    # Assuming features_df index matches the 'id' column in actuals
    common_ids = features_df.index.intersection(actuals_df.index)
    features_aligned = features_df.loc[common_ids]
    actuals_aligned = actuals_df.loc[common_ids]

    print(f"Number of common samples: {len(common_ids)}")
    print(f"Features shape: {features_aligned.shape}")

    target_columns = ['WHO-5', 'PSS-4', 'GAD-7', 'PHQ-9', 'Alienation', 'Burnout']

    relevance_tables = []

    for target in target_columns:
        y = actuals_aligned[target]

        relevance_tables.append(calculate_relevance_table(features_aligned, y, ml_task='classification', fdr_level = 0.05, hypotheses_independent = True))

    relevance_table = combine_relevance_tables(relevance_tables)

    top_features = relevance_table.nsmallest(25, 'p_value')
    print(top_features[['p_value', 'relevant']])

if __name__ == "__main__":
    main()