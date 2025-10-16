import os
from pathlib import Path
from sys import base_prefix

os.environ["NUMBA_DISABLE_CUDA"] = "1"
os.environ["TCL_LIBRARY"] = str(Path(base_prefix) / "lib" / "tcl8.6")
os.environ["TK_LIBRARY"] = str(Path(base_prefix) / "lib" / "tk8.6")

import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description='Visualize features from parquet file')

    parser.add_argument('--input', type=str,
                        default='extracted_features_full.parquet',
                        help='Path to the parquet file with features')

    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    print(df['video_id'])

    # print(df.shape)

    # #matching_columns = [col for col in df.columns if col.startswith("AU45_r__")]

    #  # Plot the specific column values

    # # Check if column exists

    # column_id = 'AU45_r__number_crossing_m__m_0.5'
    # column = df[column_id]

    # print(column)

    # if column is not None:
    #     plt.figure(figsize=(12, 6))

    #     plt.subplot(1, 2, 1)
    #     plt.scatter(range(len(df)), column, alpha=0.6, s=10)
    #     plt.title(f'Scatter plot of {column_id}')
    #     plt.ylabel('Value')
    #     plt.xlabel('Index')

    #     # Create histogram to show distribution
    #     plt.subplot(1, 2, 2)
    #     sns.histplot(column, kde=True)
    #     plt.title(f'Distribution of {column_id}')
    #     plt.xlabel('Value')

    #     plt.tight_layout()
    #     plt.show()

if __name__ == "__main__":
    main()