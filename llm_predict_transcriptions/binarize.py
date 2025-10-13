import pandas as pd

def binarize_data(input_file, output_file, thresholds=None):
    thresholds = {
        'WHO-5': 35,
        'PSS-4': 8,
        'GAD-7': 6,
        'PHQ-9': 12,
        'Alienation': 2.4375,
        'Burnout': 3.35
    }

    df = pd.read_csv(input_file)

    df_binary = df.copy()

    # Binarize each column based on its threshold
    for column, threshold in thresholds.items():
        if column in df.columns:
            df_binary[column] = (df[column] > threshold).astype(int)

    # Save the binarized data
    df_binary.to_csv(output_file, index=False)
    print(f"Binarized data saved to {output_file}")

    return df_binary

if __name__ == "__main__":
    binarize_data('actuals_scores.csv', 'actuals_bin.csv')