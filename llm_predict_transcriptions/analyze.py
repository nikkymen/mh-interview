import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy.optimize import minimize_scalar

def find_optimal_threshold(actual_values, predicted_binary, metric='f1'):
    """
    Find optimal threshold to convert continuous values to binary for maximum similarity
    """
    def objective(threshold):
        binary_pred = (actual_values >= threshold).astype(int)
        if metric == 'f1':
            return -f1_score(predicted_binary, binary_pred)
        elif metric == 'accuracy':
            return -accuracy_score(predicted_binary, binary_pred)
        elif metric == 'precision':
            return -precision_score(predicted_binary, binary_pred, zero_division=0)
        elif metric == 'recall':
            return -recall_score(predicted_binary, binary_pred, zero_division=0)

    # Search for optimal threshold
    result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
    return result.x

def analyze_thresholds():
    # Load the datasets
    predictions = pd.read_csv('predictions2/gemini-2.5-pro.csv')
    actuals = pd.read_csv('actuals_2.csv')

    # Parameters to analyze
    parameters = ['WHO-5', 'PSS-4', 'GAD-7', 'PHQ-9', 'Alienation', 'Burnout']

    # Merge on id column
    merged = pd.merge(predictions, actuals, on='id', suffixes=('_pred', '_actual'))

    results = {}

    print("Finding optimal thresholds for each parameter:")
    print("=" * 60)

    for param in parameters:
        pred_col = f"{param}_pred"
        actual_col = f"{param}_actual"

        if pred_col in merged.columns and actual_col in merged.columns:
            # Find optimal threshold
            threshold = find_optimal_threshold(
                merged[actual_col].values,
                merged[pred_col].values
            )

            # Calculate metrics with optimal threshold
            binary_actual = (merged[actual_col] >= threshold).astype(int)
            binary_pred = merged[pred_col]

            f1 = f1_score(binary_pred, binary_actual)
            accuracy = accuracy_score(binary_pred, binary_actual)
            precision = precision_score(binary_pred, binary_actual, zero_division=0)
            recall = recall_score(binary_pred, binary_actual, zero_division=0)

            results[param] = {
                'threshold': threshold,
                'f1_score': f1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall
            }

            print(f"{param}:")
            print(f"  Optimal threshold: {threshold:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print()

    # Save results to CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv('optimal_thresholds.csv')

    # Apply thresholds and save adjusted actuals
    adjusted_actuals = actuals.copy()
    for param in parameters:
        if param in results and param in adjusted_actuals.columns:
            threshold = results[param]['threshold']
            adjusted_actuals[param] = (adjusted_actuals[param] >= threshold).astype(int)

    adjusted_actuals.to_csv('adjusted_actuals.csv', index=False)

    print("Results saved to:")
    print("- optimal_thresholds.csv (threshold values and metrics)")
    print("- adjusted_actuals.csv (actuals converted to binary using optimal thresholds)")

    return results

if __name__ == "__main__":
    results = analyze_thresholds()