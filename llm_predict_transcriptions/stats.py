import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pathlib

from sklearn.metrics import confusion_matrix, classification_report

def merge_prediction_results(predictions_file, actuals_file, output_file):
    # Load both CSV files
    pred_df = pd.read_csv(predictions_file)
    actual_df = pd.read_csv(actuals_file)

    # Parameters to compare (all 6 binary parameters)
    parameters = ['WHO-5', 'PSS-4', 'GAD-7', 'PHQ-9', 'Alienation', 'Burnout']

    # Merge dataframes on 'id' column
    merged_df = pd.merge(pred_df, actual_df, on='id', suffixes=('_pred', '_actual'))

    # Add columns for prediction results
    for param in parameters:
        pred_col = f"{param}_pred" if f"{param}_pred" in merged_df.columns else param
        actual_col = f"{param}_actual" if f"{param}_actual" in merged_df.columns else param

        # Binary result: 1 if prediction matches actual value, 0 if incorrect
        merged_df[f'{param}_result'] = (merged_df[pred_col] == merged_df[actual_col]).astype(int)

    # Save to CSV
    merged_df.to_csv(output_file, index=False)
    print(f"Merged results saved to: {output_file}")

    return merged_df

def analyze_predictions(results_df, output_dir="prediction_analysis"):
    """
    Generate statistics and plots for prediction results.

    Parameters:
    results_df (DataFrame): Merged results dataframe with predictions and actuals
    output_dir (str): Directory to save results in
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Parameters analyzed
    parameters = ['WHO-5', 'PSS-4', 'GAD-7', 'PHQ-9', 'Alienation', 'Burnout']

    # Open text file for results
    with open(f"{output_dir}/prediction_stats.txt", "w") as f:
        f.write("PREDICTION STATISTICS\n")
        f.write("====================\n\n")

        # Overall accuracy
        overall_results = []
        for param in parameters:
            result_col = f"{param}_result"
            if result_col in results_df.columns:
                accuracy = results_df[result_col].mean() * 100
                overall_results.append((param, accuracy))
                f.write(f"{param} accuracy: {accuracy:.2f}%\n")

        f.write("\n")
        avg_accuracy = np.mean([r[1] for r in overall_results])
        f.write(f"Average accuracy across all parameters: {avg_accuracy:.2f}%\n\n")

        # Detailed metrics for each parameter
        for param in parameters:
            pred_col = f"{param}_pred" if f"{param}_pred" in results_df.columns else param
            actual_col = f"{param}_actual" if f"{param}_actual" in results_df.columns else param

            if pred_col in results_df.columns and actual_col in results_df.columns:
                f.write(f"\nDetailed metrics for {param}:\n")
                f.write("-------------------------\n")

                # Confusion matrix
                tn, fp, fn, tp = confusion_matrix(
                    results_df[actual_col],
                    results_df[pred_col],
                    labels=[0, 1]
                ).ravel()

                f.write(f"True Positives: {tp}\n")
                f.write(f"True Negatives: {tn}\n")
                f.write(f"False Positives: {fp}\n")
                f.write(f"False Negatives: {fn}\n\n")

                # Precision, recall, F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")

                # Classification report
                report = classification_report(
                    results_df[actual_col],
                    results_df[pred_col],
                    output_dict=True
                )

                # Class distribution
                class_dist = results_df[actual_col].value_counts()
                f.write(f"\nClass distribution:\n")
                f.write(f"Class 0: {class_dist.get(0, 0)}\n")
                f.write(f"Class 1: {class_dist.get(1, 0)}\n")

    # Create visualizations

# Plot 1: Accuracy by parameter
    plt.figure(figsize=(10, 6))
    params = [r[0] for r in overall_results]
    accuracies = [r[1] for r in overall_results]

    ax = sns.barplot(x=params, y=accuracies)
    plt.axhline(y=avg_accuracy, color='red', linestyle='--', label=f'Average: {avg_accuracy:.2f}%')
    plt.ylim(0, 100)

    plt.title('Prediction Accuracy by Parameter')
    plt.xlabel('Parameter')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/accuracy_by_parameter.png")

    # Plot 2: Combined confusion matrices
    # Calculate number of rows and columns for subplots
    n_params = len(parameters)
    n_cols = 3  # Display 3 confusion matrices per row
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_params > 1 else [axes]

    for i, param in enumerate(parameters):
        pred_col = f"{param}_pred" if f"{param}_pred" in results_df.columns else param
        actual_col = f"{param}_actual" if f"{param}_actual" in results_df.columns else param

        if pred_col in results_df.columns and actual_col in results_df.columns:
            cm = confusion_matrix(results_df[actual_col], results_df[pred_col])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                        xticklabels=["Negative (0)", "Positive (1)"],
                        yticklabels=["Negative (0)", "Positive (1)"])
            axes[i].set_title(f'{param}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_confusion_matrices.png")

    # Plot 3: Overall results comparison
    plt.figure(figsize=(12, 7))

    # Prepare data for grouped bar plot
    metrics_data = []
    for param in parameters:
        pred_col = f"{param}_pred" if f"{param}_pred" in results_df.columns else param
        actual_col = f"{param}_actual" if f"{param}_actual" in results_df.columns else param

        if pred_col in results_df.columns and actual_col in results_df.columns:
            report = classification_report(
                results_df[actual_col],
                results_df[pred_col],
                output_dict=True
            )

            metrics_data.append({
                'Parameter': param,
                'Accuracy': report['accuracy'] * 100,
                'Precision': report['weighted avg']['precision'] * 100,
                'Recall': report['weighted avg']['recall'] * 100,
                'F1-Score': report['weighted avg']['f1-score'] * 100
            })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_melted = pd.melt(metrics_df, id_vars=['Parameter'],
                             value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                             var_name='Metric', value_name='Score')

    sns.barplot(x='Parameter', y='Score', hue='Metric', data=metrics_melted)
    plt.title('Performance Metrics by Parameter')
    plt.xlabel('Parameter')
    plt.ylabel('Score (%)')
    plt.xticks(rotation=45)
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_metrics.png")

    # Plot 4 - Class Distribution Plot
    plt.figure(figsize=(12, 6))
    class_dist_data = []

    for param in parameters:
        actual_col = f"{param}_actual" if f"{param}_actual" in results_df.columns else param
        if actual_col in results_df.columns:
            # Get counts for each class
            counts = results_df[actual_col].value_counts().reset_index()
            counts.columns = ['Class', 'Count']
            counts['Parameter'] = param
            class_dist_data.append(counts)

    class_dist_df = pd.concat(class_dist_data)
    sns.barplot(x='Parameter', y='Count', hue='Class', data=class_dist_df)
    plt.title('Class Distribution by Parameter')
    plt.xlabel('Parameter')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Class')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/class_distribution.png")

    # Plot 5 - Prediction vs Actual Distribution
    plt.figure(figsize=(15, 10))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, param in enumerate(parameters):
        pred_col = f"{param}_pred" if f"{param}_pred" in results_df.columns else param
        actual_col = f"{param}_actual" if f"{param}_actual" in results_df.columns else param

        if pred_col in results_df.columns and actual_col in results_df.columns:
            # Create a DataFrame for this parameter's distribution
            dist_df = pd.DataFrame({
                'Actual': results_df[actual_col].value_counts().sort_index(),
                'Predicted': results_df[pred_col].value_counts().sort_index()
            }).fillna(0).astype(int)

            # Plot comparison of distributions
            dist_df.plot(kind='bar', ax=axes[i])
            axes[i].set_title(f'{param} Distribution')
            axes[i].set_xlabel('Class')
            axes[i].set_ylabel('Count')
            axes[i].set_xticklabels(['Negative (0)', 'Positive (1)'])

    plt.tight_layout()
    plt.savefig(f"{output_dir}/pred_vs_actual_distribution.png")

    # Plot 6 - Error Distribution (where predictions are wrong)
    plt.figure(figsize=(12, 6))
    error_counts = []

    for param in parameters:
        result_col = f"{param}_result"
        pred_col = f"{param}_pred" if f"{param}_pred" in results_df.columns else param
        actual_col = f"{param}_actual" if f"{param}_actual" in results_df.columns else param

        if all(col in results_df.columns for col in [result_col, pred_col, actual_col]):
            # Filter to just the errors
            errors = results_df[results_df[result_col] == 0]

            # Count false positives and false negatives
            fp_count = errors[(errors[pred_col] == 1) & (errors[actual_col] == 0)].shape[0]
            fn_count = errors[(errors[pred_col] == 0) & (errors[actual_col] == 1)].shape[0]

            error_counts.append({'Parameter': param, 'Error Type': 'False Positive', 'Count': fp_count})
            error_counts.append({'Parameter': param, 'Error Type': 'False Negative', 'Count': fn_count})

    error_df = pd.DataFrame(error_counts)
    sns.barplot(x='Parameter', y='Count', hue='Error Type', data=error_df)
    plt.title('Error Distribution by Parameter')
    plt.xlabel('Parameter')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Error Type')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/error_distribution.png")

    print(f"Analysis complete! Results saved to {output_dir}/")
    return results_df

if __name__ == "__main__":
    predictions = 'predictions2/gpt-oss-120b.csv'

    output_dir = f'stats/{pathlib.Path(predictions).stem}'
    merged_df = merge_prediction_results(predictions, 'actuals_bin.csv', f'{output_dir}/summary_table.csv')

    analyze_predictions(merged_df, output_dir = output_dir)