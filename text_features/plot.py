import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, cohen_kappa_score

predictions_file = 'predictions/gpt-5.csv'

# Load data
predictions = pd.read_csv(predictions_file)
actual = pd.read_csv('actuals.csv')

output_dir = "plots/" + Path(predictions_file).stem + "/"

os.makedirs(output_dir, exist_ok=True)

# Ensure both dataframes have the same IDs and are sorted consistently
predictions = predictions.sort_values('id')
actual = actual.sort_values('id')

# Metrics of interest (excluding ID column)
metrics = ["WHO-5", "PSS-4", "GAD-7", "PHQ-9", "Alienation", "Burnout"]

# Metric scales and interpretations
scales = {
    "WHO-5": 3,
    "PSS-4": 3,
    "GAD-7": 4,
    "PHQ-9": 5,
    "Alienation": 3,
    "Burnout": 3
}

# Store results
results = pd.DataFrame(columns=['Metric', 'Accuracy', 'Weighted F1', 'Cohen Kappa'])

# Create figures for combined plots
fig_cm, axes_cm = plt.subplots(2, 3, figsize=(16, 10))
fig_dist, axes_dist = plt.subplots(2, 3, figsize=(18, 10))

# Process each metric
for i, metric in enumerate(metrics):
    # Calculate subplot position
    row = i // 3
    col = i % 3

    y_true = actual[metric]
    y_pred = predictions[metric]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)

    # Add to results
    new_row = pd.DataFrame({'Metric': [metric], 'Accuracy': [accuracy],
                            'Weighted F1': [f1], 'Cohen Kappa': [kappa]})
    results = pd.concat([results, new_row], ignore_index=True)

    # Get classification report
    report = classification_report(y_true, y_pred)
    print(f"\nClassification Report for {metric}:\n")
    print(report)

    # Plot confusion matrix in the combined figure
    ax_cm = axes_cm[row, col]

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(1, scales[metric] + 1))

    # Normalize confusion matrix (by row)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0

    # Create heatmap with normalized values but show actual counts
    sns.heatmap(cm_norm, annot=cm, fmt='d', cmap='Blues',
                xticklabels=range(1, scales[metric] + 1),
                yticklabels=range(1, scales[metric] + 1),
                ax=ax_cm)

    ax_cm.set_ylabel('True Label')
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_title(f'{metric}')

    # Create bar plot of class distributions in the combined figure
    ax_dist = axes_dist[row, col]

    # Set up data for plotting class distributions
    actual_counts = y_true.value_counts().reindex(range(1, scales[metric] + 1)).fillna(0)
    predicted_counts = y_pred.value_counts().reindex(range(1, scales[metric] + 1)).fillna(0)

    df_plot = pd.DataFrame({
        'Class': list(range(1, scales[metric] + 1)) * 2,
        'Count': actual_counts.tolist() + predicted_counts.tolist(),
        'Type': ['Actual'] * scales[metric] + ['Predicted'] * scales[metric]
    })

    sns.barplot(x='Class', y='Count', hue='Type', data=df_plot, palette='Set2', ax=ax_dist)
    ax_dist.set_title(f'{metric}')
    ax_dist.set_xlabel('Class')
    ax_dist.set_ylabel('Count')
    ax_dist.legend(title='')

# Adjust layout and save combined confusion matrices
fig_cm.suptitle('Confusion Matrices for All Metrics', fontsize=16)
fig_cm.tight_layout(rect=[0, 0, 1, 0.96])
plt.figure(fig_cm.number)
plt.savefig(output_dir + 'confusion_matrix.png')

# Adjust layout and save combined class distributions
fig_dist.suptitle('Class Distributions for All Metrics', fontsize=16)
fig_dist.tight_layout(rect=[0, 0, 1, 0.96])
plt.figure(fig_dist.number)
plt.savefig(output_dir + 'class_distributions.png')

# Show overall results
print("\nOverall Results:\n")
print(results)

# Plot performance metrics comparison
plt.figure(figsize=(12, 8))
results_melted = pd.melt(results, id_vars=['Metric'],
                         value_vars=['Accuracy', 'Weighted F1', 'Cohen Kappa'],
                         var_name='Measure', value_name='Score')
sns.barplot(x='Metric', y='Score', hue='Measure', data=results_melted, palette='viridis')
plt.title('Performance Metrics by Parameter')
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='')
plt.tight_layout()
plt.savefig(output_dir + 'performance_comparison.png')
plt.close()

# Create error analysis heatmap
plt.figure(figsize=(10, 8))
error_df = pd.DataFrame()
for metric in metrics:
    error_df[metric] = predictions[metric] - actual[metric]

# Plot correlation between errors
error_corr = error_df.corr()
sns.heatmap(error_corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Prediction Errors')
plt.tight_layout()
plt.savefig(output_dir + 'error_correlation.png')
plt.close()

# Distribution of error magnitudes
plt.figure(figsize=(12, 6))
error_melted = pd.melt(error_df.abs(), var_name='Metric', value_name='Error Magnitude')
sns.boxplot(x='Metric', y='Error Magnitude', data=error_melted)
plt.title('Distribution of Error Magnitudes')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(output_dir + 'error_magnitude.png')

print("\nAnalysis complete. All visualizations have been saved.")

# Create detailed results table with predictions, actuals and errors
detailed_results = pd.DataFrame({'id': predictions['id']})

# Add predictions, actuals and errors for each metric
for metric in metrics:
    detailed_results[f"{metric}_predicted"] = predictions[metric]
    detailed_results[f"{metric}_actual"] = actual[metric]
    detailed_results[f"{metric}_error"] = abs(predictions[metric] - actual[metric])

# Save detailed results table
detailed_results.to_csv(output_dir + "detailed_results.csv", index=False)
