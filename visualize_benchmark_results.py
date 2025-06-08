import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

def load_results(benchmark_dir):
    """Load benchmark results from CSV files."""
    # Load summary file
    summary_path = os.path.join(benchmark_dir, 'model_comparison_summary.csv')
    if not os.path.exists(summary_path):
        print(f"Summary file not found at {summary_path}")
        return None, {}
    
    try:
        summary_df = pd.read_csv(summary_path)
    except Exception as e:
        print(f"Error reading summary file: {e}")
        return None, {}
    
    # Load individual model metrics
    model_metrics = {}
    for model in summary_df['Model'].unique():
        metrics_path = os.path.join(benchmark_dir, f"{model}_verification_metrics.csv")
        person_results_path = os.path.join(benchmark_dir, f"{model}_person_results.csv")
        
        if os.path.exists(metrics_path):
            try:
                model_metrics[model] = {
                    'verification_metrics': pd.read_csv(metrics_path)
                }
            except Exception as e:
                print(f"Error reading metrics for {model}: {e}")
        
        if os.path.exists(person_results_path):
            try:
                if model in model_metrics:
                    model_metrics[model]['person_results'] = pd.read_csv(person_results_path)
                else:
                    model_metrics[model] = {
                        'person_results': pd.read_csv(person_results_path)
                    }
            except Exception as e:
                print(f"Error reading person results for {model}: {e}")
    
    return summary_df, model_metrics

def plot_accuracy_comparison(summary_df, output_dir):
    """Plot accuracy comparison across models."""
    plt.figure(figsize=(12, 8))
    
    # Melt the dataframe to get a format suitable for seaborn
    plot_data = summary_df.melt(
        id_vars=['Model'],
        value_vars=['FF_Accuracy', 'PP_Accuracy', 'FP_Accuracy'],
        var_name='Verification Type',
        value_name='Accuracy (%)'
    )
    
    # Replace the verification type names for better readability
    plot_data['Verification Type'] = plot_data['Verification Type'].replace({
        'FF_Accuracy': 'Frontal-Frontal',
        'PP_Accuracy': 'Profile-Profile',
        'FP_Accuracy': 'Frontal-Profile'
    })
    
    # Create the grouped bar chart
    sns.barplot(x='Model', y='Accuracy (%)', hue='Verification Type', data=plot_data)
    
    plt.title('Verification Accuracy by Model and Face Orientation', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.1f', padding=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison_detailed.png'), dpi=300)
    plt.close()

def plot_roc_curves(model_metrics, output_dir):
    """Plot ROC curves for optimal thresholds."""
    plt.figure(figsize=(10, 8))
    
    for model, metrics in model_metrics.items():
        if 'verification_metrics' in metrics:
            df = metrics['verification_metrics']
            
            # Sort by threshold to ensure proper plotting
            df = df.sort_values('threshold')
            
            # Calculate ROC points
            far = df['far']
            tpr = 1 - df['frr']  # TPR = 1 - FRR
            
            # Find optimal threshold (closest to EER)
            eer_points = abs(far - (1 - tpr))
            optimal_idx = eer_points.argmin()
            optimal_threshold = df.iloc[optimal_idx]['threshold']
            optimal_far = far.iloc[optimal_idx]
            optimal_tpr = tpr.iloc[optimal_idx]
            
            # Calculate AUC (approximate)
            auc = np.trapz(tpr, far)
            
            # Plot the ROC curve
            plt.plot(far, tpr, label=f"{model} (AUC={auc:.3f})")
            
            # Mark the optimal point
            plt.scatter([optimal_far], [optimal_tpr], marker='o', s=100, 
                       edgecolors='black', label=f"{model} Optimal (t={optimal_threshold:.2f})")
    
    # Add diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    plt.xlabel('False Accept Rate (FAR)', fontsize=14)
    plt.ylabel('True Positive Rate (1-FRR)', fontsize=14)
    plt.title('ROC Curves with Optimal Thresholds', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_optimal.png'), dpi=300)
    plt.close()

def plot_similarity_distributions(model_metrics, output_dir):
    """Plot similarity score distributions for each model."""
    # Determine how many models we have
    num_models = len(model_metrics)
    if num_models == 0:
        return
    
    # Calculate grid dimensions
    cols = min(2, num_models)
    rows = (num_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (model, metrics) in enumerate(model_metrics.items()):
        if i >= len(axes):
            break
            
        if 'person_results' in metrics:
            df = metrics['person_results']
            
            # Extract the similarity scores
            ff_sim = df['ff_mean']
            pp_sim = df['pp_mean']
            fp_sim = df['fp_mean']
            
            # Plot distributions
            sns.kdeplot(ff_sim, ax=axes[i], label='Frontal-Frontal', shade=True)
            sns.kdeplot(pp_sim, ax=axes[i], label='Profile-Profile', shade=True)
            sns.kdeplot(fp_sim, ax=axes[i], label='Frontal-Profile', shade=True)
            
            axes[i].set_title(f"{model} Similarity Distributions")
            axes[i].set_xlabel('Cosine Similarity')
            axes[i].set_ylabel('Density')
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_distributions_by_model.png'), dpi=300)
    plt.close()

def plot_performance_metrics(summary_df, output_dir):
    """Plot various performance metrics for comparison."""
    # Create radar chart for overall performance
    metrics = ['ROC_AUC', 'FF_Accuracy', 'PP_Accuracy', 'FP_Accuracy']
    
    # Normalize processing time (lower is better)
    max_time = summary_df['Processing_Time'].max()
    summary_df['Processing_Speed'] = 1 - (summary_df['Processing_Time'] / max_time)
    metrics.append('Processing_Speed')
    
    # Convert EER to accuracy (lower is better)
    summary_df['EER_Score'] = 1 - summary_df['EER']
    metrics.append('EER_Score')
    
    # Normalize metrics to 0-1 scale for radar chart
    normalized_df = summary_df.copy()
    for metric in metrics:
        if metric in ['FF_Accuracy', 'PP_Accuracy', 'FP_Accuracy']:
            normalized_df[metric] = normalized_df[metric] / 100  # Already in percentage
        
    # Create radar chart
    models = normalized_df['Model'].tolist()
    
    # Number of variables
    N = len(metrics)
    
    # Create angle variables
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add each model
    for i, model in enumerate(models):
        values = normalized_df.loc[normalized_df['Model'] == model, metrics].values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels on spokes
    plt.xticks(angles[:-1], [m.replace('_', ' ') for m in metrics], size=12)
    
    # Remove radial labels and set y limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], size=10)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Performance Comparison', size=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_radar_chart.png'), dpi=300)
    plt.close()
    
    # Create bar chart for EER and Optimal Threshold
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.bar(x - width/2, summary_df['EER'], width, label='EER (lower is better)')
    plt.bar(x + width/2, summary_df['Optimal_Threshold'], width, label='Optimal Threshold')
    
    plt.xlabel('Model')
    plt.ylabel('Value')
    plt.title('EER and Optimal Threshold by Model')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(summary_df['EER']):
        plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(summary_df['Optimal_Threshold']):
        plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eer_threshold_comparison.png'), dpi=300)
    plt.close()

def plot_processing_time(summary_df, output_dir):
    """Plot processing time comparison."""
    plt.figure(figsize=(10, 6))
    
    # Sort by processing time
    df = summary_df.sort_values('Processing_Time')
    
    bars = plt.bar(df['Model'], df['Processing_Time'], color='skyblue')
    
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('Processing Time (seconds)', fontsize=14)
    plt.title('Model Processing Time Comparison', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}s', ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'processing_time_comparison.png'), dpi=300)
    plt.close()

def create_summary_table(summary_df, output_dir):
    """Create a visual summary table."""
    # Create a more readable summary table
    table_data = summary_df.copy()
    
    # Rename columns for better readability
    table_data = table_data.rename(columns={
        'FF_Accuracy': 'Frontal-Frontal Acc (%)',
        'PP_Accuracy': 'Profile-Profile Acc (%)',
        'FP_Accuracy': 'Frontal-Profile Acc (%)',
        'FF_Similarity': 'FF Similarity',
        'PP_Similarity': 'PP Similarity',
        'FP_Similarity': 'FP Similarity',
        'Processing_Time': 'Processing Time (s)'
    })
    
    # Round numeric columns
    for col in table_data.columns:
        if col != 'Model':
            if 'Acc' in col:
                table_data[col] = table_data[col].round(1)
            elif 'Time' in col:
                table_data[col] = table_data[col].round(2)
            else:
                table_data[col] = table_data[col].round(3)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, len(table_data) + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        loc='center',
        cellLoc='center'
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Highlight the best value in each column
    for col_idx, col in enumerate(table_data.columns[1:], start=1):
        if 'EER' in col:  # Lower is better
            best_idx = table_data[col].idxmin()
        elif 'Time' in col:  # Lower is better
            best_idx = table_data[col].idxmin()
        else:  # Higher is better
            best_idx = table_data[col].idxmax()
        
        table[(best_idx + 1, col_idx)].set_facecolor('lightgreen')
    
    plt.suptitle('Face Recognition Model Benchmark Summary', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_summary_table.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--benchmark_dir', type=str, default='benchmark_results',
                        help='Directory containing benchmark results')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations (defaults to benchmark_dir)')
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else args.benchmark_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set matplotlib defaults for better cross-platform compatibility
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
    
    # Load results
    summary_df, model_metrics = load_results(args.benchmark_dir)
    
    if summary_df is None:
        print("No benchmark results found.")
        return
    
    # Create visualizations
    print("Creating visualizations...")
    
    try:
        plot_accuracy_comparison(summary_df, output_dir)
        print("- Created accuracy comparison chart")
    except Exception as e:
        print(f"Error creating accuracy comparison chart: {e}")
    
    try:
        plot_roc_curves(model_metrics, output_dir)
        print("- Created ROC curves")
    except Exception as e:
        print(f"Error creating ROC curves: {e}")
    
    try:
        plot_similarity_distributions(model_metrics, output_dir)
        print("- Created similarity distributions")
    except Exception as e:
        print(f"Error creating similarity distributions: {e}")
    
    try:
        plot_performance_metrics(summary_df, output_dir)
        print("- Created performance metrics charts")
    except Exception as e:
        print(f"Error creating performance metrics charts: {e}")
    
    try:
        plot_processing_time(summary_df, output_dir)
        print("- Created processing time comparison")
    except Exception as e:
        print(f"Error creating processing time comparison: {e}")
    
    try:
        create_summary_table(summary_df, output_dir)
        print("- Created summary table")
    except Exception as e:
        print(f"Error creating summary table: {e}")
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == '__main__':
    main() 