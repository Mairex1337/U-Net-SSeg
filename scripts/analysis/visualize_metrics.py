import matplotlib.pyplot as plt
import seaborn as sns
from log_parser import process_run_directory
import os
import numpy as np

def plot_training_metrics(train_df, output_dir):
    
    plt.figure(figsize=(10,6))
    sns.lineplot(data=train_df[train_df['type'] == 'train'], x='epoch', y='train_loss', label='Train')
    if 'val_loss' in train_df.columns:
        sns.lineplot(data=train_df[train_df['type'] == 'val'], x='epoch', y='val_loss', label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_plot.png'))
    plt.close()
    
    if 'throughput' in train_df.columns:
        plt.figure(figsize=(10,6))
        sns.lineplot(data=train_df[train_df['type'] == 'train'], x='epoch', y='throughput')
        plt.title('Training Throughput (samples/sec)')
        plt.xlabel('Epoch')
        plt.ylabel('Throughput')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'throughput.png'))
        plt.close()
    
    if 'lr' in train_df.columns:
        plt.figure(figsize=(10,6))
        sns.lineplot(data=train_df[train_df['type'] == 'train'], x='epoch', y='lr')
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate.png'))
        plt.close()

def plot_evaluation_metrics(eval_df, class_df, output_dir):

    plt.figure(figsize=(12, 6))
    global_metrics = eval_df[['pixel_accuracy', 'mean_accuracy', 'mean_iou', 'mean_dice']].mean().to_frame().reset_index()
    global_metrics.columns = ['Metric', 'Value']
    sns.barplot(data=global_metrics, x='Metric', y='Value')
    plt.title('Global Evaluation Metrics')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'global_metrics.png'))
    plt.close()


    plt.figure(figsize=(12, 8))
    class_melted = class_df.melt(id_vars=['class'], 
                                value_vars=['iou', 'dice', 'precision', 'recall'],
                                var_name='metric', 
                                value_name='value')
    sns.boxplot(data=class_melted, x='metric', y='value')
    plt.title('Distribution of Class Metrics')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'class_metric_distributions.png'))
    plt.close()

    plt.figure(figsize=(12, 10))
    class_df_sorted = class_df.sort_values('iou', ascending=False)
    
    plt.subplot(2, 1, 1)
    sns.barplot(data=class_df_sorted.head(10), x='iou', y='class', palette='viridis')
    plt.title('Top 10 Classes by IoU Score')
    plt.xlim(0, 1)
    
    plt.subplot(2, 1, 2)
    sns.barplot(data=class_df_sorted.tail(10), x='iou', y='class', palette='magma')
    plt.title('Bottom 10 Classes by IoU Score')
    plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_iou_rankings.png'))
    plt.close()


    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=class_df, x='recall', y='precision', hue='class', 
                   s=100, alpha=0.7, palette='tab20')
    
    for line in range(class_df.shape[0]):
        if class_df['iou'].iloc[line] < 0.2 or class_df['iou'].iloc[line] > 0.8:
            plt.text(class_df['recall'].iloc[line]+0.01, 
                    class_df['precision'].iloc[line],
                    class_df['class'].iloc[line],
                    horizontalalignment='left')
    
    plt.title('Precision-Recall Tradeoff by Class')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'precision_recall_tradeoff.png'))
    plt.close()

def visualize_run(run_dir):
    output_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    train_df, eval_df, class_df = process_run_directory(run_dir)
    
    # DEBUG: Print DataFrame status
    print("\n=== DataFrame Status ===")
    print(f"Training DF: {len(train_df)} rows | Columns: {list(train_df.columns)}")
    print(f"Evaluation DF: {len(eval_df)} rows | Columns: {list(eval_df.columns) if not eval_df.empty else 'EMPTY'}")
    print(f"Class DF: {len(class_df)} rows | Columns: {list(class_df.columns) if not class_df.empty else 'EMPTY'}")
    
    if not train_df.empty:
        plot_training_metrics(train_df, output_dir)
    
    # Modified condition with more detailed checks
    eval_condition = not eval_df.empty and not class_df.empty
    print(f"\nEvaluation plots condition: {eval_condition}")
    
    if eval_condition:
        plot_evaluation_metrics(eval_df, class_df, output_dir)
        print(f"Generated evaluation visualizations in {output_dir}")
    else:
        print("Skipping evaluation plots because:")
        if eval_df.empty:
            print("- eval_df is empty")
        if class_df.empty:
            print("- class_df is empty")

if __name__ == "__main__":
    import sys
    
    # Check command line argument
    if len(sys.argv) != 2:
        print("Usage: python visualize_metrics.py <run_directory>")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    # Verify directory exists
    if not os.path.exists(run_dir):
        print(f"Error: Directory does not exist - {run_dir}")
        sys.exit(1)
    
    # Run visualization with error handling
    try:
        print(f"Starting visualization for: {run_dir}")
        visualize_run(run_dir)
        print("Visualization completed successfully")
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        sys.exit(1)