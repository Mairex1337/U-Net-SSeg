import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scripts.analysis.log_parser import compute_sem, process_run_directory


def plot_training_metrics(train_df, output_dir):
    """
    Plot training metrics from the training DataFrame.
    Args:
        train_df (pd.DataFrame): DataFrame containing training metrics.
        output_dir (str): Directory to save the plots.
    """
    
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

def plot_evaluation_metrics(eval_df, class_df, train_df, output_dir):
    """
    Plot evaluation metrics from the evaluation DataFrame and class metrics DataFrame.
    Args:
        eval_df (pd.DataFrame): DataFrame containing evaluation metrics.
        class_df (pd.DataFrame): DataFrame containing per-class metrics.
        train_df (pd.DataFrame): DataFrame containing training metrics.
        output_dir (str): Directory to save the plots.
    """
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
    sns.barplot(data=class_df_sorted.head(10), x='iou', y='class', hue='class', palette='viridis', legend=False)
    plt.title('Top 10 Classes by IoU Score')
    plt.xlim(0, 1)
    
    plt.subplot(2, 1, 2)
    sns.barplot(data=class_df_sorted.tail(10), x='iou', y='class', hue='class', palette='magma', legend=False)
    plt.title('Bottom 10 Classes by IoU Score')
    plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_iou_rankings.png'))
    plt.close()


    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=class_df, x='recall', y='precision', hue='class', 
                   s=100, alpha=0.7, palette='tab20')
    
    for line in range(class_df.shape[0]):
        plt.text(class_df['recall'].iloc[line]+0.01, 
                class_df['precision'].iloc[line],
                class_df['class'].iloc[line],
                horizontalalignment='left')
    
    plt.title('Precision-Recall Tradeoff by Class')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'precision_recall_tradeoff.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    heatmap_data = class_df.set_index('class')[['iou', 'dice', 'precision', 'recall']]
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
    plt.title("Per-Class Evaluation Metrics Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_metrics_heatmap.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=train_df[train_df['type'] == 'train'], x='epoch', y='train_loss', label='Train Loss')
    if 'val_loss' in train_df.columns:
        sns.lineplot(data=train_df[train_df['type'] == 'val'], x='epoch', y='val_loss', label='Validation Loss')
    # Add horizontal line for final evaluation loss (converted from mean dice)
    if not eval_df.empty:
        mean_eval_loss = 1 - eval_df['mean_dice'].values[0]
        plt.axhline(mean_eval_loss, color='red', linestyle='--', label='Eval (1 - Dice)')
    plt.title("Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_val_loss_comparison.png'))
    plt.close()

def model_comparison_plot(combined_df, output_path):
    """
    Create a bar plot comparing model metrics against baseline metrics.
    Args:
        combined_df (pd.DataFrame): DataFrame containing combined metrics for all runs.
        output_path (str): Path to save the comparison plot.
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=combined_df[combined_df['type'] == 'train'],
                 x='epoch', y='train_loss', hue='run_name', linestyle='-')
    sns.lineplot(data=combined_df[combined_df['type'] == 'val'],
                 x='epoch', y='val_loss', hue='run_name', linestyle='--')

    plt.title("Training and Validation Loss: Model Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_dice_with_sem(sem_df):
    """
    Plot the mean Dice score with SEM for each run.
    Args:
        sem_df (pd.DataFrame): DataFrame containing mean Dice scores and SEM.
    """
    plt.figure(figsize=(8, 5))
    plt.bar(
        sem_df['run_name'],
        sem_df['mean_dice_mean'],
        yerr=sem_df['mean_dice_sem'],
        capsize=5,
        color=['skyblue', 'lightgreen']
    )
    plt.title("Mean Dice Score with SEM")
    plt.ylabel("Mean Dice Score")
    plt.xlabel("Run")
    plt.tight_layout()
    plt.savefig("outputs/eval_dice_sem.png")
    plt.close()

def visualize_run(run_dir):
    """
    Visualize training and evaluation metrics from a run directory.
    Args:
        run_dir (str): Path to the run directory.
    """
    output_dir = os.path.join(run_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    train_df, eval_df, class_df = process_run_directory(run_dir)
    
    if not train_df.empty:
        plot_training_metrics(train_df, output_dir)
    
    eval_condition = not eval_df.empty and not class_df.empty
    
    if eval_condition:
        plot_evaluation_metrics(eval_df, class_df, train_df, output_dir)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if len(args) == 1:
        run_dir = args[0]
        if not os.path.exists(run_dir):
            print(f"Run directory '{run_dir}' does not exist.")
            sys.exit(1)
        try:
            visualize_run(run_dir)
        except Exception as e:
            print(f"Error during visualization: {e}")
            sys.exit(1)
    
    elif len(args) > 1 and args[0] == '--compare':
        run_dirs = args[1:]
        for path in run_dirs:
            if not os.path.exists(path):
                print(f"Run directory '{path}' does not exist.")
                sys.exit(1)
        
        try:
            all_train_dfs = []
            eval_dfs = []

            for run_dir in run_dirs:
                train_df, global_df, _ = process_run_directory(run_dir)
                run_name = Path(run_dir).name
                train_df["run_name"] = run_name
                global_df["run_name"] = run_name
                all_train_dfs.append(train_df)
                eval_dfs.append(global_df)

            eval_df = pd.concat(eval_dfs, ignore_index=True)
            if eval_df.empty:
                print("No evaluation metrics found in the provided run directories.")
                sys.exit(1)

            sem_df = compute_sem(eval_df)
            sem_df.to_csv(os.path.join('outputs/sem_results.csv'), index=False)
            plot_dice_with_sem(sem_df)

            train_df = pd.concat(all_train_dfs, ignore_index=True)
            model_comparison_plot(train_df, "outputs/comparison_loss_curve.png")

        except Exception as e:
            print(f"Error during comparison: {e}")
            sys.exit(1)

    else:
        print("Usage: python visualize_metrics.py <run_directory> OR python visualize_metrics.py --compare <run_dir1> <run_dir2> ...")
        sys.exit(1)