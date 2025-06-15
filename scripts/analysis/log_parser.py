import re
from pathlib import Path

import pandas as pd


def parse_training_log(log_file):
    """
    Parse training log file to extract training and validation metrics.
    Args:
        log_file (str): Path to the training log file.
    Returns:
        pd.DataFrame: DataFrame containing training and validation metrics.
    """
    pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - "
        r"Epoch (?P<epoch>\d+) - Train loss: (?P<train_loss>\d+\.\d+) - "
        r"Throughput: (?P<throughput>\d+\.\d+) samples/s"
    )
    
    val_pattern = re.compile(
        r"Epoch (?P<epoch>\d+) - Validation loss: (?P<val_loss>\d+\.\d+) - "
        r"Throughput: (?P<val_throughput>\d+\.\d+) samples/s"
    )
    
    lr_pattern = re.compile(r"LR: (?P<lr>\d+\.\d+e-\d+)")
    
    metrics = []
    current_epoch = None
    current_lr = None
    
    with open(log_file, 'r') as f:
        for line in f:
            train_match = pattern.search(line)
            if train_match:
                current_epoch = int(train_match.group('epoch'))
                metrics.append({
                    'epoch': current_epoch,
                    'train_loss': float(train_match.group('train_loss')),
                    'throughput': float(train_match.group('throughput')),
                    'lr': float(current_lr) if current_lr else None,
                    'type': 'train'
                })
                continue
                
            val_match = val_pattern.search(line)
            if val_match:
                metrics.append({
                    'epoch': int(val_match.group('epoch')),
                    'val_loss': float(val_match.group('val_loss')),
                    'val_throughput': float(val_match.group('val_throughput')),
                    'type': 'val'
                })
                continue
                
            lr_match = lr_pattern.search(line)
            if lr_match:
                current_lr = lr_match.group('lr')
    
    return pd.DataFrame(metrics)

import re
from pathlib import Path

import pandas as pd


def parse_evaluation_log(log_file):
    """
    Parse evaluation log file to extract global and per-class metrics.
    Args:
        log_file (str): Path to the evaluation log file.
    Returns:
        tuple: Global metrics as a dictionary and per-class metrics as a DataFrame.
    """
    global_pattern = re.compile(
        r"Evaluation Results:\s*\n"
        r".*Pixel Accuracy:\s+(?P<pixel_acc>\d+\.\d+)\s*\n"
        r".*Mean Accuracy:\s+(?P<mean_acc>\d+\.\d+)\s*\n"
        r".*Mean IoU:\s+(?P<mean_iou>\d+\.\d+)\s*\n"
        r".*Mean Dice \(F1\):\s+(?P<mean_dice>\d+\.\d+)\s*\n"
    )
    
    class_pattern = re.compile(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - (?P<class>[\w\s]+?)\s+"
        r"(?P<iou>\d+\.\d+)\s+"
        r"(?P<dice>\d+\.\d+)\s+"
        r"(?P<precision>\d+\.\d+)\s+"
        r"(?P<recall>\d+\.\d+)"
    )
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        global_match = global_pattern.search(content)
        if not global_match:
            print(f"Warning: Could not parse global metrics in {log_file}")
            return {}, pd.DataFrame()
            
        global_metrics = {
            'pixel_accuracy': float(global_match.group('pixel_acc')),
            'mean_accuracy': float(global_match.group('mean_acc')),
            'mean_iou': float(global_match.group('mean_iou')),
            'mean_dice': float(global_match.group('mean_dice'))
        }
        
        class_section = content.split("Per-Class Evaluation Metrics:")[-1]
        class_lines = [line.strip() for line in class_section.split('\n')]

        class_data = []
        for line in class_lines:
            match = class_pattern.search(line)
            if match:
                class_data.append({
                    'class': match.group('class'),
                    'iou': float(match.group('iou')),
                    'dice': float(match.group('dice')),
                    'precision': float(match.group('precision')),
                    'recall': float(match.group('recall'))
                })
        
        return global_metrics, pd.DataFrame(class_data)

def process_run_directory(run_dir):
    """
    Process a run directory to extract training and evaluation metrics.
    Args:
        run_dir (str): Path to the run directory.
    Returns:
        tuple: DataFrames for training metrics, global evaluation metrics, and per-class metrics.
    """
    run_dir = Path(run_dir)
    
    train_log = next(run_dir.glob('training.log'), None)
    train_df = parse_training_log(train_log) if train_log else pd.DataFrame()

    eval_log = next(run_dir.glob('eval.log'), None)
    global_metrics, class_df = parse_evaluation_log(eval_log)
    
    return train_df, pd.DataFrame([global_metrics]), class_df

def collect_eval_results(run_dirs):
    """
    Collect global evaluation results from multiple run directories.
    Args:
        run_dirs (list of str): List of run directory paths.
    Returns:
        pd.DataFrame: Combined DataFrame with global metrics for all runs.
    """
    summary = []

    for run_dir in run_dirs:
        _, global_df, _ = process_run_directory(run_dir)
        if not global_df.empty:
            global_df['run_name'] = Path(run_dir).name  # Optional: or use full path
            summary.append(global_df)

    return pd.concat(summary, ignore_index=True)

def compute_sem(eval_summary_df):
    """
    Compute mean and SEM for each run group.
    Args:
        eval_summary_df (pd.DataFrame): DataFrame with all global eval metrics.
    Returns:
        pd.DataFrame: Aggregated DataFrame with mean and SEM values.
    """
    sem_df = eval_summary_df.groupby("run_name").agg(['mean', 'sem'])
    sem_df.columns = ['_'.join(col).strip() for col in sem_df.columns.values]
    sem_df.reset_index(inplace=True)
    return sem_df

"""miou"""