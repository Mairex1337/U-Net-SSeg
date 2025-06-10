import re
from pathlib import Path
import pandas as pd

def parse_training_log(log_file):
    """Parse training log file and extract metrics"""
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
            # Parse training metrics
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
                
            # Parse validation metrics
            val_match = val_pattern.search(line)
            if val_match:
                metrics.append({
                    'epoch': int(val_match.group('epoch')),
                    'val_loss': float(val_match.group('val_loss')),
                    'val_throughput': float(val_match.group('val_throughput')),
                    'type': 'val'
                })
                continue
                
            # Parse learning rate
            lr_match = lr_pattern.search(line)
            if lr_match:
                current_lr = lr_match.group('lr')
    
    return pd.DataFrame(metrics)

import re
from pathlib import Path
import pandas as pd

def parse_evaluation_log(log_file):
    """Parse evaluation log file with more robust pattern matching"""
    # Updated patterns to match your exact format
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
        
        # Parse global metrics
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
        
        # Parse class metrics
        class_section = content.split("Per-Class Evaluation Metrics:")[-1]
        print("\n=== Raw class section ===")
        print(class_section)
        class_lines = [line.strip() for line in class_section.split('\n')]
        print(f"Found {len(class_lines)} per-class metric lines")

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
        
        if not class_data:
            print(f"Warning: Could not parse class metrics in {log_file}")
            
        return global_metrics, pd.DataFrame(class_data)

def process_run_directory(run_dir):
    """Process directory with better error handling"""
    run_dir = Path(run_dir)
    print(f"\nProcessing: {run_dir}")
    
    train_log = next(run_dir.glob('training.log'), None)
    train_df = parse_training_log(train_log) if train_log else pd.DataFrame()

    # Find evaluation log
    eval_log = next(run_dir.glob('eval.log'), None)
    
    if not eval_log:
        print("No eval.log found")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Parse evaluation log
    global_metrics, class_df = parse_evaluation_log(eval_log)
    
    if not global_metrics:
        print("Failed to parse evaluation metrics")
        
    return train_df, pd.DataFrame([global_metrics]), class_df