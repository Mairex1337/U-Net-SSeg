import os
from pathlib import Path
from visualize_metrics import visualize_run

def process_all_runs(outputs_dir='outputs'):
    """Process all model runs in the outputs directory"""
    outputs_dir = Path(outputs_dir)
    
    for model_dir in outputs_dir.iterdir():
        if model_dir.is_dir():
            for run_dir in model_dir.iterdir():
                if run_dir.is_dir():
                    print(f"Processing: {model_dir.name}/{run_dir.name}")
                    visualize_run(run_dir)

if __name__ == '__main__':
    process_all_runs()

