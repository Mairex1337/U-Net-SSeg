import argparse
import os
import random
import re
from typing import Any, Dict, Literal

import torch
import torch.multiprocessing as mp

from scripts.train_ddp import train_ddp
from src.utils import get_logger, read_config, resolve_path, write_config


def run_hyperparameter_tuning(
    model_name: Literal['baseline', 'unet'],
    loss_name: Literal['weighted_cle', 'OHEMLoss', 'mixed_cle_dice', 'dice'],
    num_tuning_trials: int = 10,
) -> Dict[str, Any]:

    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No GPUs found.")
    cfg = read_config()
    epochs = cfg['hyperparams'][model_name]['epochs']
    cfg['hyperparams'][model_name]['epochs'] = 50
    resize = cfg['transforms']['resize']
    cfg['transforms']['resize'] = [256, 448]
    cfg['runs'][model_name] = str(1)

    write_config(cfg)
    if "SLURM_JOBID" in os.environ:
        base_path = os.environ["TMPDIR"]
    else:
        base_path = resolve_path("")
    tuning_base_dir = os.path.join(base_path, "outputs", model_name, 'tuning')
    os.makedirs(tuning_base_dir, exist_ok=True)

    logger = get_logger(tuning_base_dir, "tuning.log")

    lr_search_space = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    weight_decay_search_space = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4]

    used_combinations = []

    best_metric = -float('inf')
    best_hyperparams = {}

    logger.info(f'Starting hyperparameter tuning for {num_tuning_trials} trials.')
    logger.info(f'Learning rate search space: {lr_search_space}')
    logger.info(f'Weights decay search space: {weight_decay_search_space}')

    for i in range(num_tuning_trials):
        while True:
            current_lr = random.choice(lr_search_space)
            current_weight_decay = random.choice(weight_decay_search_space)
            current_combination = [current_lr, current_weight_decay]
            if current_combination in used_combinations:
                continue
            else:
                used_combinations.append(current_combination)
                break
    
        logger.info(f"--- Tuning Trial {i+1}/{num_tuning_trials} ---")
        logger.info(f"Testing Learning Rate: {current_lr}, Weight Decay: {current_weight_decay}")

        cfg = read_config()
        
        cfg['hyperparams'][model_name]['lr'] = current_lr
        cfg['hyperparams'][model_name]['weight_decay'] = current_weight_decay
        

        run_id = cfg['runs'][model_name]
        log_path = os.path.join(tuning_base_dir, run_id, 'training.log')

        mp.spawn(
            train_ddp,
            args=(world_size, model_name, loss_name, True,),
            nprocs=world_size,
            join=True
        )
        result_str = open(log_path, 'r').readlines()[-1].split('-')[-1]
        pattern = r"metric_score:\s*(\d+\.\d*),\s*best_checkpoint_epoch:\s*(\d+)"
        match = re.search(pattern, result_str)
        metric_score = float(match.group(1))
        best_checkpoint_epoch = int(match.group(2))
        if metric_score is None or best_checkpoint_epoch is None:
            raise ValueError(f"Regex match failed, got 'None'!")


        if metric_score > best_metric:
            best_metric = metric_score
            best_hyperparams = {
                'lr': current_lr,
                'weight_decay': current_weight_decay
            }

    logger.info(f"Optimal found hyperparameters: lr = {best_hyperparams['lr']} | weight_decay = {best_hyperparams['weight_decay']}")
    cfg = read_config()
    cfg['hyperparams'][model_name]['lr'] = best_hyperparams['lr']
    cfg['hyperparams'][model_name]['weight_decay'] = best_hyperparams['weight_decay']
    cfg['runs'][model_name] = str(1)
    cfg['transforms']['resize'] = resize
    cfg['hyperparams'][model_name]['epochs'] = epochs
    write_config(cfg)

    logger.info(f"Wrote hyperparams into cfg")
    
    return best_hyperparams

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['baseline', 'unet'], required=True)
    parser.add_argument(
        '--loss',
        type=str,
        choices=['weighted_cle', 'OHEMLoss', 'mixed_cle_dice', 'dice'],
        required=True,
    )
    parser.add_argument('--trials', type=int, default=10)

    args = parser.parse_args()

    best_params = run_hyperparameter_tuning(
        model_name=args.model,
        loss_name=args.loss,
        num_tuning_trials=args.trials,
    )