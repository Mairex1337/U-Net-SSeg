import argparse
import os
from typing import Literal

import torch
import yaml

from src.data import get_dataloader
from src.training import EarlyStopping, Trainer, get_weighted_criterion
from src.utils import (SegmentationMetrics, get_device, get_logger, get_model,
                       get_run_dir, read_config, write_config)


def train(model_name: Literal['baseline', 'unet']) -> None:
    """Pipeline for training on a single device"""
    cfg = read_config()

    run_dir = get_run_dir(cfg['runs'][model_name], model_name)
    chkpt_dir = os.path.join(run_dir, 'checkpoints')
    # save copy of cfg.yaml in run dir
    with open(os.path.join(run_dir, 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    logger = get_logger(run_dir, "training.log")
    device = get_device()

    metrics = SegmentationMetrics(
    num_classes=cfg['hyperparams'][model_name]['num_classes'],
    device=device
    )

    hyperparams = cfg['hyperparams'][f'{model_name}']
    model = get_model(cfg, model_name).to(device)

    train_loader = get_dataloader(
        cfg,
        split="train",
        batch_size=hyperparams['batch_size']
    )
    val_loader = get_dataloader(
        cfg,
        split="val",
        batch_size=hyperparams['batch_size']
    )

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        weight_decay=hyperparams['weight_decay'],
        lr=hyperparams['lr']
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=hyperparams['lr'],
        steps_per_epoch=len(train_loader),
        epochs=hyperparams['epochs'],
        pct_start=0.15,
    )

    early_stopping = EarlyStopping(patience=15)

    criterion = get_weighted_criterion(cfg, device=device)

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        logger=logger,
        scheduler=scheduler,
        metrics=metrics,
        checkpoint_dir=chkpt_dir
    )

    logger.info(f"Starting training with model: {model_name}")
    for epoch in range(1, hyperparams['epochs'] + 1):
        trainer.train_epoch(epoch)
        results = trainer.validate_epoch(epoch)

        metric_score = early_stopping.get_metric_score(results)
        trainer.save_checkpoint(epoch)
        if metric_score > trainer.best_metric:
            trainer.best_checkpoint = epoch
            trainer.best_metric = metric_score

        if early_stopping(metric_score):
            logger.info(f"Early stopping training at epoch {epoch}")
            break

    trainer.determine_best_checkpoint()

    # increment run_id
    run_id = int(cfg['runs'][model_name])
    cfg['runs'][model_name] = str(run_id + 1)

    write_config(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['baseline', 'unet'], required=True)
    args = parser.parse_args()
    if args.model == None:
        raise ValueError(f"Specify the model to train via `--model [model_name]`.\n"
                         f"Valid values are ['baseline', 'unet']")
    train(args.model)
