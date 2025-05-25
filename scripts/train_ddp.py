import argparse
import inspect
import os
from typing import Literal

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data import get_dataloader
from src.training import EarlyStopping, Trainer, get_weighted_criterion
from src.utils import (SegmentationMetrics, get_logger, get_model, get_run_dir,
                       read_config, setup_ddp_process, write_config)


def train_ddp(
        rank: int,
        world_size: int,
        model_name: Literal['baseline', 'unet']
) -> None:
    """
    Dispatch a distributed training job for the specified model.

    Args:
        rank (int): Rank of the device
        world_size (int): Number of total devices
        model_name (Literal): String name of the model used

    Returns:
        None
    """
    setup_ddp_process(rank, world_size)
    torch.set_float32_matmul_precision("high")

    cfg = read_config()
    metrics = SegmentationMetrics(
    num_classes=cfg['hyperparams'][model_name]['num_classes'],
    device=rank
    )

    if rank == 0:
        run_dir = get_run_dir(cfg['runs'][model_name], model_name)
        chkpt_dir = os.path.join(run_dir, 'checkpoints')
        # save copy of cfg.yaml in run dir
        with open(os.path.join(run_dir, 'cfg.yaml'), 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        logger = get_logger(run_dir, "training.log")

    # not used on other ranks apart from rank 0
    else:
        chkpt_dir = ''
        logger = None

    if not torch.cuda.is_available():
        raise ValueError('No Cuda detected.')

    hyperparams = cfg['hyperparams'][f'{model_name}']
    model = get_model(cfg, model_name).to(rank)
    raw_model = model
    model = torch.compile(model)
    model = DDP(model, device_ids=[rank])
    logger.info()

    train_loader = get_dataloader(
        cfg,
        split="train",
        world_size=world_size,
        rank=rank,
        batch_size=hyperparams['batch_size']
    )
    val_loader = get_dataloader(
        cfg,
        split="val",
        world_size=world_size,
        rank=rank,
        batch_size=hyperparams['batch_size']
    )

    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    if rank == 0:
        logger.info(f"Using fused optimizer: {fused_available}")

    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        weight_decay=hyperparams['weight_decay'],
        lr=hyperparams['lr'],
        fused=fused_available
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=hyperparams['lr'],
        steps_per_epoch=len(train_loader),
        epochs=hyperparams['epochs'],
        pct_start=0.15,
    )

    early_stopping = EarlyStopping(patience=15)

    criterion = get_weighted_criterion(cfg, device=rank)

    trainer = Trainer(
        model=model,
        device=rank,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        logger=logger,
        scheduler=scheduler,
        metrics=metrics,
        checkpoint_dir=chkpt_dir,
        world_size=world_size,
        rank=rank
    )

    logger.info(f"Starting training with model: {model_name}, fused available: {fused_available}")

    for epoch in range(1, hyperparams['epochs'] + 1):
        if isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        trainer.train_epoch(epoch)
        results = trainer.validate_epoch(epoch)
        val_loss = results['loss']
        if rank == 0:
            trainer.save_checkpoint(epoch, raw_model)
            if val_loss < trainer.best_val_loss:
                trainer.best_checkpoint = epoch
                trainer.best_val_loss = val_loss
        stop_flag = early_stopping(results["mIoU"])

        if stop_flag:
            logger.info(f"Early stopping training at epoch {epoch}")
            break

    if rank == 0:
        trainer.determine_best_checkpoint()
        # increment run_id
        run_id = int(cfg['runs'][model_name])
        cfg['runs'][model_name] = str(run_id + 1)
        write_config(cfg)

    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['baseline', 'unet'], required=True)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp,
            args=(world_size, args.model,),
            nprocs=world_size,
            join=True)
