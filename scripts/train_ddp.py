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
from src.training import EarlyStopping, Trainer, get_loss_function
from src.utils import (SegmentationMetrics, get_best_loss, get_logger,
                       get_model, get_run_dir, read_config, setup_ddp_process,
                       write_config)


def train_ddp(
        rank: int,
        world_size: int,
        model_name: Literal['baseline', 'unet'],
        loss_name: Literal['weighted_cle', 'OHEMLoss', 'mixed_cle_dice', 'dice'],
        tuning: bool = False
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
    torch.backends.cudnn.benchmark = True

    cfg = read_config()
    metrics = SegmentationMetrics(
    num_classes=cfg['hyperparams'][model_name]['num_classes'],
    device=rank
    )

    if rank == 0:
        if not tuning:
            run_dir = get_run_dir(cfg['runs'][model_name], model_name)
        else:
            run_dir = get_run_dir(os.path.join('tuning', cfg['runs'][model_name]), model_name)
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
    model = get_model(cfg, model_name).to(rank, memory_format=torch.channels_last)
    raw_model = model
    model = DDP(model, device_ids=[rank])
    model = torch.compile(model)

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
        logger.info(f"Starting training with model: {model_name}. Using fused optimizer: {fused_available}")
        logger.info(f"Using run_dir: {run_dir}, using checkpoint dir: {chkpt_dir}")

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

    early_stopping = EarlyStopping(patience=10)

    if loss_name == 'best':
        run_id = int(cfg['runs'][model_name])
        loss_name = get_best_loss(run_dir, run_id - 3)
        if rank == 0:
            logger.info("Loss was autoselected via get_best_loss()")

    criterion = get_loss_function(loss_name, cfg, device=rank)

    logger.info(f"Loss used: {criterion.__class__.__name__}")

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


    for epoch in range(1, hyperparams['epochs'] + 1):
        if isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        trainer.train_epoch(epoch)
        results = trainer.validate_epoch(epoch)
        metric_score = early_stopping.get_metric_score(results)
        if rank == 0 and not tuning:
            trainer.save_checkpoint(epoch, raw_model)
        if metric_score > trainer.best_metric:
            trainer.best_checkpoint = epoch
            trainer.best_metric = metric_score

        if early_stopping(metric_score):
            if rank == 0 :
                logger.info(f"Early stopping training at epoch {epoch}")
            break
    dist.barrier()

    if rank == 0:
        if not tuning:
            trainer.determine_best_checkpoint()
            logger.info(f"metric_score: {trainer.best_metric}, best_checkpoint_epoch: {trainer.best_checkpoint}")
        # increment run_id
        run_id = int(cfg['runs'][model_name])
        cfg['runs'][model_name] = str(run_id + 1)
        write_config(cfg)

    dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['baseline', 'unet'], required=True)
    parser.add_argument(
        '--loss',
        type=str,
        choices=['weighted_cle', 'OHEMLoss', 'mixed_cle_dice', 'dice', 'best'],
        required=True
    )
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp,
            args=(world_size, args.model, args.loss, False,),
            nprocs=world_size,
            join=True)
