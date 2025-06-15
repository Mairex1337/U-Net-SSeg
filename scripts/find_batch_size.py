import argparse
import inspect
import os
from typing import Literal

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data import get_dataloader
from src.training import Trainer, get_weighted_criterion
from src.utils import (SegmentationMetrics, get_logger, get_model, get_run_dir,
                       read_config, setup_ddp_process, write_config)


def find_batch_size(
        rank: int,
        world_size: int,
        model_name: Literal['baseline', 'unet']
) -> None:
    """
    Finds the maximum fittable batch size for the hardware used.

    Args:
        rank (int): Rank of the device
        world_size (int): Number of total devices
        model_name (Literal): String name of the model used

    Returns:
        None
    """
    setup_ddp_process(rank, world_size)
    torch.set_float32_matmul_precision("high")

    try:
        batch_size = 1
        cfg = read_config()
        if rank == 0:
            run_dir = get_run_dir(cfg['runs'][model_name], model_name)
            chkpt_dir = os.path.join(run_dir, 'checkpoints')
            logger = get_logger(run_dir, "training.log")
        # not used on other ranks apart from rank 0
        else:
            chkpt_dir = ''
            logger = None

        while True:
            metrics = SegmentationMetrics(
                num_classes=cfg['hyperparams'][model_name]['num_classes'],
                device=rank
            )
            cfg['hyperparams'][model_name]['batch_size'] = batch_size

            if not torch.cuda.is_available():
                raise ValueError('No Cuda detected.')
            
            if rank == 0:
                logger.info(f"Starting batch size search with batch_size = {cfg['hyperparams'][model_name]['batch_size']}")
                logger.info(f"Using run_dir: {run_dir}, using checkpoint dir: {chkpt_dir}")
            hyperparams = cfg['hyperparams'][f'{model_name}']
            model = get_model(cfg, model_name).to(rank, memory_format=torch.channels_last)
            raw_model = model
            model = DDP(model, device_ids=[rank])

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

            if isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
                train_loader.sampler.set_epoch(1)
            trainer.train_epoch(1)

            del model, raw_model, trainer, train_loader, val_loader, optimizer, scheduler, criterion
            torch.cuda.empty_cache()

            batch_size *= 2

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        if rank == 0:
            logger.info(f"Max batch size that fits: {batch_size // 2}")
            cfg["hyperparams"][model_name]['batch_size'] = batch_size // 2
            write_config(cfg)
        return

    finally:
        dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['baseline', 'unet'], required=True)
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(find_batch_size,
            args=(world_size, args.model,),
            nprocs=world_size,
            join=True)
