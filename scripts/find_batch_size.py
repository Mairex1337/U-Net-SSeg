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
from src.models import baseline_model
from src.training import Trainer, get_weighted_criterion
from src.utils import (cleanup, get_logger, get_model, get_run_dir,
                       read_config, resolve_path, setup_ddp_process,
                       write_config)


def find_batch_size(
        rank: int,
        world_size: int,
        model_name: Literal['baseline', 'unet']
) -> None:
    setup_ddp_process(rank, world_size)
    try:
        batch_size = 2
        while True:
            cfg = read_config()
            cfg['hyperparams'][model_name]['batch_size'] = batch_size
            if rank == 0:
                run_dir = get_run_dir(cfg['runs'][model_name], model_name)
                chkpt_dir = os.path.join(run_dir, 'checkpoints')
                logger = get_logger(run_dir, "training.log")

            # not used on other ranks apart from rank 0
            else:
                chkpt_dir = ''
                logger = None

            if not torch.cuda.is_available():
                raise ValueError('No Cuda detected.')

            hyperparams = cfg['hyperparams'][f'{model_name}']
            model = get_model(cfg, model_name).to(rank)
            model = torch.compile(model)
            raw_model = model
            model = DDP(model, device_ids=[rank])

            train_loader = get_dataloader(
                cfg,
                train=True,
                world_size=world_size,
                rank=rank,
                batch_size=hyperparams['batch_size']
            )
            val_loader = get_dataloader(
                cfg,
                train=False,
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
            criterion = get_weighted_criterion(cfg, device=rank)

            trainer = Trainer(
                model=model,
                device=rank,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                logger=logger,
                checkpoint_dir=chkpt_dir,
                world_size=world_size,
                rank=rank
            )


            trainer.train_epoch(1)

            del model, raw_model, trainer, train_loader, val_loader
            torch.cuda.empty_cache()

            batch_size *= 2
        
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            if rank == 0:
                print(f"Max batch size that fits: {batch_size // 2}")
                cfg["hyperparams"][model_name]['batch_size'] = batch_size // 2
                write_config(cfg)
                return
        else:
            raise e

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

