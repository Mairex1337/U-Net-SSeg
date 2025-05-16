import argparse
import os
from typing import Literal

import torch
import yaml

from src.data.dataloader import get_dataloader
from src.models.baseline_model import BaselineModel
from src.utils.output import get_device, get_logger, get_run_dir
from src.utils.read_config import read_config
from src.utils.resolve_path import resolve_path
from src.utils.trainer import Trainer

MODELS = {'baseline': BaselineModel}

def train(model_name: Literal['baseline', 'unet']):
    cfg = read_config()

    run_dir = get_run_dir(cfg['runs'][model_name], model_name)
    chkpt_dir = os.path.join(run_dir, 'checkpoints')
    logger = get_logger(run_dir)
    device = get_device()

    assert model_name in MODELS.keys()
    hyperparams = cfg['hyperparams'][f'{model_name}']
    model = MODELS[model_name](
        hyperparams['input_dim'],
        hyperparams['hidden_dim'],
        hyperparams['num_classes']
    ).to(device)

    train_loader = get_dataloader(
        cfg,
        train=True,
        batch_size=hyperparams['batch_size']
    )
    val_loader = get_dataloader(
        cfg,
        train=False,
        batch_size=hyperparams['batch_size']
    )

    optimizer = torch.optim.AdamW(model.parameters(), hyperparams['lr'])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255) #TODO: make inverse weight adjusted

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        logger=logger,
        checkpoint_dir=chkpt_dir
    )

    logger.info(f"Starting training with model: {model_name}")
    for epoch in range(1, hyperparams['epochs'] + 1):
        trainer.train_epoch(epoch)
        val_loss = trainer.validate_epoch(epoch)
        trainer.save_checkpoint(epoch)
        if val_loss < trainer.best_val_loss:
            trainer.best_checkpoint = epoch
            trainer.best_val_loss = val_loss
    
    trainer.determine_best_checkpoint()

    # save copy of cfg.yaml in run dir
    with open(os.path.join(run_dir, 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # increment run_id
    run_id = int(cfg['runs'][model_name])
    cfg['runs'][model_name] = str(run_id + 1)
    cfg_path = resolve_path('cfg.yaml', 2)

    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['baseline', 'unet'])
    args = parser.parse_args()
    if args.model == None:
        raise ValueError(f"Specify the model to train via `--model [model_name]`.\n"
                         f"Valid values are ['baseline', 'unet']")
    train(args.model)
