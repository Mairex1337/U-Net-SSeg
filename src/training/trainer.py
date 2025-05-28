import os
from contextlib import nullcontext
from typing import Optional

import torch
import torch.distributed as dist
import tqdm

from src.utils import Timer


class Trainer:
    """
    Handles training and validation of a PyTorch model.

    Encapsulates the training loop, validation loop, checkpoint saving,
    and performance logging.

    Args:
        model (torch.nn.Module): The model to train.
        device (str): Device identifier, e.g., 'cuda', 'cpu', or 'mps'.
        train_loader (DataLoader): Dataloader for training data.
        val_loader (DataLoader): Dataloader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer for model parameters.
        logger (logging.Logger): Logger for training output.
        scheduler (torch.optim.lr_scheduler): LR scheduler.
        checkpoint_dir (str): Directory to save model checkpoints.
    """
    def __init__(
            self, 
            model,
            device: str,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            logger,
            scheduler,
            metrics,
            checkpoint_dir: str,
            world_size: int = 0,
            rank: int = 0,
    ) -> None:
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.scheduler = scheduler
        self.metrics = metrics
        self.checkpoint_dir = checkpoint_dir
        self.world_size = world_size
        self.rank = rank
        self.ddp = world_size > 1
        self.best_val_loss = float("inf")
        self.best_checkpoint = -1
        self._bar_format = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{rate_fmt} {postfix}]"
        self._train_samples = len(self.train_loader.dataset)
        self._val_samples = len(self.val_loader.dataset)


    def train_epoch(self, epoch:int) -> float:
        """
        Trains the model for one epoch on the training dataset.

        Args:
            epoch (int): Current epoch number, used for logging.

        Returns:
            float: Average training loss over the epoch.
        """
        ...
        self.model.train()
        torch.cuda.reset_peak_memory_stats(self.device)
        total_loss = torch.tensor(0.0, device=self.device)
        if self.rank == 0:
            loop = tqdm.tqdm(
                total=len(self.train_loader.dataset), 
                desc=f"Train epoch {epoch}",
                unit=" samples",
                bar_format=self._bar_format
            )
        with (Timer() if self.rank == 0 else nullcontext()) as t:
            for images, masks in self.train_loader:
                images, masks = images.to(self.device, memory_format=torch.channels_last), masks.to(self.device)
                self.optimizer.zero_grad()
                with (torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.ddp else nullcontext()):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.detach()
                if self.rank == 0:
                    loop.set_postfix(loss=f"{loss.item():.4f}")
                    loop.update(len(images))
        
        memory_usage = torch.tensor(torch.cuda.max_memory_reserved(self.device) / 1024**3, device=self.device)
        if self.ddp:
            gathered_memory = [torch.zeros_like(memory_usage) for _ in range(self.world_size)]
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
            dist.all_gather(gathered_memory, memory_usage)

        avg_loss = (total_loss / len(self.train_loader)).item()
        if self.rank == 0:
            current_lr = self.scheduler.get_last_lr()[0]
            self.logger.info(f"LR: {current_lr:.6e}")
            memory_list = [mem.item() for mem in (gathered_memory if self.ddp else [memory_usage])]
            memory_log = " | ".join([f"Rank {i}: {mem:.4f} GB" for i, mem in enumerate(memory_list)])
            self.logger.info(
                f"Epoch {epoch} - Train loss: {avg_loss:.4f} - Throughput: {self._train_samples / t.elapsed:.2f} samples/s\n"
                f"Peak Memory usage: {memory_log}\nTotal Peak memory usage: {sum(memory_list):.4f} GB"
            )
        return avg_loss


    def validate_epoch(self, epoch:int) -> dict:
        """
        Evaluates the model on the validation dataset.

        Args:
            epoch (int): Current epoch number, used for logging.

        Returns:
            dict: Key value pairs of all computed validation metrics.
        """
        self.model.eval()
        self.metrics.reset()
        total_loss = torch.tensor(0.0, device=self.device)
        if self.rank == 0:
            loop = tqdm.tqdm(
                total=len(self.val_loader.dataset),
                desc=f"Validate epoch {epoch}",
                unit=" samples",
                bar_format=self._bar_format
            )
        with torch.no_grad():
            with (Timer() if self.rank == 0 else nullcontext()) as t:
                for images, masks in self.val_loader:
                    images, masks = images.to(self.device, memory_format=torch.channels_last), masks.to(self.device)
                    with (torch.autocast(device_type='cuda', dtype=torch.bfloat16) if self.ddp else nullcontext()):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, masks)
                    self.metrics.update(torch.argmax(outputs, dim=1), masks)
                    total_loss += loss.detach()
                    if self.rank == 0:
                        loop.set_postfix(loss=f"{loss.item():.4f}")
                        loop.update(len(images))
        results = self.metrics.compute()
        if self.ddp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        avg_loss = (total_loss / len(self.val_loader)).item()
        if self.rank == 0:
            self.logger.info(
                f"Epoch {epoch} - Validation loss: {avg_loss:.4f} - Throughput: {self._val_samples / t.elapsed:.2f} samples/s"
            )
            self.metrics.log_metrics(results, self.logger)
        results['loss'] = avg_loss
        return results
    
    def save_checkpoint(self, epoch: int, raw_model: Optional[object] = None) -> None:
        """
        Saves the model and optimizer state for the given epoch.

        Args:
            epoch (int): Epoch number to include in the checkpoint filename.
            raw_model (Optional[object]): non ddp wrapped model
        Returns:
            None
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = f"chkpt_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        state_dict = raw_model.state_dict() if raw_model is not None else self.model.state_dict()
        torch.save({
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, checkpoint_path)

    def determine_best_checkpoint(self) -> None:
        """
        Renames the best-performing checkpoint to mark it as the best.

        Raises:
            ValueError: If no best checkpoint was recorded.
            FileNotFoundError: If the expected checkpoint file doesn't exist.
        
        Returns:
            None
        """
        best = self.best_checkpoint
        if best == -1:
            raise ValueError("No best checkpoint found")
        src = os.path.join(self.checkpoint_dir, f'chkpt_epoch_{best}.pth')
        dst = os.path.join(self.checkpoint_dir, f'chkpt_epoch_{best}_best.pth')

        if not os.path.exists(src):
            raise FileNotFoundError(f"Checkpoint {src} does not exist.")
        os.rename(src, dst)
