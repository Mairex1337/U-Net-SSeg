import os
import time

import torch
import tqdm


class Trainer:
    def __init__(
            self, 
            model,
            device: str,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            logger,
            checkpoint_dir: str,
    ):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logger
        self.best_val_loss = float("inf")
        self.checkpoint_dir = checkpoint_dir
        self.best_checkpoint = -1
        self.bar_format = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{rate_fmt} {postfix}]"


    def train_epoch(self, epoch:int):
        self.model.train()
        total_loss = 0.0
        loop = tqdm.tqdm(
            total=len(self.train_loader.dataset), 
            desc=f"Train epoch {epoch}",
            unit=" samples",
            bar_format=self.bar_format
        )
        with Timer(len(self.train_loader.dataset)) as t:
            for images, masks in self.train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
                loop.update(len(images))
        avg_loss = total_loss / len(self.train_loader)
        self.logger.info(
            f"Epoch {epoch} - Train loss: {avg_loss:.4f} - Throughput: {t.samples_per_sec:.2f} samples/s"
        )
        return avg_loss
    
    def validate_epoch(self, epoch:int) -> float:
        self.model.eval()
        total_loss = 0.0
        loop = tqdm.tqdm(
            total=len(self.val_loader.dataset),
            desc=f"Validate epoch {epoch}",
            unit="sample",
            bar_format=self.bar_format
        )
        with torch.no_grad(), Timer(len(self.val_loader.dataset)) as t:
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
                loop.update(len(images))
        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(
            f"Epoch {epoch} - Validation loss: {avg_loss:.4f} - Throughput: {t.samples_per_sec:.2f} samples/s"
        )
        return avg_loss
    
    def save_checkpoint(self, epoch: int) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = f"chkpt_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, checkpoint_path)

    def determine_best_checkpoint(self) -> None:
        best = self.best_checkpoint
        if best == -1:
            raise ValueError("No best checkpoint found")
        src = os.path.join(self.checkpoint_dir, f'chkpt_epoch_{best}.pth')
        dst = os.path.join(self.checkpoint_dir, f'chkpt_epoch_{best}_best.pth')

        if not os.path.exists(src):
            raise FileNotFoundError(f"Checkpoint {src} does not exist.")
        os.rename(src, dst)

    
class Timer:
    def __init__(self, total_samples: int) -> None:
        self.total_samples = total_samples

    def __enter__(self) -> object:
        self.start = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.time() - self.start

    @property
    def samples_per_sec(self) -> float:
        return self.total_samples / self.elapsed