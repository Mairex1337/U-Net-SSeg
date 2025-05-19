import os
import time

import torch
import tqdm


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
        total_loss = 0.0
        loop = tqdm.tqdm(
            total=len(self.train_loader.dataset), 
            desc=f"Train epoch {epoch}",
            unit=" samples",
            bar_format=self._bar_format
        )
        with Timer() as t:
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
            f"Epoch {epoch} - Train loss: {avg_loss:.4f} - Throughput: {len(self._train_samples) / t.elapsed:.2f} samples/s"
        )
        return avg_loss
    
    def validate_epoch(self, epoch:int) -> float:
        """
        Evaluates the model on the validation dataset.

        Args:
            epoch (int): Current epoch number, used for logging.

        Returns:
            float: Average validation loss over the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        loop = tqdm.tqdm(
            total=len(self.val_loader.dataset),
            desc=f"Validate epoch {epoch}",
            unit="sample",
            bar_format=self._bar_format
        )
        with torch.no_grad(), Timer() as t:
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
                loop.update(len(images))
        avg_loss = total_loss / len(self.val_loader)
        self.logger.info(
            f"Epoch {epoch} - Validation loss: {avg_loss:.4f} - Throughput: {len(self._val_samples)/t.elapsed:.2f} samples/s"
        )
        return avg_loss
    
    def save_checkpoint(self, epoch: int) -> None:
        """
        Saves the model and optimizer state for the given epoch.

        Args:
            epoch (int): Epoch number to include in the checkpoint filename.

        Returns:
            None
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        filename = f"chkpt_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
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

    
class Timer:
    """Context manager for measuring elapsed time in seconds."""

    def __enter__(self) -> "Timer":
        self.start = time.time()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.time() - self.start
    

def get_weighted_criterion(cfg: dict, device: str) -> torch.nn.CrossEntropyLoss:
    """
    Returns a weighted CrossEntropyLoss for semantic segmentation.

    Computes class weights from pixel frequencies provided,
    applying inverse frequency normalization. Ignores the class index 255.

    Args:
        cfg (dict): Configuration dictionary
        device (str): Contains the device used for training

    Returns:
        torch.nn.CrossEntropyLoss: Weighted loss function with ignore_index set to 255.
    """
    frequencies = cfg["class_distribution"]["class_frequencies"]
    total_pixels = cfg["class_distribution"]["total_pixels"]
    id_to_class = cfg["class_distribution"]["id_to_class"]
    weights_by_id = torch.tensor([
        total_pixels / (len(id_to_class) * frequencies[id_to_class[i]])
        for i in range(len(id_to_class))
    ], device=device)
    return torch.nn.CrossEntropyLoss(weight=weights_by_id, ignore_index=255)
