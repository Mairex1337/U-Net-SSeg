import torch
import tqdm
import os

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

    def train_epoch(self, epoch:int):
        self.model.train()
        total_loss = 0.0
        loop = tqdm.tqdm(self.train_loader, desc=f"Train epoch {epoch}")
        for images, masks in loop:
            images, masks = images.to(self.device), masks.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(self.train_loader)
        self.logger.log(f"Epoch {epoch} - Train loss: {avg_loss:.4f}")
        return avg_loss
    
    def validate_epoch(self, epoch:int):
        self.model.eval()
        total_loss = 0.0
        loop = tqdm.tqdm(self.val_loader, desc=f"Validate epoch {epoch}")
        with torch.no_grad():
            for images, masks in loop:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(self.val_loader)
        self.logger.log(f"Epoch {epoch} - Validation loss: {avg_loss:.4f}")
        return avg_loss
    
    def save_checkpoint(self, epoch:int) -> None:
        checkpoint_path = os.path.join(self.checkpoint_dir, f"chkp_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, checkpoint_path)
        self.logger.log(f"Checkpoint saved at {checkpoint_path}")



    

        