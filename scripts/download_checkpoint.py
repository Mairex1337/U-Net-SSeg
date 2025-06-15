import os

import gdown


def download_checkpoint() -> None:
    """
    Downloads dataset and best checkpoint for the project.

    Returns:
        None
    """
    # Download checkpoint
    checkpoint_file_id = "1hwLeGS4s-DqXg8HpslyWSseHhB7RZdVF"
    checkpoint_dir = os.path.join("outputs", "unet", "1", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    temp_checkpoint = os.path.join(checkpoint_dir, "checkpoint_temp.pth")
    gdown.download(id=checkpoint_file_id, output=temp_checkpoint, quiet=False)
    final_checkpoint = os.path.join(checkpoint_dir, "checkpoint_best.pth")
    os.rename(temp_checkpoint, final_checkpoint)
    print(f"Downloaded checkpoint to {final_checkpoint}")

if __name__ == "__main__":
    download_checkpoint()
