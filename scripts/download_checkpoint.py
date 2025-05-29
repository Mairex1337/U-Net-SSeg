import os

import gdown


def download_checkpoint():
    # Google Drive file ID
    file_id = "1QF0SlfnzxUo2geXQ_sc-sUrE7l5iEofK"
    output_dir = os.path.join("outputs", "unet", "1", "checkpoints")
    os.makedirs(output_dir, exist_ok=True)

    # Download to temp filename first
    temp_path = os.path.join(output_dir, "checkpoint_temp.pth")
    gdown.download(id=file_id, output=temp_path, quiet=False)

    # Rename with _best
    final_path = os.path.join(output_dir, "checkpoint_best.pth")
    os.rename(temp_path, final_path)
    print(f"Downloaded and saved to {final_path}")

if __name__ == "__main__":
    download_checkpoint()