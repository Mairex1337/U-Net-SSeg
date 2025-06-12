import os
import zipfile

import gdown


def download_checkpoint_and_dataset():
    # Download checkpoint
    checkpoint_file_id = "1QF0SlfnzxUo2geXQ_sc-sUrE7l5iEofK"
    checkpoint_dir = os.path.join("outputs", "unet", "1", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    temp_checkpoint = os.path.join(checkpoint_dir, "checkpoint_temp.pth")
    gdown.download(id=checkpoint_file_id, output=temp_checkpoint, quiet=False)
    final_checkpoint = os.path.join(checkpoint_dir, "checkpoint_best.pth")
    os.rename(temp_checkpoint, final_checkpoint)
    print(f"Downloaded checkpoint to {final_checkpoint}")

    # Download dataset
    dataset_file_id = "1Ra2CeH_Q5z1aiojfQdlvENvcVkfPl-xG"
    dataset_zip_path = "dataset.zip"
    gdown.download(id=dataset_file_id, output=dataset_zip_path, quiet=False)
    print(f"Downloaded dataset to {dataset_zip_path}")

    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    print("Unzipped dataset.")

if __name__ == "__main__":
    download_checkpoint_and_dataset()