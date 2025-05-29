import os
import shutil


def cleanup_temp_dirs(base_dir: str) -> None:
    """
    Deletes known temporary folders and output file from previous runs.

    Args:
        base_dir (str): The current working directory to clean up in.
    """
    paths = [
        os.path.join(base_dir, "output.json"),
        os.path.join(base_dir, "temp_input"),
        os.path.join(base_dir, "temp_output"),
    ]
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
