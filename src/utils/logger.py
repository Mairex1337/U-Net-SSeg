import logging
import os


def get_logger(run_dir: str, file_name: str) -> logging.Logger:
    """
    Sets up a logger that logs to both console and a file in the run directory.

    Args:
        run_dir (str): directory of the training run.

    Returns:
        logging.Logger: logger object
    """
    log_file = os.path.join(run_dir, file_name)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()
    return logger
