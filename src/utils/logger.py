import logging
import os
from typing import Dict, Union

import torch
from src.utils.path import read_config


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


def log_metrics(results: Dict[str, Union[float, torch.Tensor]], logger: logging.Logger) -> None:
    """
    Logs evaluation metrics for semantic segmentation.

    This function prints overall metrics (e.g., pixel accuracy, mean accuracy, mean IoU, mean Dice)
    and detailed per-class metrics including IoU, Dice (F1), precision, and recall.

    Args:
        results (Dict[str, Union[float, torch.Tensor]]):
            A dictionary containing evaluation results. Expected keys:
                - 'Pixel_Accuracy' (float)
                - 'Mean_Accuracy' (float)
                - 'mIoU' (float): Mean Intersection over Union.
                - 'mDice' (float): Mean Dice score (equivalent to F1).
                - 'IoU_per_class' (torch.Tensor): IoU scores for each class.
                - 'Dice_per_class' (torch.Tensor): Dice scores for each class.
                - 'Precision_per_class' (torch.Tensor): Precision scores for each class.
                - 'Recall_per_class' (torch.Tensor): Recall scores for each class.
        logger (logging.Logger):
            The logger instance used to write the output to a file or console.

    Returns:
        None
    """
    cfg = read_config()

    logger.info("Evaluation Results:")
    logger.info(f"{'Pixel Accuracy:':<25}{results['Pixel_Accuracy']:.4f}")
    logger.info(f"{'Mean Accuracy:':<25}{results['Mean_Accuracy']:.4f}")
    logger.info(f"{'Mean IoU:':<25}{results['mIoU']:.4f}")
    logger.info(f"{'Mean Dice (F1):':<25}{results['mDice']:.4f}")
    logger.info("\nPer-Class Evaluation Metrics:")
    logger.info(f"{'Class':<18}{'IoU':>10}{'Dice':>10}{'Prec.':>10}{'Recall':>10}")
    logger.info("-" * 58)

    for idx, (iou, dice, prec, recall) in enumerate(zip(
        results['IoU_per_class'],
        results['Dice_per_class'],
        results['Precision_per_class'],
        results['Recall_per_class']
    )):
        class_name = cfg['class_distribution']['id_to_class'].get(idx, f"Class {idx}")
        logger.info(f"{class_name:<18}{iou.item():>10.4f}{dice.item():>10.4f}{prec.item():>10.4f}{recall.item():>10.4f}")
