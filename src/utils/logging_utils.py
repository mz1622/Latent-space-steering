"""
Logging and experiment tracking utilities.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: int = logging.INFO,
    experiment_name: str = "experiment"
) -> logging.Logger:
    """
    Setup logging for experiments.

    Args:
        log_dir: Directory to save logs (if None, only console logging)
        log_level: Logging level
        experiment_name: Name of experiment for log file

    Returns:
        logger: Configured logger
    """
    logger = logging.getLogger("hallucination_mitigation")
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{experiment_name}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger


def create_experiment_dir(
    base_dir: str = "./outputs",
    experiment_name: str = "experiment"
) -> Path:
    """
    Create timestamped experiment directory.

    Args:
        base_dir: Base output directory
        experiment_name: Experiment name

    Returns:
        exp_dir: Path to experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / timestamp / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def save_metrics(metrics: dict, save_path: Path) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        save_path: Path to save to
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def save_environment_info(save_path: Path) -> None:
    """
    Save environment information.

    Args:
        save_path: Path to save environment.txt
    """
    import torch
    import platform

    info = []
    info.append(f"Python version: {sys.version}")
    info.append(f"Platform: {platform.platform()}")
    info.append(f"PyTorch version: {torch.__version__}")
    info.append(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        info.append(f"CUDA version: {torch.version.cuda}")
        info.append(f"GPU: {torch.cuda.get_device_name(0)}")

    with open(save_path, 'w') as f:
        f.write('\n'.join(info))


def save_git_info(save_path: Path) -> None:
    """
    Save git commit information.

    Args:
        save_path: Path to save git_commit.txt
    """
    import subprocess

    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()

        # Check for uncommitted changes
        status = subprocess.check_output(
            ['git', 'status', '--short'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()

        info = [f"Commit: {commit}"]
        if status:
            info.append("\nUncommitted changes detected:")
            info.append(status)

        with open(save_path, 'w') as f:
            f.write('\n'.join(info))
    except (subprocess.CalledProcessError, FileNotFoundError):
        with open(save_path, 'w') as f:
            f.write("Git information not available")
