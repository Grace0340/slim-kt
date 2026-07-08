"""Seeding, logging, device selection, and simple timers."""
from __future__ import annotations

import logging
import os
import random
import time
from contextlib import contextmanager

import numpy as np

try:
    import torch
except Exception:  # torch may be absent when only inspecting configs
    torch = None  # type: ignore


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # deterministic-ish; keep cudnn benchmark off for reproducibility
        torch.backends.cudnn.benchmark = False


def get_device() -> "str":
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_logger(name: str = "slimkt", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(level)
    return logger


@contextmanager
def timer(name: str, logger: logging.Logger | None = None):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    msg = f"{name} took {dt:.3f}s"
    (logger.info(msg) if logger else print(msg))


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
