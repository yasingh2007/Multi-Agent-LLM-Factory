"""Multi-agent training factory stages."""

from .collector import run as collect
from .cleaner import run as clean
from .labeler import run as label
from .trainer import run as train
from .evaluator import run as evaluate
from .deployer import run as deploy

__all__ = ["collect", "clean", "label", "train", "evaluate", "deploy"]
