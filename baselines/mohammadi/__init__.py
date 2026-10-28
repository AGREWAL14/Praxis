"""Mohammadi et al. (2024), arXiv:2410.23306, reproduced without modification.

See README.md in this directory. Nothing here is tuned or corrected.
"""

from .cnn import ARCHITECTURE, COMPILE, FIT, build_model, describe, fit, reshape
from .data import (
    PREPROCESSING,
    attach_paths,
    encode_labels,
    fit_scaler,
    load_partition,
    read_block,
    scaler_statistics,
    transform_in_place,
)

__all__ = [
    "ARCHITECTURE",
    "COMPILE",
    "FIT",
    "PREPROCESSING",
    "attach_paths",
    "build_model",
    "describe",
    "encode_labels",
    "fit",
    "fit_scaler",
    "load_partition",
    "read_block",
    "reshape",
    "scaler_statistics",
    "transform_in_place",
]
