"""Independent JSLLDA implementation for CISC3024 Assignment 3."""

from .core import JSLLDA, build_training_graph
from .data import cap_train_per_class, load_salinas, make_buffered_block_split, make_spatial_split
from .baseline import evaluate_svm_protocol, fit_classical_lda

__all__ = [
    "JSLLDA",
    "build_training_graph",
    "load_salinas",
    "make_spatial_split",
    "make_buffered_block_split",
    "cap_train_per_class",
    "fit_classical_lda",
    "evaluate_svm_protocol",
]
