"""
Audio Analysis Package
Consolidated module for contour extraction, evaluation, and plotting.
"""
# Workaround to avoid plotting conflicts with matlab on venvs
import xml.parsers.expat

from . import utils
from . import augmentation
from . import evaluation
from . import plotting
from . import contour_extraction

__all__ = [
    "utils",
    "augmentation",
    "evaluation",
    "plotting",
    "contour_extraction",
]
