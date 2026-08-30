"""
Plotting Utilities Subpackage
Consolidated module for visualization, metrics plotting, and preprocessing plots.
Provides clear entry points for all plotting functionality.
"""

from .plotting import (
    plot_results,
    fbid_plot,
    plot_fbid_trends,
    visualize_together,
    visualize_prediction,
    plot_metrics_mosaic,
    plot_metrics_mosaic_v2,
    plot_preprocessing,
    plot_metric_violin,
    plot_fragmentation_verification,
)

__all__ = [
    "plot_results",
    "fbid_plot",
    "plot_fbid_trends",
    "visualize_together",
    "visualize_prediction",
    "plot_metrics_mosaic",
    "plot_metrics_mosaic_v2",
    "plot_preprocessing",
    "plot_metric_violin",
    "plot_fragmentation_verification",
]
