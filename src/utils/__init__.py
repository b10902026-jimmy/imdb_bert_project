"""Utility functions module.

This module contains utility functions for configuration loading, logging,
metrics calculation, visualization, and other helper functions used throughout the project.
"""

from src.utils.config import (
    load_config,
    get_project_root,
    get_config_path,
    load_env_vars,
    get_env_var,
    merge_configs,
    load_all_configs,
)

from src.utils.metrics import (
    compute_metrics,
    log_metrics,
    get_classification_report,
    get_confusion_matrix,
)

from src.utils.visualization import (
    set_plotting_style,
    save_figure,
    plot_review_length_distribution,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    generate_wordcloud,
    plot_attention_weights,
    plot_embedding_projection,
)

__all__ = [
    # Config utilities
    "load_config",
    "get_project_root",
    "get_config_path",
    "load_env_vars",
    "get_env_var",
    "merge_configs",
    "load_all_configs",
    
    # Metrics utilities
    "compute_metrics",
    "log_metrics",
    "get_classification_report",
    "get_confusion_matrix",
    
    # Visualization utilities
    "set_plotting_style",
    "save_figure",
    "plot_review_length_distribution",
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_training_history",
    "generate_wordcloud",
    "plot_attention_weights",
    "plot_embedding_projection",
]
