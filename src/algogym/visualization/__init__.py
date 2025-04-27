# Visualization module for AlgoGym

from .base import BaseVisualizer
from .function_viz import FunctionVisualizer

try:
    from .dashboard import InteractiveDashboard
    __all__ = ["BaseVisualizer", "FunctionVisualizer", "InteractiveDashboard"]
except ImportError:
    # Optional dependency ipywidgets may not be available
    __all__ = ["BaseVisualizer", "FunctionVisualizer"]

try:
    from .animation import TrainingAnimator, TrainingRecorder
    __all__ += ["TrainingAnimator", "TrainingRecorder"]
except ImportError:
    # Optional dependency imageio may not be available
    pass 