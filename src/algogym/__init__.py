"""
AlgoGym - A Python framework for experimenting with function approximation algorithms.

AlgoGym provides tools for defining target functions, implementing approximation
algorithms, and evaluating their performance.
"""

# Main AlgoGym package 

# Expose key components at the top level for easier access
from .functions import BaseFunction, SineFunction, PolynomialFunction, RosenbrockFunction
from .algorithms import BaseAlgorithm, GeneticAlgorithm
from .data import BaseDataLoader, FunctionSampler, CsvLoader
from .evaluation import mean_squared_error, mean_absolute_error
from .core import ExperimentEngine
from .visualization import BaseVisualizer, FunctionVisualizer

try:
    from .visualization import InteractiveDashboard
    __dashboard_available__ = True
except ImportError:
    __dashboard_available__ = False

try:
    from .visualization import TrainingAnimator, TrainingRecorder
    __animation_available__ = True
except ImportError:
    __animation_available__ = False

__all__ = [
    # Core
    "ExperimentEngine",
    # Functions
    "BaseFunction",
    "SineFunction",
    "PolynomialFunction",
    "RosenbrockFunction",
    # Data
    "BaseDataLoader",
    "FunctionSampler",
    "CsvLoader",
    # Algorithms
    "BaseAlgorithm",
    "GeneticAlgorithm", 
    # Evaluation
    "mean_squared_error",
    "mean_absolute_error",
    # Visualization
    "BaseVisualizer",
    "FunctionVisualizer",
]

# Add dashboard to __all__ if available
if __dashboard_available__:
    __all__.append("InteractiveDashboard")

# Add animation components to __all__ if available
if __animation_available__:
    __all__.extend(["TrainingAnimator", "TrainingRecorder"])

__version__ = "0.1.0" 