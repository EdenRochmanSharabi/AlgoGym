import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

from algogym.functions import BaseFunction
from algogym.algorithms import BaseAlgorithm


class BaseVisualizer(ABC):
    """
    Abstract base class for visualization components in AlgoGym.
    
    Provides interfaces for visualizing function approximations, algorithm
    progress, and evaluation results.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the visualizer with optional configuration parameters.
        
        Args:
            config (Dict[str, Any], optional): Configuration parameters for the visualizer.
        """
        self.config = config if config is not None else {}
        self._validate_config()
        
    def _validate_config(self) -> None:
        """
        Validate the configuration parameters. Can be overridden by subclasses
        to add specific validation rules.
        """
        pass
    
    @abstractmethod
    def visualize_function(self, 
                          function: BaseFunction,
                          approximation: Optional[BaseAlgorithm] = None,
                          points: Optional[np.ndarray] = None,
                          values: Optional[np.ndarray] = None,
                          title: Optional[str] = None,
                          show: bool = True,
                          save_path: Optional[str] = None) -> Any:
        """
        Visualize a function and its approximation.
        
        Args:
            function (BaseFunction): The target function to visualize.
            approximation (BaseAlgorithm, optional): The algorithm that approximates the function.
            points (np.ndarray, optional): Specific input points to visualize.
            values (np.ndarray, optional): Output values corresponding to the input points.
            title (str, optional): Title for the visualization.
            show (bool, optional): Whether to display the visualization. Default is True.
            save_path (str, optional): Path to save the visualization. If None, the 
                                      visualization is not saved.
                                      
        Returns:
            Any: Visualization object or identifier.
        """
        pass
    
    @abstractmethod
    def visualize_progress(self,
                          metrics_history: Dict[str, List[float]],
                          iterations: Optional[List[int]] = None,
                          title: Optional[str] = None,
                          show: bool = True,
                          save_path: Optional[str] = None) -> Any:
        """
        Visualize the progress of an algorithm over iterations.
        
        Args:
            metrics_history (Dict[str, List[float]]): Dictionary mapping metric names to 
                                                     lists of metric values over iterations.
            iterations (List[int], optional): List of iteration numbers corresponding to the metrics.
            title (str, optional): Title for the visualization.
            show (bool, optional): Whether to display the visualization. Default is True.
            save_path (str, optional): Path to save the visualization. If None, the 
                                      visualization is not saved.
                                      
        Returns:
            Any: Visualization object or identifier.
        """
        pass
    
    @abstractmethod
    def visualize_comparison(self,
                            functions: List[BaseFunction],
                            approximations: List[BaseAlgorithm],
                            labels: Optional[List[str]] = None,
                            title: Optional[str] = None,
                            show: bool = True,
                            save_path: Optional[str] = None) -> Any:
        """
        Visualize and compare multiple function approximations.
        
        Args:
            functions (List[BaseFunction]): List of target functions to visualize.
            approximations (List[BaseAlgorithm]): List of algorithms that approximate the functions.
            labels (List[str], optional): Labels for each function-approximation pair.
            title (str, optional): Title for the visualization.
            show (bool, optional): Whether to display the visualization. Default is True.
            save_path (str, optional): Path to save the visualization. If None, the 
                                      visualization is not saved.
                                      
        Returns:
            Any: Visualization object or identifier.
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the visualizer."""
        return f"{self.__class__.__name__}(config={self.config})" 