from abc import ABC, abstractmethod
from typing import Any, Dict
import numpy as np

from ..functions import BaseFunction

class BaseAlgorithm(ABC):
    """
    Abstract base class for function approximation algorithms.

    Defines the interface for algorithms that attempt to learn or approximate
    a target function (represented by a BaseFunction instance or data samples).
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Initializes the base algorithm.

        Args:
            config (Dict[str, Any] | None): Algorithm-specific configuration parameters.
        """
        self.config = config if config is not None else {}
        self._validate_config(self.config)
        self._approximated_function = None # Store the internal model/representation

    @abstractmethod
    def _validate_config(self, config: Dict[str, Any]):
        """Validates the algorithm-specific configuration."""
        pass

    @abstractmethod
    def train(self, target_function: BaseFunction | None = None, X_data: np.ndarray | None = None, y_data: np.ndarray | None = None):
        """
        Trains or runs the algorithm to approximate the target.

        Either `target_function` or both `X_data` and `y_data` must be provided.

        Args:
            target_function (BaseFunction | None): The target function instance to approximate.
                                                  Algorithms can sample this function.
            X_data (np.ndarray | None): Input data samples (shape: (n_samples, input_dim)).
            y_data (np.ndarray | None): Target data samples (shape: (n_samples, output_dim)).
        """
        pass
    
    @abstractmethod
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Performs a single epoch of training.
        
        This method is required for visualization and animation of the training process.
        It should update the algorithm's internal state and return metrics for the epoch.
        
        Args:
            epoch (int): The current epoch number (0-based).
            
        Returns:
            Dict[str, float]: A dictionary containing metrics for this epoch.
                Common metrics include 'loss', 'mse', 'mae', etc.
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output for given input points using the learned approximation.

        Args:
            x (np.ndarray): Input points (shape: (n_samples, input_dim) or (input_dim,)).

        Returns:
            np.ndarray: Predicted output values (shape: (n_samples, output_dim) or (output_dim,)).
        """
        pass
        
    @property
    def approximated_function(self):
        """Returns the internal representation of the approximated function, if applicable."""
        return self._approximated_function

    def __repr__(self) -> str:
        config_str = ", ".join(f"{k}={v!r}" for k, v in self.config.items())
        return f"{self.__class__.__name__}({config_str})" 