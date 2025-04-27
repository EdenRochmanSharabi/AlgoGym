from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class BaseDataLoader(ABC):
    """Abstract base class for data loaders.

    Provides an interface for loading or generating training/evaluation data 
    (input points and optional target values).
    """

    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Loads or generates data samples.

        Returns:
            Tuple[np.ndarray, np.ndarray | None]: A tuple containing:
                - X: Input data samples (shape: (n_samples, input_dim)).
                - y: Target data samples (shape: (n_samples, output_dim)), 
                     or None if targets are not applicable (e.g., sampling domain).
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()" 