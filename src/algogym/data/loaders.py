import numpy as np
import pandas as pd
from typing import Tuple

from .base import BaseDataLoader
from algogym.functions import BaseFunction


class FunctionSampler(BaseDataLoader):
    """Generates data by sampling a BaseFunction within its domain."""
    def __init__(self, function: BaseFunction, n_samples: int):
        if not isinstance(function, BaseFunction):
            raise TypeError("function must be an instance of BaseFunction.")
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
            
        self.function = function
        self.n_samples = n_samples

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Samples the function's domain and calculates target values."""
        X = self.function.sample_domain(self.n_samples)
        y = self.function(X) # Calculate y values using the function
        
        # Ensure y is 2D (N, output_dim) even if output_dim is 1
        if self.function.output_dim == 1 and y.ndim == 1:
            y = y[:, np.newaxis]
            
        # Ensure X is 2D (N, input_dim) even if input_dim is 1
        if self.function.input_dim == 1 and X.ndim == 1:
            X = X[:, np.newaxis]
            
        return X, y

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(function={self.function!r}, n_samples={self.n_samples})"


class CsvLoader(BaseDataLoader):
    """Loads data from a CSV file."""
    def __init__(self, filepath: str, input_cols: list[str], target_cols: list[str] | None = None):
        self.filepath = filepath
        self.input_cols = input_cols
        self.target_cols = target_cols

    def load_data(self) -> Tuple[np.ndarray, np.ndarray | None]:
        """Reads the CSV and extracts input and target columns."""
        try:
            df = pd.read_csv(self.filepath)
        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found at: {self.filepath}")
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file {self.filepath}: {e}")

        missing_input = [col for col in self.input_cols if col not in df.columns]
        if missing_input:
            raise ValueError(f"Input columns not found in CSV: {missing_input}")
        
        X = df[self.input_cols].to_numpy()
        y = None
        
        if self.target_cols:
            missing_target = [col for col in self.target_cols if col not in df.columns]
            if missing_target:
                raise ValueError(f"Target columns not found in CSV: {missing_target}")
            y = df[self.target_cols].to_numpy()
            
        return X, y

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filepath='{self.filepath}', input_cols={self.input_cols}, target_cols={self.target_cols})" 