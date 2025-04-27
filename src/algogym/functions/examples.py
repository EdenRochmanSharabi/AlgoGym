import numpy as np
from typing import Union

from .base import BaseFunction


class SineFunction(BaseFunction):
    """A simple 1D Sine function: f(x) = sin(x)."""
    def __init__(self, domain: tuple = (-2 * np.pi, 2 * np.pi)):
        super().__init__(input_dim=1, output_dim=1, domain=domain)

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # Validate input shape and domain
        x_validated = self._validate_input(x) # Returns shape (N, 1)
        
        # Calculate function value
        result_batch = np.sin(x_validated) # Operate directly on (N, 1) array -> (N, 1)
        result_flat = result_batch.flatten() # Shape (N,)
        
        # Return appropriate shape based on original input
        if isinstance(x, (int, float)) or (isinstance(x, np.ndarray) and x.ndim == 0):
            # Input was scalar or 0D array
            return result_flat[0] # Return scalar
        elif isinstance(x, np.ndarray) and x.ndim == 1:
            # Input was 1D array (N,)
            return result_flat # Return 1D array (N,)
        else: # Should not happen for 1D function
            return result_batch # Should be unreachable, but return (N,1) just in case


class PolynomialFunction(BaseFunction):
    """A simple 1D Polynomial function: f(x) = ax^2 + bx + c."""
    def __init__(self, a: float = 1.0, b: float = -2.0, c: float = 1.0, domain: tuple = (-5.0, 5.0)):
        self.a = a
        self.b = b
        self.c = c
        super().__init__(input_dim=1, output_dim=1, domain=domain)

    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        x_validated = self._validate_input(x) # Returns shape (N, 1)
        result_batch = self.a * (x_validated**2) + self.b * x_validated + self.c # Operates element-wise -> (N, 1)
        result_flat = result_batch.flatten() # Shape (N,)
        
        # Return appropriate shape based on original input
        if isinstance(x, (int, float)) or (isinstance(x, np.ndarray) and x.ndim == 0):
            # Input was scalar or 0D array
            return result_flat[0] # Return scalar
        elif isinstance(x, np.ndarray) and x.ndim == 1:
            # Input was 1D array (N,)
            return result_flat # Return 1D array (N,)
        else: # Should not happen for 1D function
            return result_batch # Should be unreachable, but return (N,1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(a={self.a}, b={self.b}, c={self.c}, domain={self.domain})"


class RosenbrockFunction(BaseFunction):
    """
    The Rosenbrock function (2D or higher).
    A classic optimization benchmark.
    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (1 - x_i)^2) for i = 0 to D-2
    Minimum is at (1, 1, ..., 1) where f(x) = 0.
    """
    def __init__(self, input_dim: int = 2, domain_scale: float = 2.0):
        if input_dim < 2:
            raise ValueError("Rosenbrock function requires input_dim >= 2.")
        
        # Define domain typically around the minimum (1, 1, ...)
        lower_bounds = np.ones(input_dim) * -domain_scale
        upper_bounds = np.ones(input_dim) * domain_scale
        domain = (lower_bounds, upper_bounds)
        
        super().__init__(input_dim=input_dim, output_dim=1, domain=domain)

    def __call__(self, x: np.ndarray) -> Union[float, np.ndarray]:
        x_validated = self._validate_input(x) # Returns shape (N, D)
        
        # Calculate Rosenbrock value for each point in the batch
        results = np.zeros(x_validated.shape[0])
        for i in range(self.input_dim - 1):
            results += 100.0 * (x_validated[:, i+1] - x_validated[:, i]**2)**2 + (1.0 - x_validated[:, i])**2
        
        # Return appropriate shape based on original input type
        # Original input x: (D,) or (N, D)
        if isinstance(x, np.ndarray) and x.ndim < self.input_dim: 
            # Input was single point (D,) - check ndim < input_dim
            return results[0] # Return scalar
        else: # Input was batch (N, D)
            return results # Return array (N,)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, domain={self.domain})" 