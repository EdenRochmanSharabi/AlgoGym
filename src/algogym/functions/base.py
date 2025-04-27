import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple

class BaseFunction(ABC):
    """
    Abstract base class for functions to be approximated by algorithms.

    This class defines the interface that all target functions must implement.
    It includes properties for input and output dimensions and an abstract
    `__call__` method for function evaluation.
    """

    def __init__(self, input_dim: int, output_dim: int, domain: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]):
        """
        Initializes the BaseFunction.

        Args:
            input_dim (int): The dimensionality of the input space.
            output_dim (int): The dimensionality of the output space.
            domain (Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]): 
                A tuple representing the lower and upper bounds of the function's domain.
                For 1D, it's (min_x, max_x).
                For multi-dimensional, it can be ((min_x1, min_x2,...), (max_x1, max_x2,...)).
        """
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError("Input dimension must be a positive integer.")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError("Output dimension must be a positive integer.")
        
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._domain = self._validate_domain(domain, input_dim)

    @property
    def input_dim(self) -> int:
        """Returns the dimensionality of the input space."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """Returns the dimensionality of the output space."""
        return self._output_dim
        
    @property
    def domain(self) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """Returns the domain (lower_bound, upper_bound) of the function."""
        return self._domain

    def _validate_domain(self, domain: Tuple, input_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Validates the provided domain."""
        if not isinstance(domain, tuple) or len(domain) != 2:
            raise ValueError("Domain must be a tuple of length 2 (lower_bound, upper_bound).")
        
        lower_bound, upper_bound = domain
        
        if input_dim == 1:
            if not isinstance(lower_bound, (int, float)) or not isinstance(upper_bound, (int, float)):
                 raise ValueError("For 1D input, domain bounds must be scalars.")
            if lower_bound >= upper_bound:
                 raise ValueError("Lower bound must be less than upper bound.")
            return (np.array([lower_bound]), np.array([upper_bound]))
        else:
            lower_bound = np.asarray(lower_bound, dtype=float)
            upper_bound = np.asarray(upper_bound, dtype=float)
            if lower_bound.shape != (input_dim,) or upper_bound.shape != (input_dim,):
                raise ValueError(f"Domain bounds must have shape ({input_dim},) for multi-dimensional input.")
            if np.any(lower_bound >= upper_bound):
                raise ValueError("All lower bounds must be strictly less than their corresponding upper bounds.")
            return (lower_bound, upper_bound)

    @abstractmethod
    def __call__(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluates the function at a given point or batch of points.

        Args:
            x (Union[float, np.ndarray]): The input point(s). 
               - If input_dim is 1, x can be a float or a 1D NumPy array (N,).
               - If input_dim > 1, x must be a NumPy array of shape (input_dim,) or (N, input_dim).

        Returns:
            Union[float, np.ndarray]: The corresponding output value(s).
               - If output_dim is 1 and x was a single point, returns a float.
               - Otherwise, returns a NumPy array of shape (output_dim,) or (N, output_dim).
        
        Raises:
            ValueError: If the input `x` has an incorrect shape or is outside the defined domain.
        """
        pass

    def _validate_input(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Validates the input x against the function's dimension and domain."""
        if self.input_dim == 1:
            if isinstance(x, (int, float)):
                x_arr = np.array([x]) # Single point, make it (1, 1) temporarily for validation
            elif isinstance(x, np.ndarray):
                 if x.ndim == 0: # Scalar numpy value
                     x_arr = x.reshape(1, 1)
                 elif x.ndim == 1: # Batch of points (N,) -> (N, 1)
                     x_arr = x[:, np.newaxis]
                 else:
                    raise ValueError(f"Input array for 1D function must be 1D (N,), got {x.ndim}D.")
            else:
                raise TypeError("Input for 1D function must be float or 1D NumPy array.")
        elif isinstance(x, np.ndarray):
            if x.ndim == 1: # Single point (D,) -> (1, D)
                if x.shape[0] != self.input_dim:
                     raise ValueError(f"Input point has wrong dimension. Expected {self.input_dim}, got {x.shape[0]}.")
                x_arr = x[np.newaxis, :]
            elif x.ndim == 2: # Batch of points (N, D)
                if x.shape[1] != self.input_dim:
                    raise ValueError(f"Input points have wrong dimension. Expected {self.input_dim}, got {x.shape[1]}.")
                x_arr = x
            else:
                raise ValueError(f"Input array must be 1D (D,) or 2D (N, D), got {x.ndim}D.")
        else:
             raise TypeError("Input for multi-dimensional function must be a NumPy array.")
             
        # Domain check
        lower_bound, upper_bound = self.domain
        if np.any(x_arr < lower_bound) or np.any(x_arr > upper_bound):
            # Find the first violating point and dimension for a clearer error message
            violating_indices = np.where((x_arr < lower_bound) | (x_arr > upper_bound))
            first_violating_point_idx = violating_indices[0][0]
            
            if self.input_dim == 1:
                # For 1D, x_arr is (N, 1), violating_indices is (array([idx]), array([0]))
                first_violating_dim_idx = 0 # Dimension is always 0
                # Check ndim before indexing. x_arr SHOULD be (N, 1) here.
                if x_arr.ndim == 2:
                    violating_value = x_arr[first_violating_point_idx, 0]
                elif x_arr.ndim == 1: # Should not happen ideally, but handle just in case
                    violating_value = x_arr[first_violating_point_idx]
                else: # Should definitely not happen
                    raise RuntimeError(f"Unexpected x_arr shape {x_arr.shape} in domain check")
                violating_point_repr = violating_value # Use scalar value for message
                lower_b_val = lower_bound[0]
                upper_b_val = upper_bound[0]
            else:
                # For >1D, x_arr is (N, D), violating_indices is (array([idx,...]), array([dim,...]))
                first_violating_dim_idx = violating_indices[1][0]
                violating_value = x_arr[first_violating_point_idx, first_violating_dim_idx]
                violating_point_repr = x_arr[first_violating_point_idx] # Get full point for message
                lower_b_val = lower_bound[first_violating_dim_idx]
                upper_b_val = upper_bound[first_violating_dim_idx]
                
            raise ValueError(
                f"Input point {violating_point_repr} at index {first_violating_point_idx} "
                f"is outside the domain [{lower_bound}, {upper_bound}]. " # Keep original bounds repr
                f"Violation at dimension {first_violating_dim_idx}: value {violating_value} "
                f"is outside [{lower_b_val}, {upper_b_val}]"
            )

        # Return original shape if it was a single point for convenience in subclasses
        # Always return 2D (N, D) internally for consistency
        # Subclasses can handle reshaping back if needed based on original input type.
        # if isinstance(x, (int, float)) or (isinstance(x, np.ndarray) and x.ndim <= 1):
        #     return x_arr.reshape(-1, self.input_dim) # Ensure 2D shape (N, D) even if N=1

        return x_arr # Return as (N, D)

    def sample_domain(self, n_samples: int) -> np.ndarray:
        """
        Generates random samples uniformly from the function's domain.

        Args:
            n_samples (int): The number of samples to generate.

        Returns:
            np.ndarray: An array of shape (n_samples, input_dim) containing the samples.
        """
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("Number of samples must be a positive integer.")
            
        lower_bound, upper_bound = self.domain
        samples = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_samples, self.input_dim))
        
        if self.input_dim == 1:
            return samples.flatten() # Return as (N,) for 1D case
        else:
            return samples # Return as (N, D)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(input_dim={self.input_dim}, output_dim={self.output_dim}, domain={self.domain})" 