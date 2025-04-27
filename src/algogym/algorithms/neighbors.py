import numpy as np
from typing import Any, Dict
from scipy.spatial.distance import cdist

from .base import BaseAlgorithm
from algogym.functions import BaseFunction # For type hints

class KNearestNeighbors(BaseAlgorithm):
    """
    k-Nearest Neighbors (k-NN) algorithm for function approximation (regression).
    
    This is a non-parametric method that stores the training data and predicts 
    the output for a new point based on the average output of its k closest 
    neighbors in the training set.
    """
    DEFAULT_CONFIG = {
        "k": 3, # Number of neighbors
        "metric": "euclidean" # Distance metric (from scipy.spatial.distance.cdist)
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        merged_config = {**self.DEFAULT_CONFIG, **(config if config is not None else {})}
        super().__init__(merged_config)
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None
        self._input_dim: int | None = None
        self._output_dim: int | None = None
        self._full_X_data: np.ndarray | None = None
        self._full_y_data: np.ndarray | None = None
        self._current_train_size: int = 0
        self._target_function: BaseFunction | None = None

    def _validate_config(self, config: Dict[str, Any]):
        if "k" not in config or not isinstance(config["k"], int) or config["k"] <= 0:
            raise ValueError("'k' must be a positive integer.")
        # `metric` validation relies on scipy.spatial.distance.cdist handling it.

    def train(self, target_function: BaseFunction | None = None, X_data: np.ndarray | None = None, y_data: np.ndarray | None = None):
        """
        Stores the training data (X_data, y_data). k-NN is a lazy learner.
        
        Requires X_data and y_data.
        """
        if X_data is None or y_data is None:
             # k-NN fundamentally requires labeled data points to store.
             # We could sample a target_function, but it aligns better with 
             # the concept of k-NN to require explicit data.
            raise ValueError("KNearestNeighbors requires explicit X_data and y_data for training.")

        # Store target function for incremental training
        self._target_function = target_function

        # Ensure data is 2D
        if X_data.ndim == 1:
             X_data = X_data[:, np.newaxis]
        if y_data.ndim == 1:
             y_data = y_data[:, np.newaxis]

        if X_data.shape[0] != y_data.shape[0]:
            raise ValueError(f"Number of samples mismatch between X_data ({X_data.shape[0]}) and y_data ({y_data.shape[0]}).")
            
        k = self.config["k"]
        if k > X_data.shape[0]:
             print(f"Warning: k ({k}) is greater than the number of training samples ({X_data.shape[0]}). Setting k to {X_data.shape[0]}.")
             self.config["k"] = X_data.shape[0]
        
        # Store the full dataset for incremental training
        self._full_X_data = X_data
        self._full_y_data = y_data
        self._input_dim = X_data.shape[1]
        self._output_dim = y_data.shape[1]
        self._current_train_size = 0
        
        # Initialize with empty arrays
        self._X_train = np.empty((0, self._input_dim))
        self._y_train = np.empty((0, self._output_dim))
        
        # The internal "approximated function" is just the stored data for k-NN
        self._approximated_function = (self._X_train, self._y_train) 
        print(f"k-NN initialized for incremental training with {X_data.shape[0]} total data points.")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Simulates training progress for k-NN by incrementally adding data points.
        
        For k-NN, each epoch adds more training points until all data is used.
        
        Args:
            epoch (int): The current epoch number.
            
        Returns:
            Dict[str, float]: Dictionary containing metrics for this epoch.
        """
        if self._full_X_data is None or self._full_y_data is None:
            raise RuntimeError("The train method must be called before train_epoch")
        
        # Calculate how many samples to add in this epoch
        total_samples = len(self._full_X_data)
        max_epochs = 20  # Cap the number of epochs to ensure reasonable data increments
        samples_per_epoch = max(1, total_samples // max_epochs)
        
        # Calculate the new number of samples to use
        new_size = min(self._current_train_size + samples_per_epoch, total_samples)
        
        # Update the training data with more samples
        self._X_train = self._full_X_data[:new_size].copy()
        self._y_train = self._full_y_data[:new_size].copy()
        added_samples = new_size - self._current_train_size
        self._current_train_size = new_size
        
        # Update the internal model
        self._approximated_function = (self._X_train, self._y_train)
        
        # Calculate metrics by evaluating on a test set
        metrics = {}
        metrics["samples_used"] = float(self._current_train_size)
        metrics["samples_added"] = float(added_samples)
        metrics["samples_percent"] = float(self._current_train_size / total_samples * 100)
        
        # If we have a target function, evaluate performance
        if self._target_function is not None:
            # Generate test points
            test_samples = 100
            if self._input_dim == 1:
                lower, upper = self._target_function.domain
                test_x = np.linspace(lower[0], upper[0], test_samples).reshape(-1, 1)
            else:
                # For higher dimensions, sample points from the domain
                test_x = self._target_function.sample_domain(test_samples)
                if test_x.ndim == 1:
                    test_x = test_x.reshape(-1, 1)
            
            # Get true values
            test_y_true = self._target_function(test_x)
            if test_y_true.ndim == 1:
                test_y_true = test_y_true.reshape(-1, 1)
            
            # Only predict if we have enough training samples
            if len(self._X_train) >= self.config["k"]:
                # Get predictions
                test_y_pred = self.predict(test_x)
                
                # Calculate error metrics
                mse = np.mean((test_y_true - test_y_pred) ** 2)
                mae = np.mean(np.abs(test_y_true - test_y_pred))
                
                metrics["mse"] = float(mse)
                metrics["mae"] = float(mae)
                metrics["rmse"] = float(np.sqrt(mse))
            else:
                # Not enough samples for prediction yet
                metrics["mse"] = float('nan')
                metrics["mae"] = float('nan')
                metrics["rmse"] = float('nan')
        
        # Add k value and other metadata
        metrics["k"] = float(min(self.config["k"], self._current_train_size))
        metrics["epoch"] = epoch
        
        return metrics

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts the output for input points x by finding the k nearest neighbors 
        in the stored training data.
        
        Args:
            x: Input points to predict, either a 1D array representing a single point,
               or a 2D array representing multiple points.
               
        Returns:
            A 2D numpy array of shape (n_samples, output_dim) containing predictions.
        """
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("The k-NN algorithm has not been trained (data not stored). Call train() first.")

        # Ensure input x is 2D (n_predict_samples, input_dim)
        if x.ndim == 1:
            if self._input_dim == 1:
                x_proc = x.reshape(-1, 1) # Handle 1D input array (N,) -> (N, 1)
            else:
                if x.shape[0] != self._input_dim:
                    raise ValueError(f"Input point dimension mismatch. Expected {self._input_dim}, got {x.shape[0]}")
                x_proc = x[np.newaxis, :] # Single point (D,) -> (1, D)
        elif x.ndim == 2:
            if x.shape[1] != self._input_dim:
                raise ValueError(f"Input batch dimension mismatch. Expected {self._input_dim}, got {x.shape[1]}")
            x_proc = x # Batch (N, D)
        else:
            raise ValueError("Input must be 1D or 2D array.")
            
        k = min(self.config["k"], len(self._X_train))  # Ensure k doesn't exceed available samples
        metric = self.config["metric"]
        n_predict_samples = x_proc.shape[0]
        predictions = np.zeros((n_predict_samples, self._output_dim))

        # Calculate distances between each prediction point and all training points
        # Shape: (n_predict_samples, n_train_samples)
        distances = cdist(x_proc, self._X_train, metric=metric)

        for i in range(n_predict_samples):
            # For each prediction point, we need to handle ties properly
            # Using a different approach: sort points by distance AND position relative to query point
            if self._input_dim == 1:
                # For 1D data, break ties by selecting points symmetrically around the query point
                query_point = x_proc[i, 0]
                
                # Create a custom sorting key that considers both distance and position
                # 1. Primary sort by distance (smaller first)
                # 2. Secondary sort by absolute position difference from query (smaller first)
                # 3. If still tied, prefer larger values (higher indices in original array)
                sortkey = [(distances[i, j], abs(self._X_train[j, 0] - query_point), -self._X_train[j, 0]) 
                          for j in range(len(self._X_train))]
                sorted_indices = sorted(range(len(sortkey)), key=lambda j: sortkey[j])
                
                k_nearest_indices = sorted_indices[:k]
            else:
                # For multi-dimensional data, just use the distance with stable sort
                k_nearest_indices = np.argsort(distances[i], kind='stable')[:k]
            
            neighbor_outputs = self._y_train[k_nearest_indices]
            predictions[i] = np.mean(neighbor_outputs, axis=0)
            
        # Always return 2D array with shape (n_samples, output_dim)
        return predictions 