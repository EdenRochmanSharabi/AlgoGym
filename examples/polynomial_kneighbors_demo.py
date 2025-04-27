#!/usr/bin/env python3
"""
Simple demonstration of learning a polynomial function with k-neighbors visualization.

This script shows how a polynomial function is learned over time using K-Nearest Neighbors
and creates a GIF animation of the learning process.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import time
from sklearn.neighbors import KNeighborsRegressor

# Add the project root to the Python path
script_path = Path(__file__).parent.absolute()
project_root = script_path.parent
sys.path.insert(0, str(project_root))

from algogym.functions import BaseFunction
from algogym.algorithms import BaseAlgorithm
from algogym.visualization import FunctionVisualizer
try:
    import imageio.v2 as imageio  # Explicitly import imageio.v2
    from algogym.visualization import TrainingAnimator, TrainingRecorder
    ANIMATION_AVAILABLE = True
except ImportError:
    ANIMATION_AVAILABLE = False
    print("Animation components not available. Install imageio to enable.")

class PolynomialFunction(BaseFunction):
    """Polynomial function with configurable coefficients."""
    
    def __init__(self, coefficients=None):
        super().__init__(input_dim=1, output_dim=1, domain=(-1.0, 1.0))
        self.coefficients = coefficients or [0, 0, 1]  # Default to x^2
    
    def __call__(self, x):
        # Handle 2D input with shape (N, 1) by flattening it before validation
        if isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == 1:
            x_flat = x.flatten()
            x_val = self._validate_input(x_flat)
        else:
            x_val = self._validate_input(x)
            
        # Evaluate polynomial: c[0] + c[1]*x + c[2]*x^2 + ...
        result = np.zeros_like(x_val)
        for i, coef in enumerate(self.coefficients):
            result += coef * (x_val ** i)
            
        return result

class KNeighborsAlgorithm(BaseAlgorithm):
    """
    Simple algorithm that uses KNN to approximate a polynomial function.
    """
    
    def __init__(self, config=None):
        default_config = {
            'n_neighbors': 5,
            'weights': 'uniform',  # Options: 'uniform', 'distance'
            'sample_size': 10,  # Initial sample size
            'max_samples': 100,  # Maximum number of samples to collect
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(default_config)
        
        # Initialize model and data storage
        self.model = KNeighborsRegressor(
            n_neighbors=self.config['n_neighbors'],
            weights=self.config['weights']
        )
        self.X_samples = None
        self.y_samples = None
    
    def _validate_config(self, config):
        """Validate the configuration."""
        pass
    
    def train(self, target_function=None, X_data=None, y_data=None):
        """Initialize the training process."""
        if not isinstance(target_function, PolynomialFunction):
            raise ValueError("This algorithm only works with PolynomialFunction")
        
        # Store the target function
        self._target_function = target_function
        
        # Initial sampling at random points
        lower_bound, upper_bound = target_function.domain
        initial_samples = np.random.uniform(
            low=lower_bound[0],
            high=upper_bound[0],
            size=(self.config['sample_size'], 1)
        )
        
        # Get target values
        initial_values = target_function(initial_samples)
        
        # Store samples
        self.X_samples = initial_samples
        self.y_samples = initial_values
        
        # Fit initial model
        self.model.fit(self.X_samples, self.y_samples)
    
    def train_epoch(self, epoch):
        """Train for one epoch by adding more samples at strategic points."""
        # We'll add points where the error is highest
        
        # Generate test points across the domain
        lower_bound, upper_bound = self._target_function.domain
        test_points = np.linspace(lower_bound[0], upper_bound[0], 200).reshape(-1, 1)
        
        # Evaluate true function and current approximation
        true_values = self._target_function(test_points)
        pred_values = self.predict(test_points)
        
        # Calculate error at each point
        errors = np.abs(true_values - pred_values)
        
        # Find the point with highest error
        max_error_idx = np.argmax(errors)
        new_sample_x = test_points[max_error_idx]
        new_sample_y = true_values[max_error_idx]
        
        # Add the new sample with some noise to prevent overfitting
        if len(self.X_samples) < self.config['max_samples']:
            noise_x = np.random.normal(0, 0.01)  # Small position noise
            noise_y = np.random.normal(0, 0.01)  # Small value noise
            
            self.X_samples = np.vstack([self.X_samples, new_sample_x + noise_x])
            self.y_samples = np.vstack([self.y_samples, new_sample_y + noise_y])
            
            # Retrain the model with the new samples
            self.model.fit(self.X_samples, self.y_samples)
        
        # Calculate metrics on test points
        mse = np.mean((true_values - pred_values) ** 2)
        mae = np.mean(np.abs(true_values - pred_values))
        
        # Estimate coefficients for reporting (this is just a rough approximation)
        if epoch > 0 and epoch % 5 == 0:
            # Use polynomial regression to estimate coefficients
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            
            # Degree of polynomial = number of coefficients - 1
            degree = len(self._target_function.coefficients) - 1
            
            # Create polynomial features
            poly_features = PolynomialFeatures(degree=degree, include_bias=True)
            X_poly = poly_features.fit_transform(self.X_samples)
            
            # Fit linear regression
            lr_model = LinearRegression()
            lr_model.fit(X_poly, self.y_samples)
            
            # Extract coefficients
            estimated_coeffs = lr_model.coef_[0]
            if len(estimated_coeffs) < len(self._target_function.coefficients):
                # Pad with zeros if needed
                estimated_coeffs = np.pad(
                    estimated_coeffs,
                    (0, len(self._target_function.coefficients) - len(estimated_coeffs)),
                    'constant'
                )
        else:
            # Just repeat last estimation or use zeros
            if hasattr(self, 'estimated_coeffs'):
                estimated_coeffs = self.estimated_coeffs
            else:
                estimated_coeffs = np.zeros_like(self._target_function.coefficients)
                
        # Store coefficients for next time
        self.estimated_coeffs = estimated_coeffs
        
        return {
            'MSE': float(mse),
            'MAE': float(mae),
            'SampleCount': len(self.X_samples),
            'MaxError': float(np.max(errors)),
            'Coefficients': estimated_coeffs  # This is just an approximation
        }
    
    def predict(self, x):
        """Predict using the KNN model."""
        if self.X_samples is None or self.y_samples is None:
            raise RuntimeError("Model not trained. Call train() first.")
            
        # Reshape input if needed
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        # Make predictions
        return self.model.predict(x).reshape(-1, 1)

def run_demo():
    """Run the polynomial learning demonstration with k-neighbors."""
    if not ANIMATION_AVAILABLE:
        print("Animation components not available. Install imageio with: pip install imageio>=2.19.0")
        return
    
    print("Creating polynomial function learning animation with k-neighbors...")
    
    # Define function and algorithm names with lowercase naming to match required structure
    function_name = "polynomial"  # Lowercase for consistent folder naming
    algorithm_name = "k-neighbors"  # Using k-neighbors algorithm name
    base_output_dir = Path("examples") / "results"  # Change to results dir instead of output
    
    # Create specific output directories
    specific_output_dir = base_output_dir / function_name / algorithm_name
    gifs_and_images_dir = specific_output_dir / "gifs_and_images"
    frames_dir = specific_output_dir / "frames"
    
    # Create required directories
    gifs_and_images_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a cubic polynomial: 0.5 + 1.5x - 0.8x² + 2.0x³
    poly_func = PolynomialFunction([0.5, 1.5, -0.8, 2.0])
    
    # Create the learning algorithm
    algorithm = KNeighborsAlgorithm({
        'n_neighbors': 3,
        'weights': 'distance',
        'sample_size': 5,
        'max_samples': 40
    })
    
    # Initialize training
    algorithm.train(target_function=poly_func)
    
    # Create animator
    animator = TrainingAnimator(
        function=poly_func,
        config={
            # Required names for directory structure
            'function_name': function_name, 
            'algorithm_name': algorithm_name,
            'output_base_dir': str(base_output_dir),  # Pass the base path
            
            # Animation settings
            'fps': 10,
            'duration': 200,  # 200ms per frame
            'title': "Learning a Cubic Polynomial with K-Neighbors",
            'figsize': (12, 5)
        }
    )
    
    # Create recorder
    recorder = TrainingRecorder(
        animator=animator,
        capture_frequency=1,  # Capture every epoch
        max_frames=40  # Limit to 40 frames
    )
    
    # Run training and record progress
    print("Running k-neighbors simulation...")
    print(f"Target coefficients: {poly_func.coefficients}")
    start_time = time.time()
    
    all_metrics = []
    
    # Train for 40 epochs
    for epoch in range(40):
        # Train one epoch
        metrics = algorithm.train_epoch(epoch)
        all_metrics.append(metrics)
        
        # Capture frame
        recorder.on_epoch_end(algorithm, epoch, metrics)
        
        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 39:
            print(f"  Epoch {epoch}: MSE={metrics['MSE']:.6f}, Samples={metrics['SampleCount']}, MaxError={metrics['MaxError']:.6f}")
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Create GIF animation
    print("Creating GIF animation...")
    # Output path is now relative to the gifs_and_images directory
    gif_filename = "polynomial_kneighbors.gif"
    output_path = gifs_and_images_dir / gif_filename
    
    animator.create_animation(str(output_path), format='gif')
    print(f"Animation saved to {output_path}")
    
    # Plot the error progression
    plt.figure(figsize=(10, 6))
    epochs = np.arange(len(all_metrics))
    mse_values = [m['MSE'] for m in all_metrics]
    max_error_values = [m['MaxError'] for m in all_metrics]
    
    plt.plot(epochs, mse_values, 'b-', marker='o', markersize=4, label='MSE')
    plt.plot(epochs, max_error_values, 'r-', marker='x', markersize=4, label='Max Error')
    
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("K-Neighbors Error Progression")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the error plot
    error_plot_filename = "error_progression.png"
    error_plot_path = gifs_and_images_dir / error_plot_filename
    plt.savefig(error_plot_path)
    print(f"Error progression plot saved to {error_plot_path}")
    
    # Plot the sample count progression
    plt.figure(figsize=(10, 6))
    sample_counts = [m['SampleCount'] for m in all_metrics]
    
    plt.plot(epochs, sample_counts, 'g-', marker='o', markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Number of Samples")
    plt.title("K-Neighbors Sample Count Progression")
    plt.grid(True, alpha=0.3)
    
    # Save the sample count plot
    samples_plot_filename = "samples_progression.png"
    samples_plot_path = gifs_and_images_dir / samples_plot_filename
    plt.savefig(samples_plot_path)
    print(f"Sample count progression plot saved to {samples_plot_path}")
    
    return output_path

if __name__ == "__main__":
    run_demo() 