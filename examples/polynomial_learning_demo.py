#!/usr/bin/env python3
"""
Simple demonstration of learning a polynomial function with visualization.

This script shows how a polynomial function is learned over time
and creates a GIF animation of the learning process.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import time

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

class GradualLearningAlgorithm(BaseAlgorithm):
    """
    Simple algorithm that gradually learns polynomial coefficients.
    """
    
    def __init__(self, config=None):
        default_config = {
            'target_coefficients': None,  # Will be set during training
            'learning_rate': 0.1,
            'noise_level': 0.05,
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(default_config)
        
        # Initialize with zero coefficients
        self.current_coefficients = None
    
    def _validate_config(self, config):
        """Validate the configuration."""
        pass
    
    def train(self, target_function=None, X_data=None, y_data=None):
        """Initialize the training process."""
        if not isinstance(target_function, PolynomialFunction):
            raise ValueError("This algorithm only works with PolynomialFunction")
        
        # Store the target function
        self._target_function = target_function
        
        # Get target coefficients
        self.config['target_coefficients'] = target_function.coefficients
        
        # Start with zero coefficients
        self.current_coefficients = np.zeros_like(self.config['target_coefficients'])
    
    def train_epoch(self, epoch):
        """Train for one epoch, gradually updating coefficients."""
        # Get target coefficients
        target_coeffs = self.config['target_coefficients']
        
        # Update each coefficient with a step towards the target
        for i in range(len(self.current_coefficients)):
            # Calculate step towards target
            step = self.config['learning_rate'] * (target_coeffs[i] - self.current_coefficients[i])
            
            # Add noise (decreasing with epochs)
            noise = np.random.normal(0, self.config['noise_level'] / (epoch + 1)**0.5)
            
            # Update coefficient
            self.current_coefficients[i] += step + noise
        
        # Evaluate on test points
        test_points = np.linspace(-0.9, 0.9, 100).reshape(-1, 1)
        true_values = self._target_function(test_points)
        pred_values = self.predict(test_points)
        
        # Calculate metrics
        mse = np.mean((true_values - pred_values) ** 2)
        mae = np.mean(np.abs(true_values - pred_values))
        
        return {
            'MSE': float(mse),
            'MAE': float(mae),
            'Coefficients': self.current_coefficients.copy()
        }
    
    def predict(self, x):
        """Predict using current coefficients."""
        if self.current_coefficients is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Reshape input if needed
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        
        # Evaluate polynomial with current coefficients
        result = np.zeros(x.shape[0])
        x_flat = x.flatten()  # Flatten to 1D array
        
        for i, coef in enumerate(self.current_coefficients):
            result += coef * (x_flat ** i)
        
        return result.reshape(-1, 1)

def run_demo():
    """Run the polynomial learning demonstration."""
    if not ANIMATION_AVAILABLE:
        print("Animation components not available. Install imageio with: pip install imageio>=2.19.0")
        return
    
    print("Creating polynomial function learning animation...")
    
    # Define function and algorithm names with lowercase naming to match required structure
    function_name = "polynomial" # Lowercase for consistent folder naming
    algorithm_name = "genetic" # Using genetic algorithm name instead of GradualLearning
    base_output_dir = Path("examples") / "results" # Change to results dir instead of output
    
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
    algorithm = GradualLearningAlgorithm({
        'learning_rate': 0.2,
        'noise_level': 0.08
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
            'output_base_dir': str(base_output_dir), # Pass the base path
            
            # Animation settings
            'fps': 10,
            'duration': 200,  # 200ms per frame
            'title': "Learning a Cubic Polynomial",
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
    print("Running training simulation...")
    print(f"Target coefficients: {poly_func.coefficients}")
    start_time = time.time()
    
    all_coeffs = []
    
    # Train for 40 epochs
    for epoch in range(40):
        # Train one epoch
        metrics = algorithm.train_epoch(epoch)
        all_coeffs.append(metrics['Coefficients'])
        
        # Capture frame
        recorder.on_epoch_end(algorithm, epoch, metrics)
        
        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == 39:
            print(f"  Epoch {epoch}: MSE={metrics['MSE']:.6f}, Coeffs={metrics['Coefficients']}")
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    # Create GIF animation
    print("Creating GIF animation...")
    # Output path is now relative to the gifs_and_images directory
    gif_filename = "polynomial_learning.gif"
    output_path = gifs_and_images_dir / gif_filename
    # output_path = os.path.join("examples", "output", "polynomial_learning.gif") # Old path
    
    animator.create_animation(str(output_path), format='gif')
    print(f"Animation saved to {output_path}")
    
    # Plot the convergence of coefficients
    plt.figure(figsize=(10, 6))
    all_coeffs = np.array(all_coeffs)
    epochs = np.arange(len(all_coeffs))
    
    for i in range(all_coeffs.shape[1]):
        plt.plot(epochs, all_coeffs[:, i], 
                 label=f"$c_{i}$ (Target: {poly_func.coefficients[i]:.2f})")
    
    plt.xlabel("Epoch")
    plt.ylabel("Coefficient Value")
    plt.title("Coefficient Convergence During Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot to the gifs_and_images directory
    coeff_plot_filename = "coefficient_convergence.png"
    coeff_plot_path = gifs_and_images_dir / coeff_plot_filename
    # coeff_plot_path = os.path.join("examples", "output", "coefficient_convergence.png") # Old path
    plt.savefig(coeff_plot_path)
    print(f"Coefficient convergence plot saved to {coeff_plot_path}")
    
    return output_path

if __name__ == "__main__":
    run_demo() 