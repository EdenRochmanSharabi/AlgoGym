#!/usr/bin/env python3
"""
Simple test script to verify the function fixes.
"""

import numpy as np
import sys
from pathlib import Path

# Add the project root to the Python path
script_path = Path(__file__).parent.absolute()
project_root = script_path.parent
sys.path.insert(0, str(project_root))

# Import the fixed functions
from examples.animation_demo import PolynomialFunction, SinFunction, GradualLearningAlgorithm

def test_polynomial_function():
    print("Testing PolynomialFunction...")
    
    # Create a polynomial function
    poly_func = PolynomialFunction([0.5, -1.5, 1.0])
    
    # Test with different input shapes
    x_scalar = 0.5
    x_1d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    x_2d = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    
    # Evaluate with different inputs
    try:
        y_scalar = poly_func(x_scalar)
        print(f"  Scalar input: {x_scalar} -> {y_scalar}")
        
        y_1d = poly_func(x_1d)
        print(f"  1D input shape: {x_1d.shape} -> output shape: {y_1d.shape}")
        print(f"  First few values: {y_1d[:3]}")
        
        y_2d = poly_func(x_2d)
        print(f"  2D input shape: {x_2d.shape} -> output shape: {y_2d.shape if hasattr(y_2d, 'shape') else 'scalar'}")
        print(f"  First few values: {y_2d[:3]}")
        
        print("  ✓ All tests passed!")
    except Exception as e:
        print(f"  ✗ Error: {e}")

def test_sin_function():
    print("\nTesting SinFunction...")
    
    # Create a sine function
    sin_func = SinFunction(freq=2.0, amplitude=1.5)
    
    # Test with different input shapes
    x_scalar = 0.5
    x_1d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    x_2d = np.array([[0.1], [0.2], [0.3], [0.4], [0.5]])
    
    # Evaluate with different inputs
    try:
        y_scalar = sin_func(x_scalar)
        print(f"  Scalar input: {x_scalar} -> {y_scalar}")
        
        y_1d = sin_func(x_1d)
        print(f"  1D input shape: {x_1d.shape} -> output shape: {y_1d.shape}")
        print(f"  First few values: {y_1d[:3]}")
        
        y_2d = sin_func(x_2d)
        print(f"  2D input shape: {x_2d.shape} -> output shape: {y_2d.shape if hasattr(y_2d, 'shape') else 'scalar'}")
        print(f"  First few values: {y_2d[:3]}")
        
        print("  ✓ All tests passed!")
    except Exception as e:
        print(f"  ✗ Error: {e}")

def test_gradual_learning():
    print("\nTesting GradualLearningAlgorithm with PolynomialFunction...")
    
    # Create a polynomial function
    poly_func = PolynomialFunction([0.5, -1.5, 1.0])
    
    # Create an algorithm
    algorithm = GradualLearningAlgorithm({
        'learning_rate': 0.2,
        'noise_level': 0.1
    })
    
    # Initialize the algorithm
    algorithm.train(target_function=poly_func)
    
    # Train for a few epochs
    mse_values = []
    for epoch in range(10):
        metrics = algorithm.train_epoch(epoch)
        mse_values.append(metrics['MSE'])
        print(f"  Epoch {epoch}: MSE={metrics['MSE']:.6f}")
    
    # Check that MSE is decreasing
    if mse_values[0] > mse_values[-1]:
        print("  ✓ MSE decreased during training!")
    else:
        print("  ✗ MSE did not decrease as expected.")
    
    # Test prediction
    x_test = np.linspace(-0.9, 0.9, 5).reshape(-1, 1)
    y_pred = algorithm.predict(x_test)
    print(f"  Prediction shape for input {x_test.shape}: {y_pred.shape}")
    print(f"  First few predictions: {y_pred[:3].flatten()}")

if __name__ == "__main__":
    test_polynomial_function()
    test_sin_function()
    test_gradual_learning() 