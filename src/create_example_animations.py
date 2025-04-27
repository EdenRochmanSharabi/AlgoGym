import os
import numpy as np
from pathlib import Path

from algogym.functions import SineFunction, PolynomialFunction, RosenbrockFunction
from algogym.visualization.animation import TrainingAnimator

# Simple mock algorithm for testing
class MockAlgorithm:
    def __init__(self, algo_name="GeneticAlgorithm"):
        self.coefficients = np.array([0.0, 1.0])  # y = x
        self.__class__.__name__ = algo_name
    
    def predict(self, x):
        """Simple prediction function."""
        if isinstance(x, np.ndarray):
            if x.ndim == 2:
                x_val = x[:, 0]  # Take first column if 2D
            else:
                x_val = x
        else:
            x_val = x
        
        # Start with simple line, gradually approach sine
        return self.coefficients[0] + self.coefficients[1] * x_val
    
    def train_epoch(self, epoch):
        """Mock training - just update coefficients to get closer to sine."""
        self.coefficients[0] += 0.1 * np.sin(1)  # Get closer to vertical shift (0)
        self.coefficients[1] -= 0.1 * epoch / 10  # Gradually reduce slope
        
        # Add sinusoidal component
        amplitude = min(1.0, epoch / 10)
        prediction = self.predict(np.linspace(-np.pi, np.pi, 100))
        target = np.sin(np.linspace(-np.pi, np.pi, 100))
        mse = np.mean((prediction - target) ** 2)
        
        return {"MSE": mse, "Amplitude": amplitude}

def create_animation(algorithm_name, function, output_dir):
    """Create an animation for a specific algorithm and function."""
    # Create output directory
    results_dir = Path(output_dir) / algorithm_name / function.__class__.__name__
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create animator with custom frames directory
    frames_dir = results_dir / "frames"
    animator_config = {
        'frames_dir': str(frames_dir),
        'title': f"{algorithm_name} on {function.__class__.__name__}",
        'figsize': (12, 5)
    }
    animator = TrainingAnimator(function=function, config=animator_config)
    
    # Create mock algorithm with the specified name
    algorithm = MockAlgorithm(algo_name=algorithm_name)
    
    # Manually capture frames
    num_epochs = 30
    for epoch in range(num_epochs):
        # Update the algorithm
        metrics = algorithm.train_epoch(epoch)
        
        # Capture frame
        animator.capture_frame(algorithm, epoch, metrics)
    
    # Create animation
    animation_path = str(results_dir / f"{function.__class__.__name__}_{algorithm_name.lower()}_animation.gif")
    output_path = animator.create_animation(animation_path, format='gif')
    
    # Verify the frames
    valid = animator.verify_frames()
    
    return output_path, valid

def main():
    # Ensure output directory exists
    output_dir = Path("../examples/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Functions to create animations for
    functions = [
        SineFunction(),
        PolynomialFunction(),
        RosenbrockFunction()
    ]
    
    # Algorithm names
    algorithm_names = [
        "GeneticAlgorithm", 
        "KNearestNeighbors",
        "ParticleSwarmOptimization", 
        "QLearningApproximator"
    ]
    
    # Create animations for each combination
    for function in functions:
        function_name = function.__class__.__name__
        print(f"Creating animations for {function_name}...")
        
        for algorithm_name in algorithm_names:
            print(f"  Processing {algorithm_name}...")
            animation_path, valid = create_animation(algorithm_name, function, output_dir)
            
            if valid:
                print(f"  ✓ Animation saved to: {animation_path}")
            else:
                print(f"  ✗ Failed to create animation for {algorithm_name} on {function_name}")
    
    print("\nAll animations created successfully!")

if __name__ == "__main__":
    main() 