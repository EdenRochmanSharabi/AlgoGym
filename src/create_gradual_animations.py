import os
import numpy as np
from pathlib import Path

from algogym.functions import SineFunction, PolynomialFunction, RosenbrockFunction
from algogym.visualization.animation import TrainingAnimator

class GradualLearningMock:
    """A mock implementation of the GradualLearning algorithm for animation purposes."""
    
    def __init__(self, input_dim=1, function_type="sine"):
        self.__class__.__name__ = "GradualLearning"
        self.input_dim = input_dim
        self.function_type = function_type
        
        # Initialize model parameters based on function type
        if function_type == "sine":
            # Parameters for sine approximation
            # [amplitude, frequency, phase, offset]
            self.params = np.array([0.1, 1.0, 0.0, 0.0])
        elif function_type == "polynomial":
            # Parameters for polynomial approximation
            # [constant, x, x^2, x^3]
            self.params = np.array([0.0, 0.1, 0.0, 0.0])
        elif function_type == "rosenbrock":
            # Parameters for simple 2D function
            # [a, b] for Rosenbrock a(x - y)^2 + b(y - x^2)^2
            self.params = np.array([0.1, 0.1])
            self.global_best_position = np.array([0.5, 0.5])
            # Create some particles for visualization
            self.particles = []
            for i in range(10):
                self.particles.append(
                    type('Particle', (), {'position': np.random.uniform(-2, 2, 2)})
                )
        else:
            self.params = np.array([0.0, 0.0, 0.0, 0.0])
    
    def predict(self, x):
        """Predict using current parameters."""
        if self.input_dim == 1:
            if isinstance(x, np.ndarray) and x.ndim == 2:
                x_val = x.flatten()
            else:
                x_val = x
                
            if self.function_type == "sine":
                # Amplitude * sin(frequency * x + phase) + offset
                return self.params[0] * np.sin(self.params[1] * x_val + self.params[2]) + self.params[3]
            elif self.function_type == "polynomial":
                # a + bx + cx^2 + dx^3
                return self.params[0] + self.params[1] * x_val + self.params[2] * x_val**2 + self.params[3] * x_val**3
            else:
                return x_val  # Default fallback
        elif self.input_dim == 2:
            # For 2D functions like Rosenbrock, predict a simple representation
            return np.zeros(x.shape[0])  # We just show contours in 2D case
    
    def train_epoch(self, epoch):
        """Update parameters to better approximate the target function."""
        # Gradually improve parameters with each epoch
        if self.function_type == "sine":
            # Gradually approach true sine wave parameters
            target = np.array([1.0, 1.0, 0.0, 0.0])  # target sine parameters
            self.params += (target - self.params) * 0.1  # Move 10% toward target
            
        elif self.function_type == "polynomial":
            # Gradually approach a cubic function
            target = np.array([0.0, 0.0, 1.0, 0.0])  # target y = x^2
            self.params += (target - self.params) * 0.1
            
        elif self.function_type == "rosenbrock":
            # Update parameters and particle positions for Rosenbrock
            self.params += np.array([0.05, 0.1]) * (1 - np.exp(-0.1 * epoch))
            
            # Move particles toward the optimum (1,1) for Rosenbrock
            for particle in self.particles:
                # Move particles gradually toward (1,1)
                target = np.array([1.0, 1.0])
                particle.position += (target - particle.position) * 0.05
                
                # Add some random movement
                particle.position += np.random.normal(0, 0.1, 2)
            
            # Update best position
            self.global_best_position = np.mean([p.position for p in self.particles], axis=0)
            
        # Calculate mock metrics
        progress = min(1.0, epoch / 30.0)  # Progress from 0 to 1
        mse = 1.0 * (1 - progress)**2  # Decrease over time
        mae = 0.8 * (1 - progress)
        
        return {
            "MSE": mse,
            "MAE": mae,
            "Progress": progress
        }

def create_animation(function, output_dir):
    """Create an animation for GradualLearning with a specific function."""
    # Create output directory
    results_dir = Path(output_dir) / "GradualLearning"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get function name and determine function type
    function_name = function.__class__.__name__
    function_type = function_name.lower().replace("function", "")
    
    # Create animator with custom frames directory
    frames_dir = results_dir / "frames"
    animator_config = {
        'frames_dir': str(frames_dir),
        'title': f"Gradual Learning on {function_name}",
        'figsize': (12, 5)
    }
    animator = TrainingAnimator(function=function, config=animator_config)
    
    # Create mock algorithm
    algorithm = GradualLearningMock(
        input_dim=function.input_dim,
        function_type=function_type
    )
    
    # Manually capture frames
    num_epochs = 30
    for epoch in range(num_epochs):
        # Update the algorithm
        metrics = algorithm.train_epoch(epoch)
        
        # Capture frame
        animator.capture_frame(algorithm, epoch, metrics)
    
    # Create animation
    animation_path = str(results_dir / f"{function_name.lower()}_learning.gif")
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
    
    # Create animations for each function
    for function in functions:
        function_name = function.__class__.__name__
        print(f"Creating GradualLearning animation for {function_name}...")
        
        animation_path, valid = create_animation(function, output_dir)
        
        if valid:
            print(f"  ✓ Animation saved to: {animation_path}")
        else:
            print(f"  ✗ Failed to create animation for GradualLearning on {function_name}")
    
    # Create a showcase animation combining all learning algorithms
    print("\nCreating algorithm learning showcase animation...")
    showcase_dir = Path(output_dir) / "GradualLearning"
    showcase_path = str(showcase_dir / "algorithm_learning_showcase.gif")
    
    # For now, we'll just copy the sine animation as a placeholder
    # In a real implementation, you would create a customized showcase animation
    import shutil
    sine_path = str(showcase_dir / "sine_learning.gif")
    if os.path.exists(sine_path):
        shutil.copy(sine_path, showcase_path)
        print(f"  ✓ Showcase animation saved to: {showcase_path}")
    
    print("\nAll animations created successfully!")

if __name__ == "__main__":
    main() 