import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

from algogym.functions import SineFunction
from algogym.visualization.animation import TrainingAnimator

# Simple mock algorithm for testing
class MockAlgorithm:
    def __init__(self):
        self.coefficients = np.array([0.0, 1.0])  # y = x
    
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

def test_animation_frames():
    """Test the modified animation system."""
    print("Testing animation with frame saving")
    
    # Create output directory
    output_dir = Path("test_animation_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create function and animator
    function = SineFunction()
    
    # Create animator with custom frames directory
    animator_config = {
        'frames_dir': str(output_dir / 'frames')
    }
    animator = TrainingAnimator(function=function, config=animator_config)
    
    # Create mock algorithm
    algorithm = MockAlgorithm()
    
    # Manually capture frames
    num_epochs = 10
    for epoch in range(num_epochs):
        # Update the algorithm
        metrics = algorithm.train_epoch(epoch)
        
        # Capture frame
        animator.capture_frame(algorithm, epoch, metrics)
    
    # Create animation
    animation_path = str(output_dir / "test_animation.gif")
    output_path = animator.create_animation(animation_path, format='gif')
    
    # Verify the frames exist
    frames_dir = Path(animator.config['frames_dir'])
    found_frames = list(frames_dir.glob("**/frame_*.png"))
    
    print(f"Animation saved to: {output_path}")
    print(f"Found {len(found_frames)} saved frames")
    
    # Verify the frames are valid
    valid = animator.verify_frames()
    print(f"Frames validation result: {valid}")
    
    return output_path, valid

if __name__ == "__main__":
    output_path, frames_valid = test_animation_frames()
    print("Test complete!")
    if frames_valid:
        print("SUCCESS: Frames were created and validated successfully")
    else:
        print("ERROR: Frame validation failed") 