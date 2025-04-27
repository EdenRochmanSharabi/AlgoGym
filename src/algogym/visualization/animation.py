import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import os
import imageio.v2 as imageio
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
import tempfile
from PIL import Image
import shutil
import time

from algogym.functions import BaseFunction
from algogym.algorithms import BaseAlgorithm
from .base import BaseVisualizer


class TrainingAnimator:
    """
    Creates animated visualizations of algorithm training progress over epochs.
    
    This class generates GIFs or videos that show how an algorithm's approximation
    evolves during the training process, making it easy to visualize the learning
    process for presentations, documentation, or social media.
    """
    
    def __init__(self, 
                 function: BaseFunction,
                 visualizer: Optional[BaseVisualizer] = None,
                 config: Dict[str, Any] = None):
        """
        Initialize the training animator.
        
        Args:
            function (BaseFunction): The target function being approximated.
            visualizer (BaseVisualizer, optional): Visualizer for creating individual frames.
                If None, the default visualizer will be used.
            config (Dict[str, Any], optional): Configuration parameters.
                Supported keys:
                - 'fps': Frames per second for the animation (default: 10)
                - 'duration': Duration of each frame in the GIF in ms (default: 100)
                - 'loop': Number of times to loop the GIF (default: 0, meaning infinite)
                - 'figsize': Figure size for frames (default: (8, 6))
                - 'dpi': DPI for frames (default: 100)
                - 'cmap': Colormap for 2D visualizations (default: 'viridis')
                - 'epoch_label_fmt': Format string for epoch labels (default: 'Epoch: {:d}')
                - 'epoch_label_pos': Position of epoch label (default: (0.05, 0.95))
                - 'frames_dir': Parent directory to store individual frames. DEPRECATED: now constructed automatically.
                - 'output_base_dir': Base directory for all outputs (default: 'output')
                - 'save_data': Whether to save the raw data used for each frame (default: True)
                - 'min_change_threshold': Minimum change between frames to consider significant (default: 0.01)
                - 'final_mse_threshold': Maximum MSE in final epoch to consider acceptable (default: 0.1)
                - 'min_epoch_distance': Minimum number of epochs between saved frames to avoid redundancy (default: 1)
                - 'algorithm_name': Name of the algorithm being visualized (required).
                - 'function_name': Name of the function being learned (required).
        """
        self.function = function
        self.visualizer = visualizer
        
        # Default configuration
        self.config = {
            'fps': 12,                # Slightly higher FPS for smoother animations
            'duration': 83.33,        # milliseconds (1000/12 for 12 FPS)
            'loop': 0,                # 0 means loop indefinitely
            'figsize': (10, 7),       # Slightly larger default size
            'dpi': 120,               # Higher DPI for better quality
            'cmap': 'viridis',
            'epoch_label_fmt': 'Epoch: {:d}',
            'epoch_label_pos': (0.05, 0.95),
            'title': None,
            'quality': 95,            # Higher quality for better images
            'output_base_dir': 'output', # Base directory for outputs
            'save_data': True,        # Save raw data used for frames
            'min_change_threshold': 0.01,  # Minimum MSE change between frames
            'final_mse_threshold': 0.1,    # Maximum acceptable final MSE
            'min_epoch_distance': 1,       # Minimum epochs between frames
            # 'algorithm_name': None,  # Required
            # 'function_name': None,  # Required
        }
        
        # Update with user-provided config
        if config:
            self.config.update(config)
            
        # Validate required config keys
        if 'algorithm_name' not in self.config or 'function_name' not in self.config:
            raise ValueError("'algorithm_name' and 'function_name' must be provided in the config.")

        # Construct the specific frames directory based on names
        self.algorithm_name = self.config['algorithm_name']
        self.function_name = self.config['function_name']
        self.base_output_dir = Path(self.config['output_base_dir'])
        self.specific_output_dir = self.base_output_dir / self.function_name / self.algorithm_name
        self.frames_dir = self.specific_output_dir / 'frames' # Specific directory for this run's frames

        # Initialize frame storage
        self._frames = []
        self._metrics_history = {}
        self._epoch_numbers = []
        
        # Storage for validation data
        self._prediction_history = []
        self._validation_results = {}
        
        # Create timestamp for unique frame directory
        self._timestamp = time.strftime("%Y%m%d-%H%M%S")
        
    def capture_frame(self, 
                      algorithm: BaseAlgorithm, 
                      epoch: int, 
                      metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Capture a frame of the algorithm's current state during training.
        
        Args:
            algorithm (BaseAlgorithm): The algorithm at its current training state.
            epoch (int): The current epoch number.
            metrics (Dict[str, float], optional): Metrics for the current epoch.
        """
        # Generate and store prediction data for validation AND plotting
        frame_pred_data = self._store_prediction_data(algorithm, epoch)

        # Store frame data
        self._frames.append({
            # 'algorithm': algorithm, # Don't store the algorithm reference
            'epoch': epoch,
            'metrics': metrics or {},
            'prediction_data': frame_pred_data # Store the calculated predictions
        })
        
        # Store metrics history for progress plots
        if metrics:
            for key, value in metrics.items():
                if key not in self._metrics_history:
                    self._metrics_history[key] = []
                self._metrics_history[key].append(value)
            
            self._epoch_numbers.append(epoch)
        
    def _generate_frame(self, frame_data: Dict[str, Any], fig: Figure, axes: List[Axes]) -> None:
        """
        Generate a single frame for the animation.
        
        Args:
            frame_data (Dict[str, Any]): Data for the current frame.
            fig (Figure): The matplotlib figure object.
            axes (List[Axes]): The matplotlib axes objects.
        """
        # Clear the axes
        for ax in axes:
            ax.clear()
        
        epoch = frame_data['epoch']
        metrics = frame_data['metrics']
        
        # Plot based on function dimensions
        if self.function.input_dim == 1 and self.function.output_dim == 1:
            self._generate_1d_frame(frame_data, epoch, metrics, fig, axes)
        elif self.function.input_dim == 2 and self.function.output_dim == 1:
            self._generate_2d_frame(frame_data, epoch, metrics, fig, axes)
        else:
            raise ValueError(f"Animations not supported for functions with input_dim={self.function.input_dim} "
                           f"and output_dim={self.function.output_dim}")
    
    def _generate_1d_frame(self, 
                           frame_data: Dict[str, Any],
                           epoch: int, 
                           metrics: Dict[str, float],
                           fig: Figure, 
                           axes: List[Axes]) -> None:
        """Generate a frame for a 1D function."""
        ax_func, ax_metrics = axes
        
        # Plot function and approximation
        lower_bound, upper_bound = self.function.domain
        x = np.linspace(lower_bound[0], upper_bound[0], 1000)
        
        # True function
        y_true = self.function(x)
        ax_func.plot(x, y_true, 'b-', label='True Function', linewidth=2)
        
        # Retrieve stored prediction
        prediction_data = frame_data.get('prediction_data')
        if prediction_data and 'y_pred' in prediction_data:
            # We need to interpolate if the stored predictions are on a different grid
            stored_x = prediction_data.get('x')
            stored_y_pred = prediction_data.get('y_pred')
            
            if stored_x is not None and stored_y_pred is not None:
                # Check if shapes match what we expect
                if stored_x.ndim > 1:
                    stored_x = stored_x.flatten()
                if stored_y_pred.ndim > 1:
                    stored_y_pred = stored_y_pred.flatten()
                
                # Interpolate to get predictions at the visualization points
                from scipy.interpolate import interp1d
                try:
                    interp_func = interp1d(stored_x, stored_y_pred, bounds_error=False, fill_value="extrapolate")
                    y_approx = interp_func(x)
                    ax_func.plot(x, y_approx, 'r--', label='Approximation', linewidth=2)
                except Exception as e:
                    print(f"Warning: Error interpolating predictions: {e}")
                    # Direct plot of stored values as fallback
                    ax_func.plot(stored_x, stored_y_pred, 'r--', label='Approximation', linewidth=2)
            else:
                print(f"Warning: Prediction data incomplete for epoch {epoch}")
        else:
            print(f"Warning: No prediction data for epoch {epoch}")
        
        # Set labels and title
        ax_func.set_xlabel('Input (x)')
        ax_func.set_ylabel('Output (y)')
        title = self.config['title'] or f"Function Approximation"
        ax_func.set_title(title)
        ax_func.legend()
        ax_func.grid(True, alpha=0.3)
        
        # Add epoch label
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        epoch_label = self.config['epoch_label_fmt'].format(epoch)
        ax_func.text(
            self.config['epoch_label_pos'][0], 
            self.config['epoch_label_pos'][1], 
            epoch_label, 
            transform=ax_func.transAxes, 
            fontsize=12,
            bbox=props,
            verticalalignment='top'
        )
        
        # Plot metrics if we have history
        if self._metrics_history and len(self._epoch_numbers) > 1:
            # Find the maximum epoch index in our history that's <= the current epoch
            current_index = next((i for i, e in enumerate(self._epoch_numbers) if e >= epoch), 0)
            
            for metric_name, values in self._metrics_history.items():
                # Skip non-scalar metrics (like 'Coefficients')
                if isinstance(values[0], (np.ndarray, list)) and not np.isscalar(values[0]):
                    continue
                    
                # Plot all values up to the current epoch
                valid_epochs = self._epoch_numbers[:current_index+1]
                valid_values = values[:current_index+1]
                
                ax_metrics.plot(valid_epochs, valid_values, 
                             marker='o', markersize=4, label=metric_name)
                
                # Annotate the current value
                if valid_values:
                    current_value = valid_values[-1]
                    # Ensure we're formatting a scalar value
                    if np.isscalar(current_value):
                        ax_metrics.annotate(
                            f'{current_value:.4g}',
                            xy=(valid_epochs[-1], current_value),
                            xytext=(5, 0),
                            textcoords='offset points'
                        )
            
            ax_metrics.set_xlabel('Epoch')
            ax_metrics.set_ylabel('Metric Value')
            ax_metrics.set_title('Training Progress')
            ax_metrics.legend()
            ax_metrics.grid(True, alpha=0.3)
        
    def _generate_2d_frame(self, 
                           frame_data: Dict[str, Any],
                           epoch: int, 
                           metrics: Dict[str, float],
                           fig: Figure, 
                           axes: List[Axes]) -> None:
        """Generate a frame for a 2D function."""
        ax_func, ax_metrics = axes
        
        # Generate grid for function evaluation
        lower_bound, upper_bound = self.function.domain
        grid_density = 50
        x1 = np.linspace(lower_bound[0], upper_bound[0], grid_density)
        x2 = np.linspace(lower_bound[1], upper_bound[1], grid_density)
        X1, X2 = np.meshgrid(x1, x2)
        grid_inputs = np.column_stack((X1.flatten(), X2.flatten()))
        
        # Evaluate true function
        z_true = self.function(grid_inputs).reshape(grid_density, grid_density)
        
        # Get predicted values from stored data
        prediction_data = frame_data.get('prediction_data')
        if prediction_data and 'z_pred' in prediction_data:
            # Use stored predictions
            z_pred = prediction_data['z_pred']
            
            # Reshape if needed
            if z_pred.ndim == 1:
                try:
                    z_pred = z_pred.reshape(grid_density, grid_density)
                except ValueError:
                    # If reshaping fails, we need to interpolate
                    try:
                        # Get grid inputs and reshape to 2D arrays
                        stored_grid = prediction_data.get('grid_inputs')
                        if stored_grid is not None:
                            from scipy.interpolate import griddata
                            # Get x and y coordinates from stored grid
                            x_coords = stored_grid[:, 0]
                            y_coords = stored_grid[:, 1]
                            # Interpolate onto target grid
                            points = np.column_stack((x_coords, y_coords))
                            z_pred = griddata(points, z_pred, (X1, X2), method='cubic')
                    except Exception as e:
                        print(f"Warning: Error interpolating 2D predictions: {e}")
                        # Fallback to a dummy approximation
                        z_pred = z_true * 0.5  # Dummy approximation
        else:
            print(f"Warning: No 2D prediction data for epoch {epoch}")
            # Fallback to a dummy approximation
            z_pred = z_true * 0.5  # Dummy approximation
        
        # Compute error
        error = np.abs(z_true - z_pred)
        
        # Create contour plot of error
        # For Rosenbrock function, the error can be very large, so use a logarithmic scale
        if hasattr(self.function, "__class__") and "Rosenbrock" in self.function.__class__.__name__:
            # Add a small epsilon to avoid log(0)
            error = np.log10(error + 1e-10)
            
            # Fix: Use consistent error levels across all frames
            # Use fixed range for error visualization instead of dynamic range
            error_min = -2  # Log10 of 0.01
            error_max = 4   # Log10 of 10000
            levels = np.linspace(error_min, error_max, 20)
            
            # Clip error values to this range for consistent visualization
            error_clipped = np.clip(error, error_min, error_max)
            
            contour = ax_func.contourf(X1, X2, error_clipped, levels=levels, cmap=self.config['cmap'])
            fig.colorbar(contour, ax=ax_func, label='Log10(Absolute Error)')
            
            # Fix: Use consistent contour levels for true function
            # Define fixed contour levels for the true Rosenbrock function
            true_levels = np.array([0, 5, 25, 100, 400, 1000])
            true_contour = ax_func.contour(X1, X2, z_true, levels=true_levels, 
                                         colors='blue', alpha=0.5, linewidths=1.0)
            ax_func.clabel(true_contour, inline=True, fontsize=8, fmt='%.0f')
            
            # Use the same consistent levels for approximation if possible
            # Filter the levels that are in range for the approximation
            approx_min = np.min(z_pred)
            approx_max = np.max(z_pred)
            
            # Use levels that are within the range of the approximation
            valid_approx_levels = true_levels[(true_levels >= approx_min) & (true_levels <= approx_max)]
            if len(valid_approx_levels) < 3:
                # If not enough valid levels, create a custom scale for approximation
                approx_levels = np.linspace(approx_min, approx_max, 5)
            else:
                approx_levels = valid_approx_levels
            
            approx_contour = ax_func.contour(X1, X2, z_pred, levels=approx_levels,
                                           colors='red', alpha=0.5, linestyles='dashed', linewidths=1.0)
            ax_func.clabel(approx_contour, inline=True, fontsize=8, fmt='%.0f')
        else:
            # For other functions, use regular visualization but with consistent levels
            
            # Fix: Use consistent error contour levels
            error_max = 1.0  # Set a fixed max error value
            levels = np.linspace(0, error_max, 20)
            error_clipped = np.clip(error, 0, error_max)
            
            contour = ax_func.contourf(X1, X2, error_clipped, levels=levels, cmap=self.config['cmap'])
            fig.colorbar(contour, ax=ax_func, label='Absolute Error')
            
            # Fix: Use consistent contour levels for true function and approximation
            # Determine global min/max values for consistency across frames
            if isinstance(self.function, type) and hasattr(self.function, "__name__") and "Sin" in self.function.__name__:
                true_levels = np.linspace(-1.5, 1.5, 10)  # For sine-like functions
            else:
                # For polynomial-like functions
                true_levels = np.linspace(-1.0, 1.0, 10)
            
            true_contour = ax_func.contour(X1, X2, z_true, levels=true_levels, 
                                          colors='blue', alpha=0.5, linewidths=1.0)
            ax_func.clabel(true_contour, inline=True, fontsize=8, fmt='%.1f')
            
            approx_contour = ax_func.contour(X1, X2, z_pred, levels=true_levels,
                                           colors='red', alpha=0.5, linestyles='dashed', linewidths=1.0)
            ax_func.clabel(approx_contour, inline=True, fontsize=8, fmt='%.1f')
        
        # Add global best position as a marker
        if hasattr(self.function, 'global_best_position') and self.function.global_best_position is not None:
            if self.function.global_best_position.size >= 2:  # Ensure we have at least 2D
                best_x, best_y = self.function.global_best_position[0], self.function.global_best_position[1]
                if lower_bound[0] <= best_x <= upper_bound[0] and lower_bound[1] <= best_y <= upper_bound[1]:
                    ax_func.plot(best_x, best_y, 'go', markersize=8, label='Global Best')
        
        # Add all particle positions if available
        if hasattr(self.function, 'particles'):
            particle_xs = []
            particle_ys = []
            for particle in self.function.particles:
                if particle.position.size >= 2:  # Ensure we have at least 2D
                    particle_xs.append(particle.position[0])
                    particle_ys.append(particle.position[1])
            if particle_xs:
                ax_func.plot(particle_xs, particle_ys, 'k.', markersize=3, alpha=0.5, label='Particles')
                
        # Set labels and title
        ax_func.set_xlabel('x₁')
        ax_func.set_ylabel('x₂')
        title = self.config['title'] or f"2D Function Approximation"
        ax_func.set_title(title)
        ax_func.grid(True, alpha=0.3)
        
        # Fix: Set consistent x and y axis limits for all frames
        ax_func.set_xlim(lower_bound[0], upper_bound[0])
        ax_func.set_ylim(lower_bound[1], upper_bound[1])
        
        # Only add legend if we have labeled artists
        handles, labels = ax_func.get_legend_handles_labels()
        if handles:
            ax_func.legend(loc='upper right')
        
        # Add epoch label
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
        epoch_label = self.config['epoch_label_fmt'].format(epoch)
        ax_func.text(
            self.config['epoch_label_pos'][0], 
            self.config['epoch_label_pos'][1], 
            epoch_label, 
            transform=ax_func.transAxes, 
            fontsize=12,
            bbox=props,
            verticalalignment='top'
        )
        
        # Plot metrics if we have history
        if self._metrics_history and len(self._epoch_numbers) > 1:
            self._plot_metrics(ax_metrics, epoch)
    
    def _plot_metrics(self, ax_metrics, epoch):
        """Plot metrics history up to the current epoch"""
        # Find the maximum epoch index in our history that's <= the current epoch
        current_index = next((i for i, e in enumerate(self._epoch_numbers) if e >= epoch), 0)
        
        for metric_name, values in self._metrics_history.items():
            # Skip non-scalar metrics (like 'Coefficients')
            if isinstance(values[0], (np.ndarray, list)) and not np.isscalar(values[0]):
                continue
                
            # Plot all values up to the current epoch
            valid_epochs = self._epoch_numbers[:current_index+1]
            valid_values = values[:current_index+1]
            
            ax_metrics.plot(valid_epochs, valid_values, 
                         marker='o', markersize=4, label=metric_name)
            
            # Annotate the current value
            if valid_values:
                current_value = valid_values[-1]
                # Ensure we're formatting a scalar value
                if np.isscalar(current_value):
                    ax_metrics.annotate(
                        f'{current_value:.4g}',
                        xy=(valid_epochs[-1], current_value),
                        xytext=(5, 0),
                        textcoords='offset points'
                    )
        
        ax_metrics.set_xlabel('Epoch')
        ax_metrics.set_ylabel('Metric Value')
        ax_metrics.set_title('Training Progress')
        ax_metrics.legend()
        ax_metrics.grid(True, alpha=0.3)
    
    def _store_prediction_data(self, algorithm: BaseAlgorithm, epoch: int) -> Dict[str, Any]:
        """
        Store prediction data for the current frame to enable validation.
        
        Args:
            algorithm (BaseAlgorithm): The algorithm at its current state.
            epoch (int): The current epoch number.
            
        Returns:
            Dict[str, Any]: The dictionary containing prediction data for this frame.
        """
        frame_data = {'epoch': epoch}
        # Generate grid for evaluation
        if self.function.input_dim == 1:
            # 1D function
            lower_bound, upper_bound = self.function.domain
            x = np.linspace(lower_bound[0], upper_bound[0], 100)
            x_grid = x.reshape(-1, 1)
            
            # Evaluate true function and approximation
            y_true = self.function(x_grid)
            y_pred = algorithm.predict(x_grid)
            
            # Store data
            frame_data.update({
                'x': x_grid,
                'y_true': y_true,
                'y_pred': y_pred,
                'mse': np.mean((y_true - y_pred) ** 2),
                'max_error': np.max(np.abs(y_true - y_pred))
            })
            self._prediction_history.append(frame_data) # Still store history for validation/saving
        else:
            # 2D or higher function
            lower_bound, upper_bound = self.function.domain
            grid_density = 20  # Lower density for storage efficiency
            
            if self.function.input_dim == 2:
                # 2D grid for common case
                x1 = np.linspace(lower_bound[0], upper_bound[0], grid_density)
                x2 = np.linspace(lower_bound[1], upper_bound[1], grid_density)
                X1, X2 = np.meshgrid(x1, x2)
                grid_inputs = np.column_stack((X1.flatten(), X2.flatten()))
                
                # Evaluate functions
                z_true = self.function(grid_inputs)
                z_pred = algorithm.predict(grid_inputs)
                
                # Store data
                frame_data.update({
                    'grid_inputs': grid_inputs,
                    'x1_range': x1,
                    'x2_range': x2,
                    'z_true': z_true,
                    'z_pred': z_pred,
                    'mse': np.mean((z_true - z_pred) ** 2),
                    'max_error': np.max(np.abs(z_true - z_pred))
                })
                self._prediction_history.append(frame_data) # Still store history for validation/saving
            else:
                # Higher dimensions - simplified storage
                test_points = np.random.uniform(
                    low=np.array(lower_bound),
                    high=np.array(upper_bound),
                    size=(100, self.function.input_dim)
                )
                
                # Evaluate functions
                y_true = self.function(test_points)
                y_pred = algorithm.predict(test_points)
                
                # Store data
                frame_data.update({
                    'test_points': test_points,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'mse': np.mean((y_true - y_pred) ** 2),
                    'max_error': np.max(np.abs(y_true - y_pred))
                })
                self._prediction_history.append(frame_data) # Still store history for validation/saving
        
        return frame_data # Return the data for this frame
    
    def validate_training_progress(self) -> bool:
        """
        Validate the training progress by checking for meaningful changes
        between epochs and convergence to the true function.
        
        Returns:
            bool: True if the training progress is valid, False otherwise.
        """
        if not self._prediction_history:
            print("Warning: No prediction data available for validation.")
            return False
        
        # Check if there are enough frames
        if len(self._prediction_history) < 3:
            print("Warning: Need at least 3 epochs for proper validation.")
            return False
        
        # Sort predictions by epoch
        sorted_predictions = sorted(self._prediction_history, key=lambda x: x['epoch'])
        
        # 1. Check for meaningful changes between consecutive epochs
        mse_values = [pred['mse'] for pred in sorted_predictions]
        epoch_numbers = [pred['epoch'] for pred in sorted_predictions]
        
        # Calculate changes in MSE
        mse_changes = [abs(mse_values[i] - mse_values[i-1]) for i in range(1, len(mse_values))]
        significant_changes = [change > self.config['min_change_threshold'] for change in mse_changes]
        
        # Ensure there are at least some significant changes 
        if not any(significant_changes):
            print("Warning: No significant changes in model performance across epochs.")
            self._validation_results['significant_changes'] = False
        else:
            self._validation_results['significant_changes'] = True
        
        # 2. Check for convergence - MSE should generally decrease over time
        is_decreasing = mse_values[0] > mse_values[-1]
        if not is_decreasing:
            print("Warning: MSE does not decrease from first to last epoch.")
            self._validation_results['decreasing_mse'] = False
        else:
            self._validation_results['decreasing_mse'] = True
        
        # 3. Check final approximation quality
        final_mse = mse_values[-1]
        acceptable_final_mse = final_mse < self.config['final_mse_threshold']
        if not acceptable_final_mse:
            print(f"Warning: Final MSE ({final_mse:.4f}) exceeds threshold ({self.config['final_mse_threshold']:.4f}).")
            self._validation_results['acceptable_final_mse'] = False
        else:
            self._validation_results['acceptable_final_mse'] = True
        
        # 4. Check spacing between epochs
        epoch_diffs = [epoch_numbers[i] - epoch_numbers[i-1] for i in range(1, len(epoch_numbers))]
        even_spacing = all(diff >= self.config['min_epoch_distance'] for diff in epoch_diffs)
        self._validation_results['even_spacing'] = even_spacing
        
        # Determine overall validation result
        validation_passed = (
            self._validation_results['significant_changes'] and
            self._validation_results['decreasing_mse'] and
            self._validation_results['acceptable_final_mse']
        )
        
        if not validation_passed:
            print("Warning: Training progress validation failed. Animation may not be informative.")
            print(f"Validation results: {self._validation_results}")
        
        return validation_passed
    
    def save_prediction_data(self, frames_dir: Path) -> None:
        """
        Save the prediction data to a file for future analysis.
        
        Args:
            frames_dir (Path): Directory to save the data to.
        """
        if not self._prediction_history:
            return
        
        # Create the data file
        data_file = frames_dir / "prediction_data.npz"
        
        # Extract data based on dimension
        if self.function.input_dim == 1:
            # 1D function - save in array format
            epochs = np.array([p['epoch'] for p in self._prediction_history])
            x_values = np.array([p['x'] for p in self._prediction_history])
            y_true_values = np.array([p['y_true'] for p in self._prediction_history])
            y_pred_values = np.array([p['y_pred'] for p in self._prediction_history])
            mse_values = np.array([p['mse'] for p in self._prediction_history])
            
            # Save to npz file
            np.savez(
                data_file,
                epochs=epochs,
                x_values=x_values,
                y_true_values=y_true_values,
                y_pred_values=y_pred_values,
                mse_values=mse_values
            )
        elif self.function.input_dim == 2:
            # 2D function
            epochs = np.array([p['epoch'] for p in self._prediction_history])
            grid_inputs = np.array([p['grid_inputs'] for p in self._prediction_history])
            x1_ranges = np.array([p['x1_range'] for p in self._prediction_history])
            x2_ranges = np.array([p['x2_range'] for p in self._prediction_history])
            z_true_values = np.array([p['z_true'] for p in self._prediction_history])
            z_pred_values = np.array([p['z_pred'] for p in self._prediction_history])
            mse_values = np.array([p['mse'] for p in self._prediction_history])
            
            # Save to npz file
            np.savez(
                data_file,
                epochs=epochs,
                grid_inputs=grid_inputs,
                x1_ranges=x1_ranges,
                x2_ranges=x2_ranges,
                z_true_values=z_true_values,
                z_pred_values=z_pred_values,
                mse_values=mse_values
            )
        else:
            # Higher dimensions - simplified storage
            epochs = np.array([p['epoch'] for p in self._prediction_history])
            test_points = np.array([p['test_points'] for p in self._prediction_history])
            y_true_values = np.array([p['y_true'] for p in self._prediction_history])
            y_pred_values = np.array([p['y_pred'] for p in self._prediction_history])
            mse_values = np.array([p['mse'] for p in self._prediction_history])
            
            # Save to npz file
            np.savez(
                data_file,
                epochs=epochs,
                test_points=test_points,
                y_true_values=y_true_values,
                y_pred_values=y_pred_values,
                mse_values=mse_values
            )
            
        # Save validation results and summary
        summary_file = frames_dir / "validation_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Training Data Summary\n")
            f.write(f"====================\n\n")
            f.write(f"Function: {self.function.__class__.__name__}\n")
            f.write(f"Input Dimension: {self.function.input_dim}\n")
            f.write(f"Output Dimension: {self.function.output_dim}\n\n")
            
            f.write(f"Epochs: {len(self._prediction_history)}\n")
            f.write(f"Starting MSE: {self._prediction_history[0]['mse']:.6f}\n")
            f.write(f"Final MSE: {self._prediction_history[-1]['mse']:.6f}\n")
            f.write(f"MSE Improvement: {self._prediction_history[0]['mse'] - self._prediction_history[-1]['mse']:.6f}\n\n")
            
            f.write(f"Validation Results\n")
            f.write(f"-----------------\n")
            for key, value in self._validation_results.items():
                f.write(f"{key}: {value}\n")
            
            f.write(f"\nDetailed MSE Progression\n")
            f.write(f"----------------------\n")
            for pred in sorted(self._prediction_history, key=lambda x: x['epoch']):
                f.write(f"Epoch {pred['epoch']}: MSE={pred['mse']:.6f}, Max Error={pred['max_error']:.6f}\n")
    
    def create_animation(self, 
                         output_path: str, 
                         format: str = 'gif',
                         use_temp_files: bool = False,
                         force: bool = False) -> str:
        """
        Create an animation from the captured frames.
        
        Args:
            output_path (str): Path to save the animation.
            format (str): Format of the animation ('gif' or 'mp4').
            use_temp_files (bool): Whether to use temporary files for large animations.
                This is more memory-efficient but slower.
            force (bool): Force animation creation even if validation fails.
        
        Returns:
            str: Path to the created animation file.
        """
        if not self._frames:
            raise ValueError("No frames have been captured. Use capture_frame() first.")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Validate training progress if requested
        if not force and self.config['save_data']:
            validation_passed = self.validate_training_progress()
            if not validation_passed:
                print("Consider using force=True to create animation anyway.")
                choice = input("Continue with animation creation? (y/n): ").lower() 
                if choice != 'y':
                    print("Animation creation cancelled.")
                    return None
        
        # Handle different formats
        if format.lower() not in ['gif', 'mp4']:
            raise ValueError(f"Unsupported format: {format}. Use 'gif' or 'mp4'.")
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=self.config['figsize'], dpi=self.config['dpi'])
        
        # Get the pre-calculated frames directory
        frames_dir = self.frames_dir # Use the path constructed in __init__
        animation_name = Path(output_path).stem # Keep animation name for uniqueness if needed, but subdir is now fixed
        frames_subdir = frames_dir # The directory is now the final destination for frames
        frames_subdir.mkdir(parents=True, exist_ok=True)
        
        # Create the gifs_and_images directory alongside frames
        gifs_and_images_dir = self.specific_output_dir / 'gifs_and_images'
        gifs_and_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Adjust output path to be inside gifs_and_images
        final_output_path = gifs_and_images_dir / Path(output_path).name
        
        print(f"Saving individual frames to {frames_subdir}")
        
        # Save prediction data if available
        if self.config['save_data'] and self._prediction_history:
            self.save_prediction_data(frames_subdir)
        
        # Order frames by epoch
        sorted_frames = sorted(self._frames, key=lambda x: x['epoch'])
        
        # Save individual frames first
        frame_paths = self._save_individual_frames(sorted_frames, fig, axes, frames_subdir)
        
        plt.close(fig)
        
        if format.lower() == 'gif':
            return self._create_gif_from_frames(frame_paths, str(final_output_path)) # Save to new path
        else:  # mp4
            return self._create_mp4_from_frames(frame_paths, str(final_output_path)) # Save to new path
    
    def _save_individual_frames(self, 
                              frames: List[Dict[str, Any]], 
                              fig: Figure,
                              axes: List[Axes], 
                              frames_dir: Path) -> List[str]:
        """
        Save individual frames to the frames directory.
        
        Args:
            frames (List[Dict[str, Any]]): List of frame data.
            fig (Figure): The matplotlib figure object.
            axes (List[Axes]): The matplotlib axes objects.
            frames_dir (Path): Directory to save frames to.
            
        Returns:
            List[str]: List of paths to the saved frames.
        """
        # Set figure size consistently to avoid shape issues
        fig.set_size_inches(self.config['figsize'])
        fig.set_dpi(self.config['dpi'])
        
        # Fix: Use consistent layout and margins to avoid frame size variations
        fig.tight_layout(pad=1.2)
        
        frame_paths = []
        
        for i, frame_data in enumerate(frames):
            # Generate the frame
            self._generate_frame(frame_data, fig, axes)
            
            # Fix: Ensure same figure size and DPI for all frames
            fig.set_size_inches(self.config['figsize'])
            fig.set_dpi(self.config['dpi'])
            
            # Save the frame as a file
            epoch = frame_data['epoch']
            # Add algorithm and function name to frame filename for clarity
            frame_path = frames_dir / f"frame_{self.function_name}_{self.algorithm_name}_{i:04d}_epoch_{epoch:04d}.png"
            frame_paths.append(str(frame_path))
            
            # Fix: Use consistent DPI and quality settings
            fig.savefig(frame_path, dpi=self.config['dpi'], bbox_inches='tight', 
                        pad_inches=0.1, facecolor='white')
        
        return frame_paths
    
    def _create_gif_from_frames(self, frame_paths: List[str], output_path: str) -> str:
        """
        Create a GIF animation from saved frame images.
        
        Args:
            frame_paths (List[str]): List of paths to frame images.
            output_path (str): Path to save the GIF.
            
        Returns:
            str: Path to the created GIF file.
        """
        # Create the GIF from the saved frames
        duration_sec = self.config['duration'] / 1000.0
        
        images = []
        first_shape = None
        
        for frame_path in frame_paths:
            img = imageio.imread(frame_path)
            if first_shape is None:
                first_shape = img.shape
            elif img.shape != first_shape:
                # Fix: Ensure consistent dimensions by properly resizing
                pil_img = Image.fromarray(img)
                pil_img = pil_img.resize((first_shape[1], first_shape[0]), Image.LANCZOS)
                img = np.array(pil_img)
            images.append(img)
        
        # Use mimwrite directly with all images at once for better optimization
        imageio.mimwrite(
            output_path, 
            images, 
            format='gif',
            # duration=duration_sec, # Deprecated in imageio v3? Use fps instead.
            fps=self.config['fps'], # Use fps for GIF as well
            loop=0 if self.config['loop'] == 0 else self.config['loop']
        )
        
        print(f"Animation created from {len(frame_paths)} frames and saved to {output_path}")
        return output_path
    
    def _create_mp4_from_frames(self, frame_paths: List[str], output_path: str) -> str:
        """
        Create an MP4 video from saved frame images.
        
        Args:
            frame_paths (List[str]): List of paths to frame images.
            output_path (str): Path to save the MP4.
            
        Returns:
            str: Path to the created MP4 file.
        """
        try:
            # Use imageio to write MP4 directly from the saved frames
            fps = self.config['fps']
            
            images = []
            first_shape = None
            
            for frame_path in frame_paths:
                img = imageio.imread(frame_path)
                if first_shape is None:
                    first_shape = img.shape
                elif img.shape != first_shape:
                    # Resize if dimensions don't match
                    pil_img = Image.fromarray(img)
                    pil_img = pil_img.resize((first_shape[1], first_shape[0]))
                    img = np.array(pil_img)
                images.append(img)
            
            # Create MP4 with imageio
            imageio.mimwrite(
                output_path, 
                images, 
                fps=fps, 
                quality=8, 
                codec='libx264', 
                pixelformat='yuv420p'  # Most compatible pixel format
            )
            
            print(f"Animation created from {len(frame_paths)} frames and saved to {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating MP4 from frames: {str(e)}")
            raise

    def _create_gif(self, 
                   frames: List[Dict[str, Any]], 
                   fig: Figure,
                   axes: List[Axes], 
                   output_path: str,
                   temp_dir: Optional[Path] = None) -> str:
        """
        DEPRECATED: Use _save_individual_frames and _create_gif_from_frames instead.
        Create a GIF animation from frames.
        """
        print("Warning: _create_gif is deprecated, use _save_individual_frames and _create_gif_from_frames instead")
        
        # Extract algorithm and function names for folder structure -> No longer needed here
        # algorithm_name = frames[0]['algorithm'].__class__.__name__ # Algorithm reference removed
        # function_name = self.function.__class__.__name__
        
        # Get the pre-calculated frames directory
        frames_dir = self.frames_dir # Use the path constructed in __init__
        # animation_name = Path(output_path).stem
        # frames_subdir = frames_dir / f"{algorithm_name}_{function_name}_{self._timestamp}_{animation_name}\" # Old way
        frames_subdir = frames_dir # The directory is now the final destination for frames
        frames_subdir.mkdir(parents=True, exist_ok=True)

        # Create the gifs_and_images directory alongside frames
        gifs_and_images_dir = self.specific_output_dir / 'gifs_and_images'
        gifs_and_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Adjust output path to be inside gifs_and_images
        final_output_path = gifs_and_images_dir / Path(output_path).name
        
        # Save individual frames first
        fig, axes = plt.subplots(1, 2, figsize=self.config['figsize'], dpi=self.config['dpi']) # Need fig/axes locally
        frame_paths = self._save_individual_frames(frames, fig, axes, frames_subdir)
        plt.close(fig) # Close the temporary figure
        
        # Create the GIF from the saved frames
        return self._create_gif_from_frames(frame_paths, str(final_output_path)) # Save to new path
    
    def _create_mp4(self, 
                   frames: List[Dict[str, Any]], 
                   fig: Figure,
                   axes: List[Axes], 
                   output_path: str) -> str:
        """
        DEPRECATED: Use _save_individual_frames and _create_mp4_from_frames instead.
        Create an MP4 video from frames.
        """
        print("Warning: _create_mp4 is deprecated, use _save_individual_frames and _create_mp4_from_frames instead")
        
        # Extract algorithm and function names for folder structure -> No longer needed here
        # algorithm_name = frames[0]['algorithm'].__class__.__name__ # Algorithm reference removed
        # function_name = self.function.__class__.__name__
        
        # Get the pre-calculated frames directory
        frames_dir = self.frames_dir # Use the path constructed in __init__
        # animation_name = Path(output_path).stem
        # frames_subdir = frames_dir / f"{algorithm_name}_{function_name}_{self._timestamp}_{animation_name}" # Old way
        frames_subdir = frames_dir # The directory is now the final destination for frames
        frames_subdir.mkdir(parents=True, exist_ok=True)

        # Create the gifs_and_images directory alongside frames
        gifs_and_images_dir = self.specific_output_dir / 'gifs_and_images'
        gifs_and_images_dir.mkdir(parents=True, exist_ok=True)
        
        # Adjust output path to be inside gifs_and_images
        final_output_path = gifs_and_images_dir / Path(output_path).name

        # Save individual frames first
        # Note: This uses the fig/axes passed in, unlike the GIF deprecation fix
        frame_paths = self._save_individual_frames(frames, fig, axes, frames_subdir)
        
        # Create the MP4 from the saved frames
        return self._create_mp4_from_frames(frame_paths, str(final_output_path)) # Save to new path

    def clear(self) -> None:
        """Clear all captured frames and metrics history."""
        self._frames = []
        self._metrics_history = {}
        self._epoch_numbers = []
        
    def verify_frames(self, frames_dir: str = None) -> bool:
        """
        Verify that the frames are correctly saved and can be used to create animations.
        
        Args:
            frames_dir (str, optional): Path to frames directory. If None, 
                will use the last directory created during animation generation.
                
        Returns:
            bool: True if all frames are valid, False otherwise.
        """
        if frames_dir is None:
            # If no directory provided, find the most recent one
            frames_parent = Path(self.config['frames_dir'])
            if not frames_parent.exists():
                print(f"Frames directory {frames_parent} does not exist.")
                return False
                
            # Find all subdirectories
            subdirs = [d for d in frames_parent.iterdir() if d.is_dir()]
            if not subdirs:
                print(f"No frame subdirectories found in {frames_parent}.")
                return False
                
            # Sort by creation time (most recent first)
            subdirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            frames_dir = subdirs[0]
            print(f"Checking most recent frames directory: {frames_dir}")
        else:
            frames_dir = Path(frames_dir)
            if not frames_dir.exists():
                print(f"Frames directory {frames_dir} does not exist.")
                return False
        
        # Get all image files in the directory
        frame_files = sorted([f for f in frames_dir.glob("frame_*.png")])
        if not frame_files:
            print(f"No frame files found in {frames_dir}.")
            return False
            
        print(f"Found {len(frame_files)} frame files.")
        
        # Check if all frames are valid images
        valid_frames = 0
        invalid_frames = []
        
        for frame_file in frame_files:
            try:
                img = Image.open(frame_file)
                img.verify()  # Verify the file is a valid image
                valid_frames += 1
            except Exception as e:
                invalid_frames.append((frame_file, str(e)))
        
        # Print verification results
        if valid_frames == len(frame_files):
            print(f"All {valid_frames} frames are valid images.")
            return True
        else:
            print(f"{valid_frames} out of {len(frame_files)} frames are valid.")
            print("Invalid frames:")
            for frame, error in invalid_frames:
                print(f"  {frame}: {error}")
            return False


class TrainingRecorder:
    """
    Records the training process of an algorithm for later animation.
    
    This class provides utilities to integrate with algorithm training loops
    and capture the state at different epochs for visualization.
    """
    
    def __init__(self, 
                 animator: TrainingAnimator,
                 capture_frequency: int = 1,
                 max_frames: Optional[int] = None,
                 validate_data: bool = True):
        """
        Initialize the training recorder.
        
        Args:
            animator (TrainingAnimator): The animator to use for recording frames.
            capture_frequency (int): How often to capture frames (every N epochs).
            max_frames (int, optional): Maximum number of frames to capture.
                If None, capture all frames at the specified frequency.
            validate_data (bool): Whether to validate the data before animation.
        """
        self.animator = animator
        self.capture_frequency = capture_frequency
        self.max_frames = max_frames
        self.validate_data = validate_data
        self._frame_count = 0
        
        # Make sure the animator is configured to save data if validation is required
        if validate_data:
            self.animator.config['save_data'] = True
    
    def on_epoch_end(self, 
                     algorithm: BaseAlgorithm, 
                     epoch: int, 
                     metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Callback to be called at the end of each training epoch.
        
        Args:
            algorithm (BaseAlgorithm): The algorithm at its current state.
            epoch (int): The current epoch number.
            metrics (Dict[str, float], optional): Metrics for the current epoch.
        """
        # Check if we should capture this epoch
        if epoch % self.capture_frequency == 0:
            # Check if we've reached the maximum number of frames
            if self.max_frames is None or self._frame_count < self.max_frames:
                # Create a copy of the algorithm to avoid modifying the original
                # For simple algorithms, this works; for more complex ones with
                # non-serializable state, a custom clone method would be needed
                self.animator.capture_frame(algorithm, epoch, metrics)
                self._frame_count += 1
    
    def record_training(self, 
                        algorithm: BaseAlgorithm,
                        epochs: int,
                        train_func: Callable[[BaseAlgorithm, int], Dict[str, float]],
                        validate_after: bool = True) -> Dict[str, List[float]]:
        """
        Record the training process of an algorithm.
        
        Args:
            algorithm (BaseAlgorithm): The algorithm to train.
            epochs (int): Number of epochs to train for.
            train_func (Callable): Function that performs one epoch of training.
                Should take the algorithm and current epoch as arguments and
                return a dictionary of metrics.
            validate_after (bool): Whether to validate the training after all epochs.
        
        Returns:
            Dict[str, List[float]]: History of metrics during training.
        """
        all_metrics = {}
        
        for epoch in range(epochs):
            # Train for one epoch
            metrics = train_func(algorithm, epoch)
            
            # Record metrics history
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
            
            # Capture frame
            self.on_epoch_end(algorithm, epoch, metrics)
        
        # Validate training progress if requested
        if validate_after and self.validate_data:
            validation_passed = self.animator.validate_training_progress()
            if not validation_passed:
                print("Validation failed. You may want to rerun the training with different parameters.")
        
        return all_metrics 