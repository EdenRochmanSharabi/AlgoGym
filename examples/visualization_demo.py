#!/usr/bin/env python3
"""
AlgoGym Visualization Demo

This script demonstrates how to use the OOP structure of AlgoGym to generate
visualization GIFs for different algorithms learning various functions.
Results are organized with the following structure:

results/
  [function]/
    [algorithm]/
      frames/
      gifs_and_images/
"""

import os
import sys
import time
import argparse # Added for CLI arguments
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil

# Add the project root to the Python path
script_path = Path(__file__).parent.absolute()
project_root = script_path.parent
sys.path.insert(0, str(project_root))

# Import function classes
from algogym.functions import (
    BaseFunction,
    SineFunction,
    PolynomialFunction,
    RosenbrockFunction
)

# Import algorithm classes
from algogym.algorithms import (
    BaseAlgorithm,
    GeneticAlgorithm,
    KNearestNeighbors,
    QLearningApproximator,
    ParticleSwarmOptimization
)

# Import visualization tools
from algogym.visualization import FunctionVisualizer
try:
    from algogym.visualization import TrainingAnimator, TrainingRecorder
    ANIMATION_AVAILABLE = True
except ImportError:
    ANIMATION_AVAILABLE = False
    print("Animation components not available. Install imageio, Pillow and matplotlib to enable.")
    sys.exit(1)

# Create a patched animator that handles 1D functions properly
class PatchedTrainingAnimator(TrainingAnimator):
    """TrainingAnimator subclass that properly handles 1D function prediction."""
    
    def __init__(self, function, visualizer, config):
        # Explicitly handle config merging like the base class
        base_config = {
            'fps': 12, 'duration': 83.33, 'loop': 0, 'figsize': (10, 7),
            'dpi': 120, 'cmap': 'viridis', 'epoch_label_fmt': 'Epoch: {:d}',
            'epoch_label_pos': (0.05, 0.95), 'title': None, 'quality': 95,
            'output_base_dir': 'output', 'save_data': True,
            'min_change_threshold': 0.01, 'final_mse_threshold': 0.1,
            'min_epoch_distance': 1,
        }
        merged_config = {**base_config, **(config if config is not None else {})}
        
        # Call parent init with the *merged* config
        super().__init__(function, visualizer, merged_config)
        
        # --- DEBUG PRINT --- 
        print(f"DEBUG: PatchedTrainingAnimator.__init__ - self.config['save_data'] = {self.config.get('save_data')}")
        # --- END DEBUG PRINT ---
        
        # Now self.config is properly set by the parent
        # Define the data directory based on the final config
        self.data_dir = self.specific_output_dir / 'data'
        # Add a temporary directory for intermediate data
        self._temp_data_dir = Path(tempfile.mkdtemp(prefix="algogym_anim_data_"))
        self._frame_data_paths = [] # Keep track of temp file paths

    def _store_prediction_data(self, algorithm, epoch):
        """Store prediction data for the current epoch by saving it to a temporary file."""
        # Generate inputs
        frame_data = {}
        save_path = self._temp_data_dir / f"frame_data_epoch_{epoch}.npz"
        
        if self.function.input_dim == 1:
            # For 1D functions, use a flat array for input
            lower, upper = self.function.domain
            x_inputs = np.linspace(lower[0], upper[0], 1000)
            y_true = self.function(x_inputs)
            try:
                y_pred = algorithm.predict(x_inputs)
            except Exception:
                try:
                    x_inputs_2d = x_inputs.reshape(-1, 1)
                    y_pred_2d = algorithm.predict(x_inputs_2d)
                    if isinstance(y_pred_2d, np.ndarray) and y_pred_2d.ndim == 2:
                        y_pred = y_pred_2d.flatten()
                    else:
                        y_pred = y_pred_2d
                except Exception as e:
                    print(f"Error in algorithm prediction: {e}")
                    y_pred = np.zeros_like(x_inputs)
            
            frame_data = {
                'x': x_inputs,
                'y_true': y_true,
                'y_pred': y_pred,
                'epoch': epoch
            }

        elif self.function.input_dim == 2:
            lower, upper = self.function.domain
            grid_density = 20
            x1 = np.linspace(lower[0], upper[0], grid_density)
            x2 = np.linspace(lower[1], upper[1], grid_density)
            X1, X2 = np.meshgrid(x1, x2)
            grid_points = np.column_stack((X1.flatten(), X2.flatten()))
            true_values = self.function(grid_points)
            if true_values.ndim > 1: true_values = true_values.flatten()
            try:
                pred_values = algorithm.predict(grid_points)
                if pred_values.ndim > 1: pred_values = pred_values.flatten()
                if len(pred_values) != grid_density * grid_density:
                     print(f"Warning: Prediction length mismatch. Using zeros.")
                     pred_values = np.zeros_like(true_values)
            except Exception as e:
                print(f"Error in 2D prediction: {e}")
                pred_values = np.zeros_like(true_values)
            
            z_true = true_values.reshape(grid_density, grid_density)
            z_pred = pred_values.reshape(grid_density, grid_density)
            
            frame_data = {
                'X1': X1,
                'X2': X2,
                'z_true': z_true,
                'z_pred': z_pred,
                'epoch': epoch
            }
        else:
            # For higher dimensions, maybe store simplified data or skip
            print(f"Warning: Data storage not implemented for input_dim={self.function.input_dim}")
            frame_data = {'epoch': epoch} # Store epoch at least

        # Save frame data to temporary file
        try:
            np.savez(save_path, **frame_data)
            self._frame_data_paths.append(save_path) # Store path to temp file
        except Exception as e:
            print(f"Error saving temporary frame data for epoch {epoch}: {e}")
            
        # Return the data as well, in case the base class needs it immediately
        return frame_data 

    def _generate_1d_frame(self, frame_data, epoch, metrics, fig, axes):
        """Generate a frame for a 1D function using stored 1D data."""
        ax_func, ax_metrics = axes
        ax_func.clear()
        ax_metrics.clear()
        
        pred_data = frame_data.get('prediction_data', {})
        x = pred_data.get('x')
        y_true = pred_data.get('y_true')
        y_pred = pred_data.get('y_pred')

        if x is None or y_true is None or y_pred is None:
            ax_func.text(0.5, 0.5, "Incomplete data for frame", ha='center', va='center', transform=ax_func.transAxes)
        else:
            # Ensure data is 1D for plotting
            if x.ndim > 1: x = x.flatten()
            if y_true.ndim > 1: y_true = y_true.flatten()
            if y_pred.ndim > 1: y_pred = y_pred.flatten()
                
            # Plot true function
            ax_func.plot(x, y_true, 'b-', label='True Function', linewidth=2)
            # Plot approximation
            ax_func.plot(x, y_pred, 'r--', label='Approximation', linewidth=2)
            
            # Add labels and title
            ax_func.set_xlabel('Input (x)')
            ax_func.set_ylabel('Output (y)')
            title = self.config.get('title', f"{self.algorithm_name} on {self.function_name}")
            ax_func.set_title(f"{title}")
            ax_func.legend()
            ax_func.grid(True, alpha=0.3)
            
            # Add epoch label
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
            epoch_label = self.config['epoch_label_fmt'].format(epoch)
            ax_func.text(0.05, 0.95, epoch_label, transform=ax_func.transAxes, fontsize=12, 
                         bbox=props, verticalalignment='top')

        # Plot metrics
        self._plot_metrics(metrics, epoch, ax_metrics)

    def _generate_2d_frame(self, frame_data, epoch, metrics, fig, axes):
        """Generate a frame for a 2D function with size matching."""
        ax_func, ax_metrics = axes
        ax_func.clear()
        ax_metrics.clear()
        
        # Extract grid data from frame
        pred_data = frame_data.get('prediction_data', {})
        X1 = pred_data.get('X1')
        X2 = pred_data.get('X2')
        z_true = pred_data.get('z_true')
        z_pred = pred_data.get('z_pred')

        if X1 is None or X2 is None or z_true is None or z_pred is None:
            ax_func.text(0.5, 0.5, "Incomplete data for frame", 
                        ha='center', va='center', transform=ax_func.transAxes)
        else:
            # Plot true function as surface with contour
            contour_true = ax_func.contourf(X1, X2, z_true, 
                                            levels=10, 
                                            cmap='viridis', 
                                            alpha=0.7)
            # Add colorbar only once
            # fig.colorbar(contour_true, ax=ax_func, shrink=0.5, aspect=5, label='True')
            
            # Plot approximation as contour lines
            contour_pred = ax_func.contour(X1, X2, z_pred, 
                                        levels=10, 
                                        colors='r', 
                                        alpha=0.7, 
                                        linewidths=1)
                                        
            # Add epoch text
            title = self.config.get('title', f"{self.algorithm_name} on {self.function_name}")
            ax_func.set_title(f"{title} - Epoch {epoch+1}")
            ax_func.set_xlabel("x1")
            ax_func.set_ylabel("x2")
            ax_func.grid(True, alpha=0.3)
            
            # Set consistent limits
            ax_func.set_xlim(X1.min(), X1.max())
            ax_func.set_ylim(X2.min(), X2.max())

        # Plot metrics in the second axis
        self._plot_metrics(metrics, epoch, ax_metrics)

    def _plot_metrics(self, metrics, epoch, ax):
        """Plot metrics history in the given axis."""
        # Clear the axis
        ax.clear()
        
        plotted_metric_names = [] # Keep track of metrics actually plotted
        metrics_to_plot_keys = ['mse', 'mae', 'best_fitness'] # Explicitly choose metrics
        
        # Plot each selected metric if available and valid
        plot_index = 0
        for metric_name in metrics_to_plot_keys:
            if metric_name not in self._metrics_history:
                continue # Skip if metric history doesn't exist

            values = self._metrics_history[metric_name]

            # Filter out NaN or non-numeric values for plotting
            valid_indices = [idx for idx, v in enumerate(values) if isinstance(v, (int, float)) and not np.isnan(v)]
            
            if not valid_indices:
                continue # Skip this metric if no valid data points

            # Get epochs and values corresponding to valid data
            # Adjust epoch numbers if we captured epoch -1
            epoch_offset = -1 if -1 in self._epoch_numbers else 0
            epochs_to_plot = [self._epoch_numbers[idx] for idx in valid_indices if idx < len(self._epoch_numbers)]
            values_to_plot = [values[idx] for idx in valid_indices]

            # Ensure data consistency after filtering
            if not epochs_to_plot or not values_to_plot or len(epochs_to_plot) != len(values_to_plot):
                print(f"Warning: Skipping plot for metric '{metric_name}' due to data mismatch or empty valid data.")
                continue # Skip if data is inconsistent

            # Use a different color for each metric
            color = plt.cm.tab10(plot_index % 10)
            plot_index += 1
            
            # Plot metric values
            ax.plot(epochs_to_plot, values_to_plot, 
                  'o-', label=metric_name, color=color, markersize=3) # Smaller markers
            
            plotted_metric_names.append(metric_name) # Record that this metric was plotted

        # No metrics were plotted
        if not plotted_metric_names:
            ax.text(0.5, 0.5, "No selected metrics available to plot", 
                  ha='center', va='center', transform=ax.transAxes)
            return
            
        # Add labels and title
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Metric Value")
        ax.set_title("Training Progress")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add legend only if we plotted metrics
        if plotted_metric_names:
            ax.legend(loc='best')
            
        # Highlight current epoch (adjusting for potential epoch -1)
        if epoch in self._epoch_numbers:
            try:
                epoch_idx = self._epoch_numbers.index(epoch)
                # Highlight only for metrics that were actually plotted
                for metric_name in plotted_metric_names:
                    values = self._metrics_history[metric_name]
                    # Check index bounds and validity before plotting highlight
                    if epoch_idx < len(values) and isinstance(values[epoch_idx], (int, float)) and not np.isnan(values[epoch_idx]):
                        ax.plot(epoch, values[epoch_idx], 'ro', markersize=8)
            except ValueError:
                pass # Epoch might not be in list if capture frequency > 1

    def save_prediction_data(self) -> None:
        """
        Save the collected prediction data and validation summary to the data directory.
        Reads data from temporary files generated by _store_prediction_data.
        Saves aggregated metrics to prediction_data.txt
        """
        if not self._frame_data_paths: # Check if temp files were created
            print("Skipping data saving: No temporary data files found.")
            return

        # Ensure the data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Aggregating and saving prediction data to {self.data_dir}")

        # --- Aggregate Data from Temp Files (Load only necessary keys) ---
        all_epochs = []
        # Use metrics history for MSE as it's more reliable than predicting again
        # Fallback to prediction history if metrics history is missing
        has_mse_metric = 'mse' in self._metrics_history and len(self._metrics_history['mse']) == len(self._epoch_numbers)
        
        if has_mse_metric:
            print("Using MSE from metrics history for saving.")
            aggregated_epochs = np.array(self._epoch_numbers)
            aggregated_mse = np.array(self._metrics_history['mse'])
        else:
            print("Warning: MSE not found or length mismatch in metrics history. Trying to load from temp files.")
            # Try loading epoch and potentially calculated MSE from temp files as fallback
            loaded_epochs = []
            loaded_mse = []
            for temp_path in sorted(self._frame_data_paths, key=lambda p: int(p.stem.split('_')[-1])):
                try:
                    data = np.load(temp_path, allow_pickle=True)
                    loaded_epochs.append(data['epoch'])
                    # Check if MSE was stored during prediction step (might not exist)
                    if 'mse' in data:
                        loaded_mse.append(data['mse'])
                    else:
                        loaded_mse.append(np.nan) # Mark as missing
                except Exception as e:
                    print(f"Error loading temp data file {temp_path}: {e}")
            aggregated_epochs = np.array(loaded_epochs)
            aggregated_mse = np.array(loaded_mse)
            if np.isnan(aggregated_mse).all():
                 print("Warning: Could not retrieve valid MSE from temp files either.")
                 aggregated_mse = None # Indicate MSE is unavailable
        
        if aggregated_epochs.size == 0:
            print("Error: Could not load any epoch data.")
            return

        # --- Save Aggregated TXT Data ---
        txt_data_file = self.data_dir / "prediction_data.txt"
        try:
            with open(txt_data_file, 'w') as f:
                f.write(f"# Prediction Data Summary\n")
                f.write(f"# Function: {self.function.__class__.__name__}\n")
                f.write(f"# Algorithm: {self.algorithm_name}\n")
                f.write(f"# Epochs Recorded: {len(aggregated_epochs)}\n")
                f.write(f"# Column Headers: Epoch, MSE\n")
                f.write("# ------------------------------------\n")
                
                if aggregated_mse is not None and len(aggregated_epochs) == len(aggregated_mse):
                    for epoch, mse in zip(aggregated_epochs, aggregated_mse):
                        f.write(f"{epoch}, {mse:.8f}\n")
                else:
                     f.write("# MSE data unavailable or length mismatch.\n")
                     for epoch in aggregated_epochs:
                         f.write(f"{epoch}, nan\n") # Write epoch with NaN for MSE
                         
            print(f"Aggregated prediction metrics saved to {txt_data_file}")
        except Exception as e:
            print(f"Error saving aggregated prediction data TXT: {e}")
            
        # --- Save Validation Summary ---
        summary_file = self.data_dir / "validation_summary.txt"
        try:
            with open(summary_file, 'w') as f:
                f.write(f"Training Data Summary\n")
                f.write(f"====================\n\n")
                f.write(f"Function: {self.function.__class__.__name__}\n")
                f.write(f"Algorithm: {self.algorithm_name}\n")
                f.write(f"Input Dimension: {self.function.input_dim}\n")
                f.write(f"Output Dimension: {self.function.output_dim}\n\n")

                f.write(f"Epochs Recorded: {len(aggregated_epochs)}\n")
                # Use the aggregated MSE for summary stats
                if aggregated_mse is not None and len(aggregated_mse) > 0:
                    f.write(f"Starting MSE: {aggregated_mse[0]:.6f}\n")
                    f.write(f"Final MSE: {aggregated_mse[-1]:.6f}\n")
                    # Calculate improvement safely
                    if not np.isnan(aggregated_mse[0]) and not np.isnan(aggregated_mse[-1]):
                         improvement = aggregated_mse[0] - aggregated_mse[-1]
                         f.write(f"MSE Improvement: {improvement:.6f}\n\n")
                    else:
                         f.write("MSE Improvement cannot be calculated (NaN values present).\n\n")
                else:
                    f.write("MSE metrics not available for summary stats.\n\n")

                f.write(f"Validation Results\n")
                f.write(f"-----------------\n")
                # Assuming validation results are stored in self._validation_results
                if hasattr(self, '_validation_results') and self._validation_results:
                    for key, value in self._validation_results.items():
                        f.write(f"{key}: {value}\n")
                else:
                    f.write("Validation was not performed or results are unavailable.\n")

                f.write(f"\nDetailed MSE Progression\n")
                f.write(f"----------------------------------------------\n")
                # Use the aggregated epochs and MSE for detailed list
                if aggregated_mse is not None and len(aggregated_epochs) == len(aggregated_mse):
                     for epoch, mse in zip(aggregated_epochs, aggregated_mse):
                         f.write(f"Epoch {epoch}: MSE={mse:.6f}\n")
                else:
                     f.write("MSE data unavailable or length mismatch for detailed progression.\n")

            print(f"Validation summary saved to {summary_file}")
        except Exception as e:
             print(f"Error saving validation summary: {e}")
             
        # Clean up temporary directory
        try:
            shutil.rmtree(self._temp_data_dir)
            print(f"Cleaned up temporary data directory: {self._temp_data_dir}")
        except Exception as e:
            print(f"Error removing temporary data directory {self._temp_data_dir}: {e}")

    # Clean up __del__ to remove temp dir if object is garbage collected
    def __del__(self):
        if hasattr(self, '_temp_data_dir') and self._temp_data_dir.exists():
             try:
                 shutil.rmtree(self._temp_data_dir)
             except Exception as e:
                 # Suppress errors during cleanup
                 pass 

    def save_final_prediction_comparison(self, algorithm):
        """Saves a comparison of the final model's predictions vs true values."""
        if not self.function or not algorithm:
            print("Skipping final prediction comparison: Missing function or algorithm.")
            return

        os.makedirs(self.data_dir, exist_ok=True)
        output_file = self.data_dir / "final_predictions.csv"
        print(f"Saving final prediction comparison to {output_file}")

        try:
            # Generate test points
            if self.function.input_dim == 1:
                lower, upper = self.function.domain
                x_test = np.linspace(lower[0], upper[0], 500) # Use 500 points for 1D
                # For function call, pass the 1D array; for predict, pass 2D
                x_test_func_input = x_test 
                x_test_pred_input = x_test.reshape(-1, 1) 
                headers = ["x", "y_true", "y_pred"]
            elif self.function.input_dim == 2:
                grid_density = 30 # Use denser grid for final comparison
                lower, upper = self.function.domain
                x1 = np.linspace(lower[0], upper[0], grid_density)
                x2 = np.linspace(lower[1], upper[1], grid_density)
                X1, X2 = np.meshgrid(x1, x2)
                x_test_pred_input = np.column_stack((X1.flatten(), X2.flatten()))
                x_test_func_input = x_test_pred_input # Function expects (N, D)
                x_test = x_test_pred_input # Use the combined grid points for saving x1, x2
                headers = ["x1", "x2", "y_true", "y_pred"]
            else: 
                # Higher dimensions: Sample randomly
                n_samples = 500
                x_test_input = self.function.sample_domain(n_samples)
                x_test_func_input = x_test_input
                x_test_pred_input = x_test_input
                x_test = x_test_input
                headers = [f"x{i+1}" for i in range(self.function.input_dim)] + ["y_true", "y_pred"]

            # Get true values
            y_true = self.function(x_test_func_input) # Use appropriate input shape
            if isinstance(y_true, np.ndarray) and y_true.ndim > 1 and y_true.shape[1] == 1:
                y_true = y_true.flatten()
            elif np.isscalar(y_true): # Handle scalar output for single point case if it ever happens
                 y_true = np.array([y_true])
            
            # Get predicted values
            y_pred = algorithm.predict(x_test_pred_input) # Use appropriate input shape
            if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1 and y_pred.shape[1] == 1:
                 y_pred = y_pred.flatten()
            elif np.isscalar(y_pred):
                 y_pred = np.array([y_pred])
                 
            # Ensure consistent shapes for saving (make them all 1D arrays)
            if x_test.ndim > 1 and x_test.shape[1] == 1 and self.function.input_dim == 1: x_test = x_test.flatten()
            if y_true.ndim == 1: y_true = y_true # Already 1D
            if y_pred.ndim == 1: y_pred = y_pred # Already 1D

            # Reshape for hstack (ensure they are column vectors)
            if x_test.ndim == 1: x_test = x_test.reshape(-1, 1)
            if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
            if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)

            # For 2D input, x_test is already (N, 2), keep it that way
            if self.function.input_dim == 2:
                data_to_save = np.hstack((x_test_pred_input, y_true, y_pred))
            elif self.function.input_dim == 1:
                 data_to_save = np.hstack((x_test, y_true, y_pred))
            else: # Higher dimensions
                 data_to_save = np.hstack((x_test_pred_input, y_true, y_pred))

            # Save to CSV
            header_str = ",".join(headers)
            np.savetxt(output_file, data_to_save, delimiter=",", header=header_str, comments='')
            print(f"Successfully saved final predictions to {output_file}")

        except Exception as e:
            print(f"Error saving final prediction comparison: {e}")
            import traceback
            traceback.print_exc()

# Define output directory
RESULTS_DIR = os.path.join(script_path, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def create_functions():
    """Create a collection of functions to use for visualization."""
    return {
        "Sine": SineFunction(domain=(-3.0, 3.0)),
        "Polynomial": PolynomialFunction(a=0.5, b=-1.0, c=0.2, domain=(-2.0, 2.0)),
        "Rosenbrock": RosenbrockFunction(input_dim=2, domain_scale=2.0)
    }

def create_algorithms():
    """Create algorithm factories for each algorithm type."""
    return {
        "GeneticAlgorithm": lambda function: GeneticAlgorithm({
            "population_size": 150, # Increased population for larger hidden layer
            "generations": 50, # Initial generations for setup, main loop controls epochs
            "mutation_rate": 0.2,
            "mutation_strength": 0.1,
            "crossover_rate": 0.7,
            "tournament_size": 5, # Increased tournament size slightly
            "hidden_layer_size": 100, # Increased hidden layer size for more detail
            "elitism": True,
            "verbose": False # Reduced verbosity for cleaner output
        }),
        "QLearning": lambda function: QLearningApproximator({
            "episodes":      30,     # Reverted: Initial episodes for setup
            "alpha":         0.5,    # Increased Learning Rate
            "gamma":         0.9,    # Reverted
            "epsilon_start": 1.0,    # Reverted
            "epsilon_end":   0.01,   # Reverted
            "epsilon_decay": 0.99,   # Reverted: Faster decay
            "n_state_bins_per_dim": 20,  # Reverted: Finer grid
            "n_action_bins":        40,  # Reverted: Finer actions
            "reward_scale":   2.0,   # Reverted
            "verbose":       False
        }),
        "KNN": lambda function: KNearestNeighbors({
            "k": 3,
            "metric": "euclidean"
        }),
        "PSO": lambda function: ParticleSwarmOptimization({
            "num_particles": 20,
            "inertia_weight": 0.7,
            "cognitive_weight": 1.5,
            "social_weight": 1.5,
            "bounds": [(domain[0][0], domain[1][0]) for domain in [function.domain]],
            "function": function
        })
    }

def sample_function(function, n_samples=200):
    """Generate training data by sampling the given function."""
    # For 1D functions, sample evenly across the domain
    if function.input_dim == 1:
        lower, upper = function.domain
        # For 1D functions, use flat arrays to avoid shape issues
        X_data = np.linspace(lower[0], upper[0], n_samples)
        y_data = function(X_data)
        
        # Return 2D arrays for algorithm consumption
        X_data_2d = X_data.reshape(-1, 1)
        if isinstance(y_data, np.ndarray) and y_data.ndim == 1:
            y_data_2d = y_data.reshape(-1, 1)
        else:
            y_data_2d = y_data
            
        return X_data_2d, y_data_2d
    else:
        # For higher dimensions, sample random points within the domain
        lower, upper = function.domain
        X_data = np.random.uniform(
            low=lower,
            high=upper,
            size=(n_samples, function.input_dim)
        )
        
        # Get function outputs
        y_data = function(X_data)
        
        # Ensure output has proper shape
        if isinstance(y_data, np.ndarray) and y_data.ndim == 1:
            y_data = y_data.reshape(-1, 1)
            
        return X_data, y_data

def create_visualization(algorithm, function, function_name, algorithm_name, epochs=30):
    """
    Create a visualization of an algorithm learning a function.
    
    Args:
        algorithm: The algorithm instance
        function: The target function
        function_name: String name of the function
        algorithm_name: String name of the algorithm
        epochs: Number of training epochs
    
    Returns:
        Path to the generated GIF
    """
    print(f"\nCreating visualization for {algorithm_name} learning {function_name} function...")
    
    # Define output paths
    output_dir = os.path.join(RESULTS_DIR, function_name, algorithm_name)
    frames_dir = os.path.join(output_dir, "frames")
    gifs_dir = os.path.join(output_dir, "gifs_and_images")
    
    # Create directories
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(gifs_dir, exist_ok=True)
    
    # Create visualization config
    viz_config = {
        'fps': 8,
        'duration': 200,
        'title': f"{algorithm_name} learning {function_name} function",
        'figsize': (12, 5),
        'dpi': 100,
        'min_change_threshold': 0.0001,
        'final_mse_threshold': 100.0,
        'output_base_dir': RESULTS_DIR,
        'grid_density': 20,
        'algorithm_name': algorithm_name,
        'function_name': function_name,
        'save_data': True  # Explicitly enable data saving
    }
    
    # Create appropriate visualizer for the function dimensionality
    visualizer = FunctionVisualizer()
    
    # Create animator and recorder with our patched class
    animator = PatchedTrainingAnimator(
        function=function,
        visualizer=visualizer,
        config=viz_config
    )
    
    # Ensure the data directory exists *after* animator is created
    os.makedirs(animator.data_dir, exist_ok=True)

    recorder = TrainingRecorder(
        animator=animator,
        capture_frequency=1,
        max_frames=epochs,
        validate_data=False
    )
    
    # Sample function to get training data
    X_data, y_data = sample_function(function)
    
    # Initialize training
    initial_metrics = {} # Store metrics before first epoch
    if algorithm_name == "PSO" and hasattr(algorithm, "function") and algorithm.function is None:
        algorithm.function = function
        if hasattr(algorithm, 'evaluate_particles'):
            algorithm.evaluate_particles() # Evaluate initial PSO state
            initial_metrics = {"best_fitness": algorithm.global_best_fitness, "mse": np.nan, "mae": np.nan} 
    else:
        # Pass target_function here if needed by algorithm.train for context
        # For GA, train initializes population and evaluates it once
        algorithm.train(target_function=function, X_data=X_data, y_data=y_data)
        # Calculate initial metrics for GA (based on initial best individual)
        if algorithm_name == "GeneticAlgorithm" and algorithm.best_individual:
            y_pred_initial = algorithm.best_individual.predict(X_data)
            initial_mse = np.mean((y_pred_initial - y_data)**2)
            initial_mae = np.mean(np.abs(y_pred_initial - y_data))
            initial_metrics = {
                "best_fitness": algorithm.best_fitness,
                "mse": initial_mse,
                "mae": initial_mae
            }
        elif algorithm_name == "KNN":
             # KNN starts with 0 samples, so initial error is NaN
             initial_metrics = {"mse": np.nan, "mae": np.nan} 
        # QLearning has no meaningful initial metrics before first step
        elif algorithm_name == "QLearning":
             initial_metrics = {"mse": np.nan, "mae": np.nan, "reward": np.nan}

    # --- Record initial state (epoch -1) --- 
    if initial_metrics: # Only record if we have some initial metrics
        recorder.on_epoch_end(algorithm, -1, initial_metrics) 

    # Train for specified epochs with early stopping
    start_time = time.time()
    error_threshold = 1e-5  # Define MSE threshold for early stopping
    stopped_early = False

    for epoch in range(epochs):
        # Train one epoch
        metrics = algorithm.train_epoch(epoch)
        
        # Record frame
        recorder.on_epoch_end(algorithm, epoch, metrics)
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}/{epochs}: ", end="")
            if "mse" in metrics and not np.isnan(metrics['mse']):
                print(f"MSE={metrics['mse']:.6f}", end=" ")
            elif "best_fitness" in metrics:
                print(f"Best Fitness={metrics['best_fitness']:.6f}", end=" ")
            if "reward" in metrics:
                print(f"Reward={metrics['reward']:.4f}", end=" ")
            print()
            
        # Check for early stopping based on MSE
        if 'mse' in metrics and not np.isnan(metrics['mse']) and metrics['mse'] < error_threshold:
            print(f"\nStopping early at epoch {epoch+1} because MSE ({metrics['mse']:.6f}) < {error_threshold}")
            stopped_early = True
            break # Exit the training loop
    
    if not stopped_early:
        print(f"\nTraining completed after {epochs} epochs.")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    
    # Save the final prediction comparison *after* training loop completes
    animator.save_final_prediction_comparison(algorithm)

    # Generate GIF filename BEFORE calling create_animation
    gif_filename = f"{function_name.lower()}_{algorithm_name.lower()}_learning.gif"
    # Use the gifs_dir path defined earlier
    gif_path = os.path.join(gifs_dir, gif_filename)

    # Create animation FIRST to populate history
    full_path = animator.create_animation(
        gif_path, # Pass the correct path
        format='gif',
        force=True
    )

    # --- Save Epoch Summary Data AFTER animation creation ---
    animator.save_prediction_data() # Explicitly save epoch/MSE summary

    return full_path # full_path was defined by the earlier create_animation call

def run_visualization_suite(allowed_functions=None, allowed_algorithms=None):
    """Run the visualization suite for selected function-algorithm combinations."""
    # Create functions and algorithms
    all_functions = create_functions()
    all_algorithm_factories = create_algorithms()
    
    # --- Filter functions and algorithms based on CLI args --- 
    functions_to_run = {}
    if allowed_functions:
        for name in allowed_functions:
            if name in all_functions:
                functions_to_run[name] = all_functions[name]
            else:
                print(f"Warning: Function '{name}' not found, skipping.")
        if not functions_to_run:
            print("Error: No valid functions selected.")
            return
    else:
        functions_to_run = all_functions

    algorithm_factories_to_run = {}
    if allowed_algorithms:
        for name in allowed_algorithms:
            if name in all_algorithm_factories:
                algorithm_factories_to_run[name] = all_algorithm_factories[name]
            else:
                print(f"Warning: Algorithm '{name}' not found, skipping.")
        if not algorithm_factories_to_run:
            print("Error: No valid algorithms selected.")
        return
    else:
        algorithm_factories_to_run = all_algorithm_factories
    # --- End Filtering --- 

    # Track created visualizations
    visualizations = []
    
    # Run each selected combination
    for function_name, function in functions_to_run.items(): # Use filtered dict
        print(f"\n{'='*60}")
        print(f"Visualizing learning for {function_name} function")
        print(f"{'='*60}")
        
        for algo_name, algo_factory in algorithm_factories_to_run.items(): # Use filtered dict
            # Skip unsuitable combinations
            if function.input_dim > 1 and algo_name == "QLearning":
                print(f"Skipping {algo_name} for {function_name} (not suitable for input_dim > 1)")
                continue
                
            try:
                # Create algorithm instance
                algorithm = algo_factory(function)
                
                # Create visualization - Adjust epochs per algorithm
                if algo_name == "QLearning":
                    epochs = 300 # Increase QL epochs for better learning
                else:
                    epochs = 300 # Longer default for GA, KNN, PSO 
                    
                viz_path = create_visualization(
                    algorithm=algorithm,
                    function=function,
                    function_name=function_name,
                    algorithm_name=algo_name,
                    epochs=epochs # Pass the specific epochs
                )
                
                visualizations.append((function_name, algo_name, viz_path))
            except Exception as e:
                print(f"Error visualizing {algo_name} on {function_name}: {str(e)}")
    
    # Print summary
    print("\nCreated visualizations:")
    for func_name, algo_name, path in visualizations:
        print(f"  {func_name} + {algo_name}: {path}")

def main():
    """Main function to parse arguments and run the visualization demo."""
    if not ANIMATION_AVAILABLE:
        print("Error: Animation components not available.")
        print("Please install required packages: pip install imageio Pillow matplotlib")
        return
        
    # --- Argument Parsing --- 
    parser = argparse.ArgumentParser(description="Run AlgoGym visualization demo.")
    parser.add_argument(
        "--functions",
        nargs='+',
        metavar='FUNC_NAME',
        help="List of function names to run (e.g., Sine Polynomial Rosenbrock). Runs all if omitted."
    )
    parser.add_argument(
        "--algorithms",
        nargs='+',
        metavar='ALGO_NAME',
        help="List of algorithm names to run (e.g., GeneticAlgorithm QLearning KNN PSO). Runs all if omitted."
    )
    args = parser.parse_args()
    # --- End Argument Parsing ---

    print("Starting AlgoGym Visualization Suite...")
    if args.functions:
        print(f"Running for specific functions: {args.functions}")
    if args.algorithms:
        print(f"Running for specific algorithms: {args.algorithms}")
        
    # Pass parsed arguments to the suite runner
    run_visualization_suite(allowed_functions=args.functions, allowed_algorithms=args.algorithms)
    
    print("\nVisualization demo completed successfully!")
    print(f"Results available in: {RESULTS_DIR}")

if __name__ == "__main__":
    main() 