import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple, Union
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from algogym.functions import BaseFunction
from algogym.algorithms import BaseAlgorithm
from .base import BaseVisualizer


class FunctionVisualizer(BaseVisualizer):
    """
    Visualizer for function approximation using matplotlib.
    
    Provides visualization methods for 1D and 2D functions and their approximations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the function visualizer with optional configuration.
        
        Args:
            config (Dict[str, Any], optional): Configuration parameters for visualization.
                Possible keys:
                - 'figsize': Tuple[int, int], figure size (width, height) in inches
                - 'dpi': int, dots per inch for the figure
                - 'cmap': str, colormap name for 2D/3D plots
                - 'n_samples': int, number of samples to use for visualization
                - 'grid_density': int, density of the grid for 2D input functions
                - 'contour_levels': int, number of contour levels for 2D input functions
        """
        super().__init__(config)
        
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        default_config = {
            'figsize': (10, 6),
            'dpi': 100,
            'cmap': 'viridis',
            'n_samples': 1000,
            'grid_density': 50,
            'contour_levels': 15,
            'alpha': 0.7,  # Transparency for scatter plots
            'scatter_size': 30,  # Size of scatter points
            'line_width': 2,  # Width of lines in plots
        }
        
        # Update default with provided config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def visualize_function(self, 
                          function: BaseFunction,
                          approximation: Optional[BaseAlgorithm] = None,
                          points: Optional[np.ndarray] = None,
                          values: Optional[np.ndarray] = None,
                          title: Optional[str] = None,
                          show: bool = True,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize a function and its approximation.
        
        Supports 1D functions (input_dim=1) and 2D->1D functions (input_dim=2, output_dim=1).
        For higher dimensional functions, provides a dimensionality reduction visualization.
        
        Args:
            function (BaseFunction): The target function to visualize.
            approximation (BaseAlgorithm, optional): The algorithm that approximates the function.
            points (np.ndarray, optional): Specific input points to visualize.
            values (np.ndarray, optional): Output values corresponding to the input points.
            title (str, optional): Title for the visualization.
            show (bool): Whether to display the visualization.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            plt.Figure: The matplotlib figure object.
            
        Raises:
            ValueError: If the function dimensions are not supported for visualization.
        """
        # Set figure title
        if title is None:
            title = f"Function Visualization: {function.__class__.__name__}"
            if approximation is not None:
                title += f" vs. {approximation.__class__.__name__}"
        
        # Choose visualization method based on input and output dimensions
        if function.input_dim == 1 and function.output_dim == 1:
            return self._visualize_1d_function(function, approximation, points, values, title, show, save_path)
        elif function.input_dim == 2 and function.output_dim == 1:
            return self._visualize_2d_function(function, approximation, points, values, title, show, save_path)
        else:
            raise ValueError(f"Direct visualization not supported for functions with input_dim={function.input_dim} "
                             f"and output_dim={function.output_dim}. Consider using dimensionality reduction.")
    
    def _visualize_1d_function(self,
                              function: BaseFunction,
                              approximation: Optional[BaseAlgorithm] = None,
                              points: Optional[np.ndarray] = None,
                              values: Optional[np.ndarray] = None,
                              title: str = "1D Function Visualization",
                              show: bool = True,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize a 1D function (input_dim=1, output_dim=1) and its approximation.
        
        Args:
            function (BaseFunction): The target function to visualize.
            approximation (BaseAlgorithm, optional): The algorithm that approximates the function.
            points (np.ndarray, optional): Specific input points to visualize.
            values (np.ndarray, optional): Output values corresponding to the input points.
            title (str): Title for the visualization.
            show (bool): Whether to display the visualization.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            plt.Figure: The matplotlib figure object.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.config['figsize'], dpi=self.config['dpi'])
        
        # Generate x values for plotting the true function
        lower_bound, upper_bound = function.domain
        x = np.linspace(lower_bound[0], upper_bound[0], self.config['n_samples'])
        
        # Evaluate the true function
        y_true = function(x)
        
        # Plot the true function
        ax.plot(x, y_true, 'b-', label='True Function', linewidth=self.config['line_width'])
        
        # Plot the approximation if available
        if approximation is not None:
            try:
                y_approx = approximation.predict(x.reshape(-1, 1)).flatten()
                ax.plot(x, y_approx, 'r--', label='Approximation', linewidth=self.config['line_width'])
            except Exception as e:
                print(f"Error plotting approximation: {e}")
        
        # Plot data points if available
        if points is not None and values is not None:
            if points.ndim > 1:
                points = points.flatten()
            if values.ndim > 1:
                values = values.flatten()
            
            ax.scatter(points, values, c='g', alpha=self.config['alpha'], 
                       s=self.config['scatter_size'], label='Data Points')
        
        # Set labels and title
        ax.set_xlabel('Input (x)')
        ax.set_ylabel('Output (y)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add function details
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = f"Domain: [{lower_bound[0]:.2f}, {upper_bound[0]:.2f}]"
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Improve layout
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
        
        # Show or close the figure
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def _visualize_2d_function(self,
                             function: BaseFunction,
                             approximation: Optional[BaseAlgorithm] = None,
                             points: Optional[np.ndarray] = None,
                             values: Optional[np.ndarray] = None,
                             title: str = "2D Function Visualization",
                             show: bool = True,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize a 2D->1D function (input_dim=2, output_dim=1) and its approximation.
        
        Creates a 3D surface plot and a 2D contour plot of the function.
        
        Args:
            function (BaseFunction): The target function to visualize.
            approximation (BaseAlgorithm, optional): The algorithm that approximates the function.
            points (np.ndarray, optional): Specific input points to visualize.
            values (np.ndarray, optional): Output values corresponding to the input points.
            title (str): Title for the visualization.
            show (bool): Whether to display the visualization.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            plt.Figure: The matplotlib figure object.
        """
        # Create figure with 2 subplots: 3D surface and 2D contour
        fig = plt.figure(figsize=(self.config['figsize'][0] * 2, self.config['figsize'][1]), 
                         dpi=self.config['dpi'])
        
        # 3D surface plot
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
        # 2D contour plot
        ax2 = fig.add_subplot(1, 2, 2)
        
        # Generate grid for plotting
        lower_bound, upper_bound = function.domain
        density = self.config['grid_density']
        x1 = np.linspace(lower_bound[0], upper_bound[0], density)
        x2 = np.linspace(lower_bound[1], upper_bound[1], density)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Prepare grid inputs for function evaluation
        grid_inputs = np.column_stack((X1.flatten(), X2.flatten()))
        
        # Evaluate true function on grid
        z_true = function(grid_inputs).reshape(density, density)
        
        # Plot true function as surface
        surf1 = ax1.plot_surface(X1, X2, z_true, cmap=self.config['cmap'], alpha=0.8,
                              linewidth=0, antialiased=True)
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5, label='Function Value')
        
        # Plot true function as contour
        contour1 = ax2.contourf(X1, X2, z_true, levels=self.config['contour_levels'],
                              cmap=self.config['cmap'])
        fig.colorbar(contour1, ax=ax2, shrink=0.5, aspect=5, label='Function Value')
        
        # Plot approximation if available
        if approximation is not None:
            try:
                # Evaluate approximation on grid
                z_approx = approximation.predict(grid_inputs).reshape(density, density)
                
                # Add approximation as wireframe to surface plot
                ax1.plot_wireframe(X1, X2, z_approx, color='r', linewidth=0.5, 
                                 alpha=0.7, label='Approximation')
                
                # Add approximation as contour lines to contour plot
                contour2 = ax2.contour(X1, X2, z_approx, levels=self.config['contour_levels'],
                                     colors='r', linewidths=2, alpha=0.7)
                ax2.clabel(contour2, inline=True, fontsize=8, fmt='%.1f')
            except Exception as e:
                print(f"Error plotting approximation: {e}")
        
        # Plot data points if available
        if points is not None and values is not None:
            if points.shape[1] == 2:  # Ensure points are 2D
                # For 3D plot
                ax1.scatter(points[:, 0], points[:, 1], values, c='g', 
                          marker='o', s=self.config['scatter_size'],
                          alpha=self.config['alpha'], label='Data Points')
                
                # For contour plot - size by value for visibility
                scatter_size = values.flatten() / max(values.flatten()) * self.config['scatter_size'] * 3
                scatter_size = np.maximum(scatter_size, self.config['scatter_size'] * 0.5)  # Minimum size
                ax2.scatter(points[:, 0], points[:, 1], c='g', s=scatter_size,
                          alpha=self.config['alpha'], label='Data Points')
        
        # Set labels and title
        ax1.set_xlabel('x₁')
        ax1.set_ylabel('x₂')
        ax1.set_zlabel('f(x₁, x₂)')
        ax1.set_title(f"{title} - 3D Surface")
        
        ax2.set_xlabel('x₁')
        ax2.set_ylabel('x₂')
        ax2.set_title(f"{title} - Contour")
        
        # Add legend and grid
        ax1.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add function details text box
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        textstr = (f"Domain: x₁∈[{lower_bound[0]:.2f}, {upper_bound[0]:.2f}], "
                  f"x₂∈[{lower_bound[1]:.2f}, {upper_bound[1]:.2f}]")
        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        # Improve layout
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
        
        # Show or close the figure
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def visualize_progress(self,
                          metrics_history: Dict[str, List[float]],
                          iterations: Optional[List[int]] = None,
                          title: Optional[str] = None,
                          show: bool = True,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize the progress of an algorithm over iterations using metrics.
        
        Args:
            metrics_history (Dict[str, List[float]]): Dictionary mapping metric names to 
                                                    lists of metric values over iterations.
            iterations (List[int], optional): List of iteration numbers. If None, uses
                                            sequential indices starting from 0.
            title (str, optional): Title for the visualization.
            show (bool): Whether to display the visualization.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            plt.Figure: The matplotlib figure object.
        """
        # Default title
        if title is None:
            title = "Algorithm Progress"
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config['figsize'], dpi=self.config['dpi'])
        
        # Set x-axis values (iterations)
        if iterations is None:
            # Find the maximum length of any metric history
            max_len = max(len(values) for values in metrics_history.values())
            iterations = list(range(max_len))
        
        # Plot each metric
        for metric_name, metric_values in metrics_history.items():
            # Ensure metric_values length matches iterations length
            valid_len = min(len(metric_values), len(iterations))
            ax.plot(iterations[:valid_len], metric_values[:valid_len], 
                  linewidth=self.config['line_width'], marker='o', 
                  markersize=4, label=metric_name)
        
        # Set labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Use log scale if values cover multiple orders of magnitude
        for metric_values in metrics_history.values():
            if len(metric_values) > 0:
                max_val = max(metric_values)
                min_val = min(v for v in metric_values if v > 0) if any(v > 0 for v in metric_values) else max_val
                if max_val > 0 and min_val > 0 and max_val / min_val > 100:
                    ax.set_yscale('log')
                    break
        
        # Add annotations for best values
        for metric_name, metric_values in metrics_history.items():
            if len(metric_values) > 0:
                best_idx = np.argmin(metric_values)
                best_value = metric_values[best_idx]
                best_iter = iterations[best_idx] if best_idx < len(iterations) else best_idx
                
                ax.annotate(f'Best: {best_value:.4g}',
                          xy=(best_iter, best_value),
                          xytext=(10, 10),
                          textcoords='offset points',
                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        # Improve layout
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
        
        # Show or close the figure
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
    
    def visualize_comparison(self,
                            functions: List[BaseFunction],
                            approximations: List[BaseAlgorithm],
                            labels: Optional[List[str]] = None,
                            title: Optional[str] = None,
                            show: bool = True,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize and compare multiple function approximations.
        
        Currently supports comparison of 1D functions (input_dim=1, output_dim=1).
        
        Args:
            functions (List[BaseFunction]): List of target functions to visualize.
            approximations (List[BaseAlgorithm]): List of algorithms that approximate the functions.
            labels (List[str], optional): Labels for each function-approximation pair.
            title (str, optional): Title for the visualization.
            show (bool): Whether to display the visualization.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            plt.Figure: The matplotlib figure object.
            
        Raises:
            ValueError: If the function dimensions are not supported for comparison.
        """
        # Validate inputs
        if len(functions) != len(approximations):
            raise ValueError("The number of functions must match the number of approximations.")
        
        n_comparisons = len(functions)
        
        # Generate default labels if not provided
        if labels is None:
            labels = [f"Comparison {i+1}" for i in range(n_comparisons)]
        elif len(labels) != n_comparisons:
            raise ValueError("The number of labels must match the number of functions.")
        
        # Default title
        if title is None:
            title = "Function Approximation Comparison"
        
        # Check that all functions have the same input dimension
        input_dims = [func.input_dim for func in functions]
        if len(set(input_dims)) != 1:
            raise ValueError("All functions must have the same input dimension for comparison.")
        
        input_dim = input_dims[0]
        
        # Choose visualization method based on input dimension
        if input_dim == 1:
            return self._visualize_1d_comparison(functions, approximations, labels, title, show, save_path)
        else:
            raise ValueError(f"Comparison visualization not supported for functions with input_dim={input_dim}.")
    
    def _visualize_1d_comparison(self,
                               functions: List[BaseFunction],
                               approximations: List[BaseAlgorithm],
                               labels: List[str],
                               title: str,
                               show: bool,
                               save_path: Optional[str]) -> plt.Figure:
        """
        Visualize and compare multiple 1D function approximations.
        
        Args:
            functions (List[BaseFunction]): List of 1D functions to visualize.
            approximations (List[BaseAlgorithm]): List of algorithms that approximate the functions.
            labels (List[str]): Labels for each function-approximation pair.
            title (str): Title for the visualization.
            show (bool): Whether to display the visualization.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            plt.Figure: The matplotlib figure object.
        """
        n_comparisons = len(functions)
        
        # Create figure with subplots
        n_cols = min(2, n_comparisons)
        n_rows = (n_comparisons + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(self.config['figsize'][0] * n_cols,
                                                      self.config['figsize'][1] * n_rows),
                               dpi=self.config['dpi'], squeeze=False)
        
        # Flatten axes for easy iteration
        axes = axes.flatten()
        
        # Create color cycle for consistent colors across subplots
        colors = plt.cm.tab10.colors
        
        # Plot each comparison
        for i, (func, algo, label, ax) in enumerate(zip(functions, approximations, labels, axes)):
            # Generate x values for plotting
            lower_bound, upper_bound = func.domain
            x = np.linspace(lower_bound[0], upper_bound[0], self.config['n_samples'])
            
            # Evaluate the true function
            y_true = func(x)
            
            # Plot the true function
            ax.plot(x, y_true, color=colors[0], linestyle='-', 
                  label='True Function', linewidth=self.config['line_width'])
            
            # Plot the approximation
            try:
                y_approx = algo.predict(x.reshape(-1, 1)).flatten()
                ax.plot(x, y_approx, color=colors[1], linestyle='--', 
                      label='Approximation', linewidth=self.config['line_width'])
                
                # Calculate and display error metrics
                mse = np.mean((y_true - y_approx) ** 2)
                mae = np.mean(np.abs(y_true - y_approx))
                
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                textstr = f"MSE: {mse:.4g}\nMAE: {mae:.4g}"
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                      verticalalignment='top', bbox=props)
                
            except Exception as e:
                print(f"Error plotting approximation for {label}: {e}")
            
            # Set labels and title
            ax.set_xlabel('Input (x)')
            ax.set_ylabel('Output (y)')
            ax.set_title(f"{label}")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        # Set overall title
        fig.suptitle(title, fontsize=16)
        
        # Improve layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Leave room for the suptitle
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
        
        # Show or close the figure
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return fig 