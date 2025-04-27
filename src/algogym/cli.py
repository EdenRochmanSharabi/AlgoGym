#!/usr/bin/env python3
"""
AlgoGym CLI

Command Line Interface for training and visualizing function approximation algorithms.
"""

import argparse
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import inspect

# Import AlgoGym components
from algogym.functions import BaseFunction
from algogym.algorithms import BaseAlgorithm
import algogym.functions
import algogym.algorithms
from algogym.visualization.animation import TrainingAnimator, TrainingRecorder

# Import specific functions and algorithms for the dictionaries
from algogym.functions import (
    SineFunction,
    PolynomialFunction,
    RosenbrockFunction
)
from algogym.algorithms import (
    GeneticAlgorithm,
    KNearestNeighbors, 
    ParticleSwarmOptimization,
    QLearningApproximator
)

# Try to import animation components
try:
    import imageio.v2 as imageio
    import matplotlib.animation
    ANIMATION_AVAILABLE = True
except ImportError:
    ANIMATION_AVAILABLE = False
    print("Animation components not available. Install imageio to enable.")

# Dictionary of available functions
AVAILABLE_FUNCTIONS = {
    'sine': {
        'class': SineFunction,
        'display_name': 'Sine Function',
        'difficulty': 'Easy',
        'default_params': {}
    },
    'polynomial': {
        'class': PolynomialFunction,
        'display_name': 'Polynomial Function',
        'difficulty': 'Easy',
        'default_params': {}
    },
    'rosenbrock': {
        'class': RosenbrockFunction,
        'display_name': 'Rosenbrock Function',
        'difficulty': 'Hard',
        'default_params': {}
    }
}

# Dictionary of available algorithms
AVAILABLE_ALGORITHMS = {
    'genetic': {
        'class': GeneticAlgorithm,
        'display_name': 'Genetic Algorithm',
        'default_params': {
            'population_size': 50,
            'generations': 100,
            'mutation_rate': 0.1,
            'crossover_rate': 0.7
        }
    },
    'knn': {
        'class': KNearestNeighbors,
        'display_name': 'K-Nearest Neighbors',
        'default_params': {
            'k': 3
        }
    },
    'pso': {
        'class': ParticleSwarmOptimization,
        'display_name': 'Particle Swarm Optimization',
        'default_params': {
            'num_particles': 30,
            'inertia_weight': 0.7,
            'cognitive_coeff': 1.5,
            'social_coeff': 1.5
        }
    },
    'qlearning': {
        'class': QLearningApproximator,
        'display_name': 'Q-Learning Approximator',
        'default_params': {
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'exploration_rate': 0.2
        }
    }
}

# Define a GradualLearningAlgorithm class here temporarily to match the algorithm name in CLI
class GradualLearningAlgorithm(BaseAlgorithm):
    """
    A simple algorithm that gradually learns to approximate a function.
    
    This is a placeholder for CLI compatibility.
    """
    
    def __init__(self, config=None):
        """
        Initialize the gradual learning algorithm.
        """
        config = config or {}
        super().__init__(config)
        
    def train(self, target_function=None, X_data=None, y_data=None):
        """Train the algorithm on a target function."""
        pass
        
    def train_epoch(self, epoch):
        """Train for one epoch."""
        return {"MSE": 0.0}
        
    def predict(self, x):
        """Make predictions."""
        return np.zeros_like(x)

def list_available_functions():
    """Print a table of available functions."""
    print("\nAVAILABLE FUNCTIONS:")
    print("=" * 80)
    print(f"{'Name':<15} {'Difficulty':<10} {'Description':<55}")
    print("-" * 80)
    for key, info in AVAILABLE_FUNCTIONS.items():
        print(f"{key:<15} {info['difficulty']:<10} {info['display_name']:<55}")
    print("=" * 80)

def list_available_algorithms():
    """Print a table of available algorithms."""
    print("\nAVAILABLE ALGORITHMS:")
    print("=" * 80)
    print(f"{'Name':<15} {'Description':<65}")
    print("-" * 80)
    for key, info in AVAILABLE_ALGORITHMS.items():
        print(f"{key:<15} {info['display_name']:<65}")
    print("=" * 80)

def create_function(func_name, params=None):
    """Create a function by name with optional parameters."""
    params = params or {}
    
    # Look up function in available functions
    function_class = None
    
    # Try direct class lookup first
    for name, cls in inspect.getmembers(algogym.functions, inspect.isclass):
        if name.lower() == func_name.lower() or name.lower() == f"{func_name.lower()}function":
            if issubclass(cls, BaseFunction) and cls != BaseFunction:
                function_class = cls
                break
    
    # If not found, look in AVAILABLE_FUNCTIONS dictionary
    if function_class is None and func_name in AVAILABLE_FUNCTIONS:
        function_class = AVAILABLE_FUNCTIONS[func_name]['class']
    
    if function_class is None:
        print(f"Function '{func_name}' not found.")
        return None
    
    try:
        return function_class(**params)
    except Exception as e:
        print(f"Error creating function '{func_name}': {str(e)}")
        return None

def create_algorithm(algo_name, params=None):
    """Create an algorithm by name with optional parameters."""
    params = params or {}
    
    # Look up algorithm in available algorithms
    algorithm_class = None
    
    # Try direct class lookup first
    for name, cls in inspect.getmembers(algogym.algorithms, inspect.isclass):
        if name.lower() == algo_name.lower() or name.lower() == f"{algo_name.lower()}algorithm":
            if issubclass(cls, BaseAlgorithm) and cls != BaseAlgorithm:
                algorithm_class = cls
                break
    
    # If not found, look in AVAILABLE_ALGORITHMS dictionary
    if algorithm_class is None and algo_name in AVAILABLE_ALGORITHMS:
        algorithm_class = AVAILABLE_ALGORITHMS[algo_name]['class']
    
    if algorithm_class is None:
        print(f"Algorithm '{algo_name}' not found.")
        return None
    
    try:
        # Add minimum required parameters to ensure it works
        # GeneticAlgorithm needs population_size
        if algorithm_class == GeneticAlgorithm and 'population_size' not in params:
            params['population_size'] = 50
            
        # Create a config dictionary for the algorithm
        config = {}
        if algo_name in AVAILABLE_ALGORITHMS:
            # Start with default params from the dictionary
            config.update(AVAILABLE_ALGORITHMS[algo_name]['default_params'])
        
        # Override with user-provided params
        config.update(params)
        
        # Create the algorithm with the config
        return algorithm_class(config)
    except Exception as e:
        print(f"Error creating algorithm '{algo_name}': {str(e)}")
        return None

def train_algorithm(
    function: BaseFunction, 
    algorithm: BaseAlgorithm, 
    epochs: int, 
    create_animation: bool = False,
    output_dir: str = 'output',
    verbose: bool = True
) -> Tuple[Dict[str, List[float]], Optional[str]]:
    """
    Train an algorithm on a function and optionally create an animation.
    
    Args:
        function: The target function
        algorithm: The algorithm to train
        epochs: Number of training epochs
        create_animation: Whether to create an animation
        output_dir: Directory to save outputs
        verbose: Whether to print progress messages
    
    Returns:
        Tuple of (metrics history, animation path or None)
    """
    # Create algorithm-specific output directory
    algorithm_name = algorithm.__class__.__name__
    algorithm_output_dir = os.path.join(output_dir, algorithm_name)
    os.makedirs(algorithm_output_dir, exist_ok=True)
    
    # Initialize training
    algorithm.train(target_function=function)
    
    # Setup animation if requested
    if create_animation and ANIMATION_AVAILABLE:
        animator = TrainingAnimator(
            function=function,
            config={
                'fps': 10,
                'duration': 200,  # 200ms per frame
                'title': f"{algorithm_name} on {function.__class__.__name__}",
                'figsize': (12, 5)
            }
        )
        
        recorder = TrainingRecorder(
            animator=animator,
            capture_frequency=1,  # Capture every epoch
            max_frames=epochs  # Cap at the number of epochs
        )
    elif create_animation and not ANIMATION_AVAILABLE:
        print("Warning: Animation requested but required components not available.")
        print("Install imageio with: pip install imageio>=2.19.0")
        create_animation = False
    
    # Train for the specified number of epochs
    if verbose:
        print(f"Training algorithm on {function.__class__.__name__} for {epochs} epochs...")
    
    start_time = time.time()
    all_metrics = {}
    
    for epoch in range(epochs):
        # Train one epoch
        metrics = algorithm.train_epoch(epoch)
        
        # Store metrics
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            
            # Only store scalar metrics in the history
            if np.isscalar(value):
                all_metrics[key].append(float(value))
        
        # Capture frame for animation if requested
        if create_animation and ANIMATION_AVAILABLE:
            recorder.on_epoch_end(algorithm, epoch, metrics)
        
        # Print progress every 10 epochs (or less for small epoch counts)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            metrics_str = ', '.join([f"{k}: {v:.6f}" for k, v in metrics.items() 
                                  if np.isscalar(v)])
            print(f"  Epoch {epoch}/{epochs-1}: {metrics_str}")
    
    training_time = time.time() - start_time
    if verbose:
        print(f"Training completed in {training_time:.2f} seconds")
    
    # Create and save animation if requested
    animation_path = None
    if create_animation and ANIMATION_AVAILABLE:
        if verbose:
            print("Creating animation...")
        
        # Generate a unique filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        function_name = function.__class__.__name__.lower()
        filename = f"{function_name}_learning_{timestamp}.gif"
        animation_path = os.path.join(algorithm_output_dir, filename)
        
        # Create the animation
        animator.create_animation(animation_path, format='gif')
        
        if verbose:
            print(f"Animation saved to {animation_path}")
    
    return all_metrics, animation_path

def plot_metrics(metrics, title=None, output_path=None, algorithm_name=None, function_name=None):
    """Plot training metrics and optionally save to file."""
    plt.figure(figsize=(10, 6))
    
    for metric_name, values in metrics.items():
        if metric_name in ('MSE', 'MAE'):  # Only plot scalar metrics
            plt.plot(values, marker='.', label=metric_name)
    
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    
    if title:
        plt.title(title)
    else:
        plt.title('Training Metrics')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use logarithmic scale if values span multiple orders of magnitude
    for values in metrics.values():
        if len(values) > 1:
            max_val = max(values)
            min_val = min(v for v in values if v > 0) if any(v > 0 for v in values) else max_val
            if max_val > 0 and min_val > 0 and max_val / min_val > 100:
                plt.yscale('log')
                break
    
    # Save if path provided
    if output_path:
        plt.savefig(output_path)
        print(f"Metrics plot saved to {output_path}")
    
    plt.close()

def compare_algorithms(
    function: BaseFunction,
    algorithms: Dict[str, BaseAlgorithm],
    epochs: int,
    output_dir: str = 'output',
    verbose: bool = True
) -> Dict[str, Dict[str, List[float]]]:
    """
    Train multiple algorithms on the same function and compare results.
    
    Args:
        function: The target function
        algorithms: Dictionary mapping algorithm names to algorithm instances
        epochs: Number of training epochs
        output_dir: Directory to save outputs
        verbose: Whether to print progress messages
    
    Returns:
        Dictionary mapping algorithm names to their metrics history
    """
    # Ensure output directory exists
    compare_dir = os.path.join(output_dir, "comparison")
    os.makedirs(compare_dir, exist_ok=True)
    
    # Train each algorithm and collect metrics
    all_algorithm_metrics = {}
    
    for algo_name, algorithm in algorithms.items():
        if verbose:
            print(f"\nTraining {algo_name}...")
        
        # Create algorithm-specific output directory
        algo_output_dir = os.path.join(output_dir, algorithm.__class__.__name__)
        os.makedirs(algo_output_dir, exist_ok=True)
        
        # Train the algorithm
        metrics, _ = train_algorithm(
            function=function,
            algorithm=algorithm,
            epochs=epochs,
            create_animation=False,  # No animation for comparison
            output_dir=output_dir,
            verbose=verbose
        )
        
        all_algorithm_metrics[algo_name] = metrics
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Plot MSE for each algorithm
    for algo_name, metrics in all_algorithm_metrics.items():
        if 'MSE' in metrics:
            plt.plot(metrics['MSE'], marker='.', label=f"{algo_name} - MSE")
    
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title(f'Algorithm Comparison on {function.__class__.__name__}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    
    # Save comparison plot
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    function_name = function.__class__.__name__.lower()
    comparison_plot_path = os.path.join(compare_dir, f"comparison_{function_name}_{timestamp}.png")
    plt.savefig(comparison_plot_path)
    
    if verbose:
        print(f"\nComparison plot saved to {comparison_plot_path}")
    
    plt.close()
    
    return all_algorithm_metrics

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AlgoGym CLI - Train and visualize function approximation algorithms",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Command mode
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List available functions and algorithms
    list_parser = subparsers.add_parser('list', help='List available functions and algorithms')
    
    # Train a single algorithm on a function
    train_parser = subparsers.add_parser('train', help='Train an algorithm on a function')
    train_parser.add_argument('--function', '-f', default='sine', 
                             help='Function to approximate (default: sine)')
    train_parser.add_argument('--algorithm', '-a', default='genetic',
                             help='Algorithm to train (default: genetic)')
    train_parser.add_argument('--epochs', '-e', type=int, default=50,
                             help='Number of training epochs (default: 50)')
    train_parser.add_argument('--learning-rate', '-lr', type=float, default=0.1,
                             help='Learning rate for the algorithm (default: 0.1)')
    train_parser.add_argument('--noise-level', '-n', type=float, default=0.05,
                             help='Noise level for the algorithm (default: 0.05)')
    train_parser.add_argument('--model-type', '-mt', default='polynomial',
                             help='Model type for algorithm (polynomial, fourier) (default: polynomial)')
    train_parser.add_argument('--model-complexity', '-mc', type=int, default=5,
                             help='Model complexity (default: 5)')
    train_parser.add_argument('--output-dir', '-o', default='output',
                             help='Directory to save outputs (default: output)')
    train_parser.add_argument('--animate', '-ani', action='store_true',
                             help='Create an animation of the training process')
    train_parser.add_argument('--quiet', '-q', action='store_true',
                             help='Suppress progress output')
    
    # Compare multiple algorithms on a function
    compare_parser = subparsers.add_parser('compare', help='Compare multiple algorithms on a function')
    compare_parser.add_argument('--function', '-f', default='sine',
                               help='Function to approximate (default: sine)')
    compare_parser.add_argument('--algorithms', '-a', nargs='+', default=['genetic'],
                               help='Algorithms to compare (default: genetic)')
    compare_parser.add_argument('--epochs', '-e', type=int, default=50,
                               help='Number of training epochs (default: 50)')
    compare_parser.add_argument('--output-dir', '-o', default='output',
                               help='Directory to save outputs (default: output)')
    compare_parser.add_argument('--quiet', '-q', action='store_true',
                               help='Suppress progress output')
    
    # All-in-one command to train an algorithm on all functions
    all_parser = subparsers.add_parser('all', help='Train algorithm on all functions')
    all_parser.add_argument('--algorithm', '-a', default='genetic',
                           help='Algorithm to train (default: genetic)')
    all_parser.add_argument('--epochs', '-e', type=int, default=30,
                           help='Number of training epochs per function (default: 30)')
    all_parser.add_argument('--output-dir', '-o', default='output',
                           help='Directory to save outputs (default: output)')
    all_parser.add_argument('--animate', '-ani', action='store_true',
                           help='Create animations of the training process')
    all_parser.add_argument('--quiet', '-q', action='store_true',
                           help='Suppress progress output')
    
    # Check frames command
    verify_parser = subparsers.add_parser('verify-frames', help='Verify animation frames')
    verify_parser.add_argument('--frames-dir', '-d', 
                              help='Path to frames directory. If not provided, will use the most recent frames directory.')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'list':
        print("Available Functions:")
        list_available_functions()
        print("\nAvailable Algorithms:")
        list_available_algorithms()
        
    elif args.command == 'train':
        # Create function
        function = create_function(args.function)
        if function is None:
            print(f"Error: Function '{args.function}' not found.")
            return 1
        
        # Create algorithm
        params = {
            'learning_rate': args.learning_rate,
            'noise_level': args.noise_level,
            'model_type': args.model_type,
            'model_complexity': args.model_complexity
        }
        algorithm = create_algorithm(args.algorithm, params)
        if algorithm is None:
            print(f"Error: Algorithm '{args.algorithm}' not found.")
            return 1
        
        # Train algorithm
        metrics, animation_path = train_algorithm(
            function=function,
            algorithm=algorithm,
            epochs=args.epochs,
            create_animation=args.animate,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
        
        # Output the results
        if not args.quiet:
            print("\nTraining complete.")
            if animation_path:
                print(f"Animation saved to: {animation_path}")
                
    elif args.command == 'compare':
        # Create function
        function = create_function(args.function)
        if function is None:
            print(f"Error: Function '{args.function}' not found.")
            return 1
        
        # Create algorithms
        algorithms = {}
        for algo_name in args.algorithms:
            algorithm = create_algorithm(algo_name)
            if algorithm is None:
                print(f"Warning: Algorithm '{algo_name}' not found, skipping.")
                continue
            algorithms[algo_name] = algorithm
        
        if not algorithms:
            print("Error: No valid algorithms specified.")
            return 1
        
        # Compare algorithms
        compare_algorithms(
            function=function,
            algorithms=algorithms,
            epochs=args.epochs,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
        
    elif args.command == 'all':
        # Create algorithm
        algorithm = create_algorithm(args.algorithm)
        if algorithm is None:
            print(f"Error: Algorithm '{args.algorithm}' not found.")
            return 1
        
        # Get all available functions
        all_functions = {}
        for name, cls in inspect.getmembers(algogym.functions, inspect.isclass):
            if issubclass(cls, BaseFunction) and cls != BaseFunction:
                all_functions[name] = cls
        
        # Train on each function
        for func_name, func_class in all_functions.items():
            print(f"\nTraining on {func_name}...")
            function = func_class()
            
            metrics, animation_path = train_algorithm(
                function=function,
                algorithm=algorithm,
                epochs=args.epochs,
                create_animation=args.animate,
                output_dir=args.output_dir,
                verbose=not args.quiet
            )
            
    elif args.command == 'verify-frames':
        # Verify animation frames
        from algogym.visualization.animation import TrainingAnimator
        from algogym.functions import SineFunction
        
        # Create a temporary animator
        animator = TrainingAnimator(function=SineFunction())
        
        # Verify the frames
        if args.frames_dir:
            valid = animator.verify_frames(frames_dir=args.frames_dir)
        else:
            valid = animator.verify_frames()
            
        # Return appropriate exit code
        return 0 if valid else 1
    
    else:
        parser.print_help()
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 