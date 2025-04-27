import numpy as np
from typing import Any, Dict, List, Tuple, Optional, Union
import random

from .base import BaseAlgorithm

class Particle:
    """
    Class representing a particle in the swarm.
    Each particle has:
    - A position vector (potential solution)
    - A velocity vector (direction and speed of movement)
    - Personal best position found so far
    - Personal best fitness value
    """
    def __init__(self, bounds: List[Tuple[float, float]], dim: int):
        """
        Initialize a particle with random position and velocity.
        
        Args:
            bounds: List of tuples (min, max) for each dimension
            dim: Number of dimensions for the particle
        """
        # Generate random position within bounds
        self.position = np.zeros(dim)
        for i in range(dim):
            self.position[i] = random.uniform(bounds[i][0], bounds[i][1])
        
        # Initialize velocity as random values scaled by bounds range
        self.velocity = np.zeros(dim)
        for i in range(dim):
            bound_range = bounds[i][1] - bounds[i][0]
            self.velocity[i] = random.uniform(-0.1, 0.1) * bound_range
        
        # Initialize personal best
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')  # Initialize with worst possible fitness (we're minimizing)
        self.fitness = float('inf')
    
    def update_personal_best(self, fitness: float):
        """Update the particle's personal best if current position is better."""
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position.copy()
            
    def update_position(self, bounds: List[Tuple[float, float]]):
        """
        Update particle position based on velocity and constrain within bounds.
        
        Args:
            bounds: List of tuples (min, max) for each dimension
        """
        self.position += self.velocity
        
        # Constrain position to be within bounds
        for i in range(len(self.position)):
            if self.position[i] < bounds[i][0]:
                self.position[i] = bounds[i][0]
                self.velocity[i] *= -0.5  # Bounce back with reduced velocity
            elif self.position[i] > bounds[i][1]:
                self.position[i] = bounds[i][1]
                self.velocity[i] *= -0.5  # Bounce back with reduced velocity

class ParticleSwarmOptimization(BaseAlgorithm):
    """
    Particle Swarm Optimization (PSO) algorithm implementation.
    
    PSO is a population-based optimization algorithm inspired by social behavior
    of bird flocking or fish schooling. Particles move in the search space guided
    by their personal best position and the global best position found by the swarm.
    """
    DEFAULT_CONFIG = {
        "num_particles": 30,
        "inertia_weight": 0.7,
        "cognitive_weight": 1.5,
        "social_weight": 1.5,
        "bounds": [(-5.0, 5.0)],  # Default bounds for each dimension
    }
    
    def __init__(self, config: Dict[str, Any] | None = None):
        """
        Initialize PSO algorithm with configuration parameters.
        
        Args:
            config: Dictionary with configuration parameters
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config if config is not None else {})}
        super().__init__(merged_config)
        
        self.function = self.config.get("function", None)
        self.bounds = self.config["bounds"]
        
        # Determine dimensionality from bounds
        if isinstance(self.bounds[0], tuple) and len(self.bounds) == 1 and self.function and hasattr(self.function, "input_dim"):
            # Expand bounds to match function input_dim if only one bound is provided
            self.bounds = self.bounds * self.function.input_dim
        
        self.dim = len(self.bounds)
        
        # Initialize particles
        self.particles = [Particle(self.bounds, self.dim) for _ in range(self.config["num_particles"])]
        
        # Initialize global best
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.best_solution = None  # Initialize best_solution property
        
        # If function is provided, evaluate initial positions
        if self.function:
            self.evaluate_particles()
    
    def evaluate_particles(self):
        """Evaluate fitness of all particles using the objective function."""
        if self.function is None:
            raise ValueError("No objective function provided for evaluation")
        
        for particle in self.particles:
            # Evaluate fitness at current position
            fitness = self.function(particle.position)
            particle.fitness = fitness
            
            # Update personal best
            particle.update_personal_best(fitness)
            
            # Update global best
            if fitness < self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = particle.position.copy()
    
    def update_velocities(self):
        """Update velocities of all particles."""
        inertia = self.config["inertia_weight"]
        cognitive = self.config["cognitive_weight"]
        social = self.config["social_weight"]
        
        for particle in self.particles:
            # Generate random factors
            r1 = np.random.random(self.dim)
            r2 = np.random.random(self.dim)
            
            # Calculate new velocity
            cognitive_component = cognitive * r1 * (particle.best_position - particle.position)
            social_component = social * r2 * (self.global_best_position - particle.position)
            
            # Update velocity with inertia
            particle.velocity = inertia * particle.velocity + cognitive_component + social_component
    
    def update_particles(self):
        """Update velocities and positions of all particles, then evaluate them."""
        # Update velocities
        self.update_velocities()
        
        # Update positions
        for particle in self.particles:
            particle.update_position(self.bounds)
        
        # Evaluate particles
        self.evaluate_particles()
    
    def train(self, target_function=None, X_data=None, y_data=None, epochs=100):
        """
        Train the PSO algorithm to find optimal solution.
        
        Args:
            target_function: Function to optimize (if not provided at initialization)
            X_data: Not used directly in PSO - for API compatibility
            y_data: Not used directly in PSO - for API compatibility
            epochs: Number of iterations to run
            
        Returns:
            Dictionary with training results
        """
        if target_function is not None:
            self.function = target_function
        
        if self.function is None:
            raise ValueError("No target function provided for training")
        
        # Run optimization for specified number of epochs
        for epoch in range(epochs):
            # Use the train_epoch method to update for one epoch
            metrics = self.train_epoch(epoch)
            
            # Print progress every 10 epochs or for first/last epoch
            if epoch % 10 == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs}: Best fitness = {metrics['best_fitness']:.6f}")
        
        self.best_solution = self.global_best_position
        
        # Return results
        return {
            "best_position": self.global_best_position.tolist(),
            "best_fitness": float(self.global_best_fitness)
        }
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Performs a single epoch of PSO training.
        
        Updates all particles' velocities and positions, then evaluates them.
        
        Args:
            epoch (int): The current epoch number.
            
        Returns:
            Dict[str, float]: Dictionary containing metrics for this epoch.
        """
        # Store previous best fitness for calculating improvement
        previous_best_fitness = self.global_best_fitness
        
        # Update particles for one epoch
        self.update_particles()
        
        # Calculate improvement
        improvement = previous_best_fitness - self.global_best_fitness
        
        # Prepare particle statistics
        particle_velocities = np.array([np.linalg.norm(p.velocity) for p in self.particles])
        particle_fitness_values = np.array([p.fitness for p in self.particles])
        
        # --- Calculate MSE/MAE using the global best position --- 
        # PSO minimizes the function, so fitness *is* the objective value at the best position.
        # We need a representative dataset (e.g., from initialization or sampled) 
        # to calculate approximation errors like MSE/MAE against.
        # This requires storing X_data/y_data or the function itself during init/train.
        # For now, return NaN as PSO doesn't inherently do approximation.
        mse = float('nan')
        mae = float('nan')
        
        # --- TODO: If PSO needs to report MSE/MAE for function *approximation* tasks: ---
        # 1. Modify `train` to accept and store `X_data`, `y_data` like other algorithms.
        # 2. Use `self.predict(self._X_data)` with the current best position to get `y_pred`.
        # 3. Calculate MSE/MAE: mse = np.mean((self._y_data - y_pred)**2), mae = np.mean(np.abs(self._y_data - y_pred))
        # ----------------------------------------------------------------------------------

        # Return metrics
        return {
            "best_fitness": float(self.global_best_fitness), # Objective value at best position
            "mse": mse,
            "mae": mae,
            "improvement": float(improvement),
            "mean_fitness": float(np.mean(particle_fitness_values)),
            "mean_velocity": float(np.mean(particle_velocities)),
            "position_norm": float(np.linalg.norm(self.global_best_position)) if self.global_best_position is not None else 0.0,
            "epoch": epoch
        }
    
    def get_global_best_position(self):
        """Return the global best position found by the swarm."""
        return self.global_best_position
    
    def get_global_best_fitness(self):
        """Return the global best fitness value found by the swarm."""
        return self.global_best_fitness
    
    def predict(self, X):
        """
        Make predictions using the best solution found.
        
        Args:
            X: Input data points as numpy array
            
        Returns:
            Predicted values
        """
        if not hasattr(self, 'best_solution') or self.best_solution is None:
            raise RuntimeError("No best position found. Make sure to train the algorithm first.")
        
        if self.function is None:
            raise ValueError("No target function is set for prediction")
        
        # For PSO, we don't build a model - we just find the best position to minimize the function
        # So we evaluate the actual function on the input
        
        # Save original shape for reshaping at the end
        original_shape = X.shape
        original_ndim = X.ndim
        
        # Handle 1D inputs
        if original_ndim == 1:
            # For 1D inputs, the function expects a 1D array
            try:
                result = self.function(X)
                # Make sure results match the original shape of X
                if np.isscalar(result):
                    return result
                return result.reshape(-1)
            except Exception as e:
                print(f"Error in PSO predict with 1D input: {e}")
                # Fall back method - reshape and try again
                X_reshaped = X.reshape(-1)
                return self.function(X_reshaped)
        else:
            # For 2D inputs, we need to call the function for each row
            num_samples = X.shape[0]
            results = []
            
            for i in range(num_samples):
                # Extract a single sample
                x_i = X[i]
                # Evaluate the function
                try:
                    results.append(self.function(x_i))
                except Exception as e:
                    print(f"Error in PSO predict at sample {i}: {e}")
                    # Fallback - try with reshaped input
                    results.append(self.function(x_i.reshape(-1)))
            
            # Convert results to numpy array with correct shape
            results_array = np.array(results)
            
            # Make sure we return a 1D array for consistency
            if results_array.ndim > 1:
                results_array = results_array.reshape(-1)
                
            return results_array

    def _validate_config(self, config: Dict[str, Any]):
        """Validate the configuration parameters."""
        # Check for required keys
        if "bounds" not in config:
            raise ValueError("Missing required config key: bounds")
        
        # Validate bounds
        bounds = config["bounds"]
        if not isinstance(bounds, list):
            raise ValueError("Bounds must be a list of tuples (min, max)")
        
        # Validate particle count
        if "num_particles" in config and (not isinstance(config["num_particles"], int) or config["num_particles"] <= 0):
            raise ValueError("num_particles must be a positive integer")
        
        # Validate weights
        for weight_key in ["inertia_weight", "cognitive_weight", "social_weight"]:
            if weight_key in config and (not isinstance(config[weight_key], (int, float)) or config[weight_key] < 0):
                raise ValueError(f"{weight_key} must be a non-negative number") 