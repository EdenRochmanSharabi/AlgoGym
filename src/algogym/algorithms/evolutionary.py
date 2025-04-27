import numpy as np
import random
from typing import Any, Dict, List, Tuple

from .base import BaseAlgorithm
from algogym.functions import BaseFunction # Use absolute import for type hinting if needed elsewhere
from algogym.data import FunctionSampler # Use absolute import for sampling

# --- Simple Neural Network for Individuals ---
# (Could be moved to a separate 'models' module later)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500))) # Clip to avoid overflow

def relu(x):
    return np.maximum(0, x)

class SimpleNN:
    """A very basic feedforward neural network."""
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases to zero for a zero initial prediction
        self.W1 = np.zeros((input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.zeros((hidden_size, output_size))
        self.b2 = np.zeros(output_size)
        self.activation = relu # Or sigmoid

    def predict(self, X):
        # Make sure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Forward pass
        z1 = X.dot(self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU activation
        z2 = a1.dot(self.W2) + self.b2
        return z2

    def get_weights(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def set_weights(self, weights):
        self.W1, self.b1, self.W2, self.b2 = weights
        
    def get_flat_weights(self):
        return np.concatenate([w.flatten() for w in self.get_weights()])
        
    def set_flat_weights(self, flat_weights):
        shapes = [(w.shape) for w in self.get_weights()]
        sizes = [np.prod(s) for s in shapes]
        indices = np.cumsum(sizes)
        
        new_weights = []
        start = 0
        for i, shape in enumerate(shapes):
            end = indices[i]
            new_weights.append(flat_weights[start:end].reshape(shape))
            start = end
            
        self.set_weights(new_weights)

# --- Genetic Algorithm Implementation ---

class GeneticAlgorithm(BaseAlgorithm):
    """
    A simple Genetic Algorithm for function approximation using NNs as individuals.
    
    Approximates a function given data points (X, y).
    """
    DEFAULT_CONFIG = {
        "population_size": 50,
        "generations": 100,
        "mutation_rate": 0.1,
        "mutation_strength": 0.1,
        "crossover_rate": 0.7,
        "tournament_size": 5,
        "hidden_layer_size": 10, # NN specific
        "verbose": True
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        config_to_validate = config if config is not None else {}
        # Validate config before merging with defaults
        required_keys = ["population_size", "generations", "mutation_rate", "hidden_layer_size"]
        for key in required_keys:
            if key not in config_to_validate:
                raise ValueError(f"Missing required config key: {key}")
        
        merged_config = {**self.DEFAULT_CONFIG, **(config if config is not None else {})}
        super().__init__(merged_config)
        self.population: List[SimpleNN] = []
        self.fitness: List[float] = []
        self.best_individual: SimpleNN | None = None
        self.best_fitness: float = float('inf')
        self._input_dim: int | None = None
        self._output_dim: int | None = None

    def _validate_config(self, config: Dict[str, Any]):
        required_keys = ["population_size", "generations", "mutation_rate", "hidden_layer_size"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
        # Add more type/value checks as needed
        if not isinstance(config["population_size"], int) or config["population_size"] <= 0:
            raise ValueError("population_size must be a positive integer.")
        if not isinstance(config["generations"], int) or config["generations"] <= 0:
            raise ValueError("generations must be a positive integer.")
        if not isinstance(config["mutation_rate"], (int, float)) or config["mutation_rate"] < 0 or config["mutation_rate"] > 1:
            raise ValueError("mutation_rate must be a number between 0 and 1.")
        if not isinstance(config["hidden_layer_size"], int) or config["hidden_layer_size"] <= 0:
            raise ValueError("hidden_layer_size must be a positive integer.")

    def _initialize_population(self):
        if self._input_dim is None or self._output_dim is None:
            raise RuntimeError("Input/Output dimensions not set before population initialization.")
        self.population = [
            SimpleNN(self._input_dim, self.config["hidden_layer_size"], self._output_dim)
            for _ in range(self.config["population_size"])
        ]
        self.best_individual = None
        self.best_fitness = float('inf')

    def _calculate_fitness(self, individual: SimpleNN, X_data: np.ndarray, y_data: np.ndarray) -> float:
        predictions = individual.predict(X_data)
        # Use Mean Squared Error as fitness (lower is better)
        mse = np.mean((predictions - y_data)**2)
        # Add small epsilon to avoid zero fitness if perfect match
        return mse + 1e-9 

    def _evaluate_population(self, X_data: np.ndarray, y_data: np.ndarray):
        self.fitness = [
            self._calculate_fitness(ind, X_data, y_data) for ind in self.population
        ]
        best_current_idx = np.argmin(self.fitness)
        if self.fitness[best_current_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_current_idx]
            # Deep copy might be needed if SimpleNN state is complex
            self.best_individual = SimpleNN(self._input_dim, self.config["hidden_layer_size"], self._output_dim)
            self.best_individual.set_weights(self.population[best_current_idx].get_weights())
            
    def _tournament_selection(self) -> SimpleNN:
        tournament_size = self.config["tournament_size"]
        indices = random.sample(range(self.config["population_size"]), tournament_size)
        tournament_fitness = [self.fitness[i] for i in indices]
        winner_index_in_tournament = np.argmin(tournament_fitness)
        winner_original_index = indices[winner_index_in_tournament]
        return self.population[winner_original_index]

    def _crossover(self, parent1: SimpleNN, parent2: SimpleNN) -> Tuple[SimpleNN, SimpleNN]:
        child1 = SimpleNN(self._input_dim, self.config["hidden_layer_size"], self._output_dim)
        child2 = SimpleNN(self._input_dim, self.config["hidden_layer_size"], self._output_dim)
        child1.set_weights(parent1.get_weights()) # Start with copies
        child2.set_weights(parent2.get_weights())

        if random.random() < self.config["crossover_rate"]:
            # Single point crossover on flattened weights
            p1_flat = parent1.get_flat_weights()
            p2_flat = parent2.get_flat_weights()
            crossover_point = random.randint(1, len(p1_flat) - 1)
            
            c1_flat = np.concatenate((p1_flat[:crossover_point], p2_flat[crossover_point:]))
            c2_flat = np.concatenate((p2_flat[:crossover_point], p1_flat[crossover_point:]))
            
            child1.set_flat_weights(c1_flat)
            child2.set_flat_weights(c2_flat)
            
        return child1, child2

    def _mutate(self, individual: SimpleNN):
         weights = individual.get_weights()
         new_weights = []
         for w in weights:
             if random.random() < self.config["mutation_rate"]:
                 mutation = np.random.randn(*w.shape) * self.config["mutation_strength"]
                 new_weights.append(w + mutation)
             else:
                 new_weights.append(w)
         individual.set_weights(new_weights)

    def train(self, target_function: BaseFunction | None = None, X_data: np.ndarray | None = None, y_data: np.ndarray | None = None):
        if X_data is None or y_data is None:
            # This basic GA version requires explicit data
            if target_function is not None:
                # Option: Sample the function if data not given (requires n_samples in config)
                n_samples = self.config.get("train_samples", 100) # Example default
                print(f"Warning: No training data (X, y) provided. Sampling {n_samples} points from target_function.")
                sampler = FunctionSampler(target_function, n_samples)
                X_data, y_data = sampler.load_data()
            else:
                raise ValueError("GeneticAlgorithm requires either (X_data, y_data) or target_function for training.")

        if X_data.ndim == 1:
             X_data = X_data[:, np.newaxis]
        if y_data.ndim == 1:
             y_data = y_data[:, np.newaxis]
             
        self._input_dim = X_data.shape[1]
        self._output_dim = y_data.shape[1]
        
        # Store training data for epoch-by-epoch training
        self._X_data = X_data
        self._y_data = y_data
        
        self._initialize_population()
        self._evaluate_population(X_data, y_data)

        if self.config.get("verbose", False):
            print(f"Generation 0: Best Fitness = {self.best_fitness:.6f}")

        for generation in range(1, self.config["generations"] + 1):
            # Use the train_epoch method for each generation
            metrics = self.train_epoch(generation - 1)  # Zero-based epoch index
            
            if self.config.get("verbose", False) and generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {metrics['best_fitness']:.6f}")
                
        # Set the internal approximated function to the best NN found
        self._approximated_function = self.best_individual
        if self.config.get("verbose", False):
             print(f"Training finished. Final Best Fitness = {self.best_fitness:.6f}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Performs a single generation (epoch) of genetic algorithm training.
        
        Args:
            epoch (int): The current epoch/generation number.
            
        Returns:
            Dict[str, float]: Dictionary containing metrics for this generation.
        """
        if self._X_data is None or self._y_data is None:
            raise RuntimeError("Train method must be called before train_epoch")
        
        # Store previous best fitness to measure improvement
        previous_best_fitness = self.best_fitness if self.best_individual is not None else float('inf')
        
        # Create new population
        new_population = []
        
        # Optional: Elitism - keep the best individual
        if self.config.get("elitism", True) and self.best_individual is not None:
            # Create copy of best individual
            elite = SimpleNN(self._input_dim, self.config["hidden_layer_size"], self._output_dim)
            elite.set_weights(self.best_individual.get_weights())
            new_population.append(elite)
        
        # Fill the rest of the population through selection, crossover, and mutation
        while len(new_population) < self.config["population_size"]:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            child1, child2 = self._crossover(parent1, parent2)
            
            self._mutate(child1)
            self._mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.config["population_size"]:
                new_population.append(child2)
        
        # Update population
        self.population = new_population
        
        # Evaluate new population
        self._evaluate_population(self._X_data, self._y_data)
        
        # Calculate metrics
        fitness_values = np.array(self.fitness)
        improvement = previous_best_fitness - self.best_fitness if self.best_individual is not None else 0.0

        # Calculate MSE and MAE using the best individual found in this epoch
        mse = float('nan')
        mae = float('nan')
        if self.best_individual is not None:
            y_pred = self.best_individual.predict(self._X_data)
            mse = np.mean((y_pred - self._y_data)**2)
            mae = np.mean(np.abs(y_pred - self._y_data))
            # Note: self.best_fitness is already the MSE of the best individual
        
        # Return metrics
        return {
            "best_fitness": float(self.best_fitness), # This is essentially the MSE of the best individual
            "mse": float(mse),
            "mae": float(mae),
            "mean_fitness": float(np.mean(fitness_values)),
            "min_fitness": float(np.min(fitness_values)),
            "max_fitness": float(np.max(fitness_values)),
            "fitness_std": float(np.std(fitness_values)),
            "improvement": float(improvement),
            "generations_completed": epoch + 1,
            "epoch": epoch
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.best_individual is None:
            raise RuntimeError("Algorithm has not been trained yet.")
            
        if x.ndim == 1:
            if self._input_dim == 1:
                 x_proc = x.reshape(-1, 1) # Handle 1D input array (N,) -> (N, 1)
            else:
                 if x.shape[0] != self._input_dim:
                      raise ValueError(f"Input point dimension mismatch. Expected {self._input_dim}, got {x.shape[0]}")
                 x_proc = x[np.newaxis, :] # Single point (D,) -> (1, D)
        elif x.ndim == 2:
            if x.shape[1] != self._input_dim:
                 raise ValueError(f"Input batch dimension mismatch. Expected {self._input_dim}, got {x.shape[1]}")
            x_proc = x # Batch (N, D)
        else:
            raise ValueError("Input must be 1D or 2D array.")

        predictions = self.best_individual.predict(x_proc)
        
        # Return original shape if single input was given
        if x.ndim == 1 and predictions.shape[0] == 1:
            return predictions.flatten() # Return (output_dim,) or scalar if output_dim=1
        else:
            return predictions # Return (N, output_dim) 