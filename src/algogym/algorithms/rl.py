import numpy as np
import random
import time
from typing import Any, Dict, Tuple, List

from .base import BaseAlgorithm
from algogym.functions import BaseFunction


class QLearningApproximator(BaseAlgorithm):
    """
    Uses Q-Learning to approximate a function by discretizing state and action space.

    The input space (function domain) is discretized into states.
    The output range is discretized into actions.
    The agent learns a Q-table Q(state, action) to predict the best output (action)
    for a given input region (state).
    
    Note: This is more suitable for low-dimensional functions due to the 
          curse of dimensionality with the Q-table.
    """
    DEFAULT_CONFIG = {
        "episodes": 10000,
        "alpha": 0.1,         # Learning rate
        "gamma": 0.9,         # Discount factor
        "epsilon_start": 1.0,   # Exploration rate start
        "epsilon_end": 0.01,  # Exploration rate end
        "epsilon_decay": 0.995, # Exploration rate decay factor
        "n_state_bins_per_dim": 10, # Bins for discretizing each input dimension
        "n_action_bins": 10,     # Bins for discretizing the output (action) space
        "reward_scale": 1.0,   # Scaling factor for reward calculation
        "verbose": True
    }

    def __init__(self, config: Dict[str, Any] | None = None):
        config_to_validate = config if config is not None else {}
        # Validate required keys before merging with defaults
        required_keys = ["episodes", "alpha", "gamma", "epsilon_start", "n_state_bins_per_dim", "n_action_bins"]
        for key in required_keys:
            if key not in config_to_validate:
                raise ValueError(f"Missing required Q-learning config key: {key}")
        
        merged_config = {**self.DEFAULT_CONFIG, **(config if config is not None else {})}
        super().__init__(merged_config)
        self.q_table: np.ndarray | None = None
        self.state_bins: List[np.ndarray] = []
        self.action_values: np.ndarray | None = None
        self.epsilon = self.config["epsilon_start"]
        self._input_dim: int | None = None
        self._output_dim: int | None = None
        self._domain_min: np.ndarray | None = None
        self._domain_max: np.ndarray | None = None

    def _validate_config(self, config: Dict[str, Any]):
        # Validate data types and ranges for Q-learning specific config
        if not isinstance(config["n_state_bins_per_dim"], int) or config["n_state_bins_per_dim"] <= 0:
            raise ValueError("n_state_bins_per_dim must be a positive integer")
        if not isinstance(config["n_action_bins"], int) or config["n_action_bins"] <= 0:
            raise ValueError("n_action_bins must be a positive integer")
            
        # Warn if some parameters seem unusual
        if config["alpha"] < 0 or config["alpha"] > 1:
            print(f"Warning: alpha={config['alpha']} should typically be in [0, 1]")
        if config["gamma"] < 0 or config["gamma"] > 1:
            print(f"Warning: gamma={config['gamma']} should typically be in [0, 1]")

    def _discretize_state(self, x: np.ndarray) -> Tuple[int, ...]:
        """Converts a continuous input vector x into a discrete state tuple."""
        if self._input_dim is None or not self.state_bins:
             raise RuntimeError("State discretization bins not initialized.")
        if x.shape != (self._input_dim,):
             raise ValueError(f"Input x has wrong shape for state discretization. Expected ({self._input_dim},), got {x.shape}")
             
        state_indices = []
        n_bins = self.config["n_state_bins_per_dim"]
        for i in range(self._input_dim):
            # np.digitize returns the index of the bin (1-based), subtract 1 for 0-based index
            idx = np.digitize(x[i], self.state_bins[i]) - 1
            # Clip index to be within valid range [0, n_bins - 1]
            idx = np.clip(idx, 0, n_bins - 1)
            state_indices.append(idx)
            
        return tuple(state_indices)

    def _initialize_discretization(self, function: BaseFunction):
        """Sets up the state and action discretization based on the function."""
        self._input_dim = function.input_dim
        self._output_dim = function.output_dim
        self._domain_min, self._domain_max = function.domain
        
        if self._output_dim > 1:
            print("Warning: QLearningApproximator currently only supports 1D output. Using first dimension.")
            # TODO: Handle multi-output functions (e.g., separate Q-tables or vector actions)
            self._output_dim = 1 

        # State discretization
        n_bins = self.config["n_state_bins_per_dim"]
        self.state_bins = []
        for i in range(self._input_dim):
            # Create n_bins + 1 edges for n_bins intervals
            # Use linspace excluding the endpoints for bin edges to match np.digitize behavior
            # The bins themselves will cover the domain [min, max]
            bins = np.linspace(self._domain_min[i], self._domain_max[i], n_bins + 1)[1:-1]
            self.state_bins.append(bins)
            
        state_shape = tuple([n_bins] * self._input_dim)
        n_actions = self.config["n_action_bins"]
        q_table_shape = state_shape + (n_actions,)
        self.q_table = np.zeros(q_table_shape)
        print(f"Initialized Q-table with shape: {q_table_shape}")

        # Action discretization (Estimate range by sampling function)
        print("Estimating action space by sampling function...")
        n_samples = 200 # Number of samples to estimate output range
        sample_x = function.sample_domain(n_samples)
        sample_y = function(sample_x)
        if self._output_dim == 1 and sample_y.ndim == 1:
             sample_y = sample_y[:, np.newaxis]
             
        min_y = np.min(sample_y[:, 0])
        max_y = np.max(sample_y[:, 0])
        # Add a small margin to the estimated range
        margin = (max_y - min_y) * 0.05 + 1e-6
        self.action_values = np.linspace(min_y - margin, max_y + margin, n_actions)
        print(f"Discretized action space ({n_actions} actions) between {self.action_values[0]:.4f} and {self.action_values[-1]:.4f}")

    def _get_action_value(self, action_index: int) -> float:
        if self.action_values is None:
             raise RuntimeError("Action values not initialized.")
        return self.action_values[action_index]
        
    def _choose_action(self, state: Tuple[int, ...]) -> int:
        """Chooses an action using epsilon-greedy strategy."""
        if self.q_table is None:
             raise RuntimeError("Q-table not initialized.")
             
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.config["n_action_bins"] - 1) # Explore
        else:
            return np.argmax(self.q_table[state]) # Exploit
            
    def _calculate_reward(self, target_value: float, action_value: float) -> float:
         """Calculates reward based on proximity to target value."""
         # Reward is higher when the action value is closer to the target value
         # Use negative squared error, scaled.
         error = (target_value - action_value)**2
         # Normalize reward to be higher for smaller errors
         # This reward structure might need tuning based on the function scale
         reward = self.config["reward_scale"] * np.exp(-error) 
         return reward

    def train(self, target_function: BaseFunction | None = None, X_data: np.ndarray | None = None, y_data: np.ndarray | None = None):
        if target_function is None:
            raise ValueError("QLearningApproximator requires a target_function for training (to get rewards)." )

        print(f"Starting Q-Learning training for {self.config['episodes']} episodes...")
        self._initialize_discretization(target_function)
        if self.q_table is None: # Check added for type safety
            raise RuntimeError("Q-table initialization failed.")

        # Store the target function for later use
        self._target_function = target_function
        
        alpha = self.config["alpha"]
        gamma = self.config["gamma"]
        start_time = time.time()

        for episode in range(self.config["episodes"]):
            # Call train_epoch to perform a single episode of training
            metrics = self.train_epoch(episode)

            if self.config.get("verbose", False) and (episode + 1) % (self.config["episodes"] // 10) == 0:
                elapsed = time.time() - start_time
                print(f"Episode {episode + 1}/{self.config['episodes']} | Epsilon: {self.epsilon:.4f} | Reward: {metrics['reward']:.4f} | Time: {elapsed:.2f}s")
        
        # Store learned Q-table and discretization info
        self._approximated_function = {
            "q_table": self.q_table,
            "state_bins": self.state_bins,
            "action_values": self.action_values,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim # Store intended output dim (even if only 1 used)
        }
        print("Q-Learning training finished.")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Performs a single episode of Q-Learning.
        
        Args:
            epoch (int): The current epoch/episode number.
            
        Returns:
            Dict[str, float]: Dictionary containing metrics for this episode.
        """
        if self._target_function is None or self.q_table is None:
            raise RuntimeError("Train method must be called before train_epoch")
        
        # Get configuration parameters
        alpha = self.config["alpha"]
        gamma = self.config["gamma"]
        
        # Sample a starting point (environment state)
        x_current = self._target_function.sample_domain(1)
        # Ensure x_current is always 1D array for discretization
        if self._target_function.input_dim == 1 and isinstance(x_current, np.ndarray) and x_current.ndim == 0:
             x_current = np.array([x_current.item()]) 
        elif isinstance(x_current, np.ndarray) and x_current.ndim == 2:
             x_current = x_current.flatten()
             
        state = self._discretize_state(x_current)
        action = self._choose_action(state)
        action_value = self._get_action_value(action)
        
        # Get reward by comparing action value to true function value
        true_value = self._target_function(x_current)
        # Handle potential scalar output from function
        if isinstance(true_value, np.ndarray) and true_value.size == 1:
            true_value = true_value.item()
        elif isinstance(true_value, np.ndarray):
             true_value = true_value[0] # Use first dimension if multi-output
             
        reward = self._calculate_reward(float(true_value), action_value)
        error = abs(float(true_value) - action_value)
        
        # Get next state (sample another point)
        # In function approximation, the 'next state' isn't strictly determined by the action.
        # We sample another point to update the Q-value based on potential future rewards.
        x_next = self._target_function.sample_domain(1)
        if self._target_function.input_dim == 1 and isinstance(x_next, np.ndarray) and x_next.ndim == 0:
             x_next = np.array([x_next.item()])
        elif isinstance(x_next, np.ndarray) and x_next.ndim == 2:
             x_next = x_next.flatten()
             
        next_state = self._discretize_state(x_next)
        
        # Q-Learning update
        old_value = self.q_table[state + (action,)]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + alpha * (reward + gamma * next_max - old_value)
        self.q_table[state + (action,)] = new_value
        
        # Calculate TD error (Temporal Difference)
        td_error = reward + gamma * next_max - old_value
        
        # Decay epsilon
        old_epsilon = self.epsilon
        self.epsilon = max(self.config["epsilon_end"], self.epsilon * self.config["epsilon_decay"])
        
        # Calculate q-value statistics
        max_q = np.max(self.q_table)
        min_q = np.min(self.q_table)
        mean_q = np.mean(self.q_table)
        
        # --- Calculate actual MSE and MAE over a grid for this epoch --- 
        eval_mse = float('nan')
        eval_mae = float('nan')
        n_eval_points = 100 # Number of points to evaluate MSE/MAE over
        if self._input_dim == 1:
            eval_x = np.linspace(self._domain_min[0], self._domain_max[0], n_eval_points)
            eval_x_pred_input = eval_x.reshape(-1, 1)
        else:
            # For >1D, sample randomly within domain for evaluation
            # Using the same points every epoch might be slightly better for comparison,
            # but random sampling is simpler here.
            eval_x_pred_input = np.random.uniform(low=self._domain_min, high=self._domain_max, size=(n_eval_points, self._input_dim))
            eval_x = eval_x_pred_input # Use the sampled points directly for function eval if needed
            
        try:
            # Pass correct shape to function (__call__ might expect 1D for 1D input)
            if self._input_dim == 1:
                 y_true_eval = self._target_function(eval_x)
            else:
                 y_true_eval = self._target_function(eval_x_pred_input)
                 
            # Ensure y_true_eval is 1D or (N, 1) before comparison
            if isinstance(y_true_eval, np.ndarray) and y_true_eval.ndim > 1 and y_true_eval.shape[1] == 1:
                y_true_eval = y_true_eval.flatten()
            elif np.isscalar(y_true_eval):
                y_true_eval = np.full(n_eval_points, y_true_eval) # Broadcast if scalar

            # Get predictions using the current Q-table policy
            y_pred_eval = self.predict(eval_x_pred_input) 
            if isinstance(y_pred_eval, np.ndarray) and y_pred_eval.ndim > 1 and y_pred_eval.shape[1] == 1:
                y_pred_eval = y_pred_eval.flatten()
            elif np.isscalar(y_pred_eval):
                 y_pred_eval = np.full(n_eval_points, y_pred_eval) # Broadcast if scalar
                 
            # Ensure shapes match for calculation
            if y_true_eval.shape == y_pred_eval.shape:
                eval_mse = np.mean((y_true_eval - y_pred_eval) ** 2)
                eval_mae = np.mean(np.abs(y_true_eval - y_pred_eval))
            else:
                print(f"Warning: Shape mismatch during QL epoch eval: y_true {y_true_eval.shape}, y_pred {y_pred_eval.shape}")

        except Exception as e:
            print(f"Warning: Error during QL epoch evaluation: {e}")
        # --------------------------------------------------------------

        # Return metrics
        return {
            "reward": float(reward),        # Reward for the single step taken
            "error": float(error),          # Absolute error for the single step taken
            "td_error": float(td_error),      # TD error for the single step taken
            "epsilon": float(self.epsilon),
            "epsilon_decay": float(old_epsilon - self.epsilon),
            "max_q": float(max_q),
            "min_q": float(min_q),
            "mean_q": float(mean_q),
            "q_update": float(new_value - old_value),
            "mse": float(eval_mse), # Use the evaluated MSE
            "mae": float(eval_mae), # Use the evaluated MAE
            "epoch": epoch
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.q_table is None or not self.state_bins or self.action_values is None or self._input_dim is None or self._output_dim is None:
            raise RuntimeError("The Q-learning algorithm has not been trained or initialized properly. Call train() first.")

        # Ensure input x is 2D (n_predict_samples, input_dim)
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
            
        n_predict_samples = x_proc.shape[0]
        # Use float for predictions, match expected output dim (even if only 1D used internally)
        # Use self._output_dim directly, as self._approximated_function might not be set during train_epoch evaluation
        predictions = np.zeros((n_predict_samples, self._output_dim))

        for i in range(n_predict_samples):
            state = self._discretize_state(x_proc[i])
            best_action_index = np.argmax(self.q_table[state])
            predictions[i, 0] = self._get_action_value(best_action_index) # Predict 1st dim
            # If multi-output needed later, predict for other dims here
            # Ensure remaining columns are zero if self._output_dim > 1
            if self._output_dim > 1:
                 predictions[i, 1:] = 0 # Explicitly zero out other dimensions
            
        # Return original shape if single input was given
        if x.ndim == 1 and predictions.shape[0] == 1:
             # Handle case where original output dim > 1 but we only predict 1
             # Use self._output_dim directly
             if self._output_dim == 1:
                  return predictions.flatten()[0] # Return scalar
             else: 
                  return predictions.flatten() # Return (output_dim,)
        else:
             # Handle case where original output dim > 1 but we only predict 1
             # Use self._output_dim directly
             if self._output_dim == 1:
                  return predictions[:, 0] # Return (N,) array
             else:
                  return predictions # Return (N, output_dim)

import time # Add time import 