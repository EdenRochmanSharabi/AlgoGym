# AlgoGym: Visualizing Algorithm Learning

## Introduction

AlgoGym is a Python framework designed for experimenting with and visualizing various algorithms as they learn to approximate or optimize mathematical functions. It provides a structured environment based on `BaseFunction` and `BaseAlgorithm` classes, allowing for easy implementation and comparison of different approaches.

This project demonstrates the learning process of several common algorithms on benchmark functions like Sine, Polynomial, and Rosenbrock, generating animated GIFs and summary data for analysis.

## Features

*   **Object-Oriented Design:** Uses base classes for functions and algorithms for extensibility.
*   **Implemented Algorithms:**
    *   Genetic Algorithm (using a simple Neural Network per individual)
    *   Q-Learning (discretized state-action space)
    *   K-Nearest Neighbors (incremental learning)
    *   Particle Swarm Optimization
*   **Benchmark Functions:** Includes 1D (Sine, Polynomial) and 2D (Rosenbrock) examples.
*   **Visualization:** Generates animated GIFs showing the algorithm's approximation improving over epochs, alongside key performance metrics.
*   **Data Logging:** Saves training metrics (MSE, MAE, Best Fitness) and final prediction comparisons to text and CSV files in the `examples/results` directory for detailed analysis.

## Installation

1.  Clone the repository.
2.  (Recommended) Create a virtual environment: `python -m venv .venv` and activate it.
3.  Install required dependencies:
```bash
    pip install numpy matplotlib imageio Pillow
```

## Usage

To run the full visualization suite for all implemented algorithms and functions:

```bash
python examples/visualization_demo.py
```

You can also run visualizations for specific functions or algorithms using command-line arguments:

*   `--functions FUNC_NAME [FUNC_NAME ...]`: Run only for the specified function(s). Available: `Sine`, `Polynomial`, `Rosenbrock`.
*   `--algorithms ALGO_NAME [ALGO_NAME ...]`: Run only for the specified algorithm(s). Available: `GeneticAlgorithm`, `QLearning`, `KNN`, `PSO`.

**Examples:**

```bash
# Run only the Sine function with all algorithms
python examples/visualization_demo.py --functions Sine

# Run only GeneticAlgorithm and QLearning on all functions
python examples/visualization_demo.py --algorithms GeneticAlgorithm QLearning

# Run only Polynomial function with PSO
python examples/visualization_demo.py --functions Polynomial --algorithms PSO
```

Results, including GIFs and data files, will be saved in the `examples/results` directory, organized by function and then by algorithm.

## Algorithm Overview & Performance Analysis

This section details the algorithms implemented in AlgoGym and analyzes their behavior observed during testing within this framework.

---

### 1. Genetic Algorithm (GA)

*   **Description:** A population-based evolutionary algorithm inspired by natural selection. Each individual in the population represents a potential solution (in this case, a simple feed-forward neural network approximating the target function). Solutions evolve over generations through selection (tournament selection), crossover, and mutation.
*   **Implementation Details:**
    *   Individuals are small neural networks (1 hidden layer).
    *   Key parameters tuned: `population_size`, `hidden_layer_size`, `mutation_rate`, `mutation_strength`, `crossover_rate`, `tournament_size`.
    *   The `train_epoch` method performs one generation of evolution. Fitness is typically based on the inverse of the Mean Squared Error (MSE) of the individual's prediction against sample data.
*   **Performance Analysis:**
    *   **Successes:** Generally effective at approximating both 1D and 2D functions. Increasing the complexity (`hidden_layer_size` from 20 to 100 and `population_size` from 100 to 150) significantly improved its ability to capture the function's shape and achieve lower final MSE, especially noticeable on the Polynomial function.
    *   **Challenges:** Convergence can slow down significantly in later epochs as improvements become marginal. Finding the optimal set of hyperparameters requires experimentation. Can be computationally more intensive than other methods, especially with larger populations or networks.
*   **Pros:** Robust to local optima, effective for complex, non-differentiable search spaces, parallelizable.
*   **Cons:** Can be slow to converge, many hyperparameters to tune, requires careful fitness function design.

*   **Example Visualizations:**

    *   **Sine Function:**
        `[Insert GIF for GA on SineFunction here - e.g., examples/results/Sine/GeneticAlgorithm/gifs_and_images/sine_geneticalgorithm_learning.gif]`
        *Comment:* Observe how the red dashed line (approximation) gradually conforms to the blue line (true function) over 300 epochs. The MSE metric generally decreases, though potentially plateauing towards the end. The increased `hidden_layer_size` helps capture the curves better than initial runs.

    *   **Polynomial Function:**
        `[Insert GIF for GA on PolynomialFunction here - e.g., examples/results/Polynomial/GeneticAlgorithm/gifs_and_images/polynomial_geneticalgorithm_learning.gif]`
        *Comment:* Similar convergence pattern. The final fit is quite good, demonstrating the effectiveness of the increased network complexity for this function.

    *   **Rosenbrock Function (2D):**
        `[Insert GIF for GA on RosenbrockFunction here - e.g., examples/results/Rosenbrock/GeneticAlgorithm/gifs_and_images/rosenbrock_geneticalgorithm_learning.gif]`
        *Comment:* Visualization shows the contour plot of the approximation (red lines) evolving over the true function's filled contour plot. Convergence is much slower due to the higher dimensionality and complexity of the Rosenbrock function, but the MSE metric shows consistent improvement over 300 epochs.

---

### 2. Q-Learning Approximator

*   **Description:** A model-free reinforcement learning algorithm adapted for function approximation. It discretizes the input space (states) and the output space (actions). It learns a Q-table, `Q(state, action)`, which estimates the expected reward (related to prediction accuracy) of taking a specific output action in a given input state region.
*   **Implementation Details:**
    *   The input domain is divided into bins (`n_state_bins_per_dim`).
    *   The estimated output range is divided into discrete action values (`n_action_bins`).
    *   Uses an epsilon-greedy strategy for exploration vs. exploitation.
    *   Updates Q-values based on the Bellman equation (reward + discounted future value).
    *   Key parameters tuned: `alpha` (learning rate), `gamma` (discount factor), `epsilon_decay`, `n_state_bins_per_dim`, `n_action_bins`.
*   **Performance Analysis:**
    *   **Successes:** After fixing an initial bug related to accessing uninitialized attributes during prediction within `train_epoch`, the algorithm demonstrated learning behavior. It responds significantly to parameter changes.
    *   **Failures/Challenges:** Highly sensitive to hyperparameter tuning. An experiment with a coarse grid (`n_state_bins_per_dim=10`, `n_action_bins=20`) combined with a high learning rate (`alpha=0.3`) and slow epsilon decay resulted in the algorithm failing to learn effectively (MSE remained constant). Reverting to a finer grid (20/40 bins) and trying a very high learning rate (`alpha=0.5`) showed learning but with potential instability/oscillation in MSE. The discretization inherently limits the approximation accuracy – the output is piecewise constant across state bins. Not suitable for high-dimensional input spaces ("curse of dimensionality").
*   **Pros:** Can learn without a direct model of the function, conceptually simple RL approach.
*   **Cons:** Suffers from the curse of dimensionality, accuracy limited by discretization, performance highly dependent on parameter tuning (grid size, alpha, epsilon decay), convergence can be slow or unstable.

*   **Example Visualizations:**

    *   **Sine Function:**
        `[Insert GIF for QL on SineFunction here - e.g., examples/results/Sine/QLearning/gifs_and_images/sine_qlearning_learning.gif]`
        *Comment:* The approximation (red dashed line) will appear step-like due to the state and action discretization. With the `alpha=0.5` setting, observe how the MSE metric decreases but might show some variability, indicating the aggressive learning rate potentially causing oscillations in the Q-table values.

    *   **Polynomial Function:**
        `[Insert GIF for QL on PolynomialFunction here - e.g., examples/results/Polynomial/QLearning/gifs_and_images/polynomial_qlearning_learning.gif]`
        *Comment:* Similar step-like approximation. The learning progress (MSE reduction) is evident but may not be as smooth as the GA due to the interplay of discretization and the learning parameters.

---

### 3. K-Nearest Neighbors (KNN)

*   **Description:** A non-parametric, instance-based learning algorithm. To predict the output for a new input point, it finds the `k` nearest points in its stored training data and typically averages their corresponding output values.
*   **Implementation Details:**
    *   Stores training data (`X_data`, `y_data`) incrementally during `train_epoch`.
    *   Prediction involves finding `k` neighbors in `X_data` based on a distance metric (Euclidean) and averaging their `y_data`.
    *   Key parameter: `k` (number of neighbors).
*   **Performance Analysis:**
    *   **Successes:** Shows clear MSE reduction when applied to the 2D Rosenbrock function, indicating the core learning mechanism is functional.
    *   **Failures/Challenges:** **Persistent Visualization Error:** The script currently fails to generate visualizations for KNN on 1D functions (Sine, Polynomial) due to a shape mismatch error (`Input array for 1D function must be 1D (N,), got 2D.`). This bug prevents visual analysis of its 1D performance.
*   **Pros:** Simple to understand and implement, requires no explicit training phase (just data storage), adapts locally.
*   **Cons:** Prediction can be computationally expensive with large datasets, performance sensitive to the choice of `k` and distance metric, requires significant memory to store data, struggles with high-dimensional data, **currently has a visualization bug for 1D functions in this demo**.

*   **Example Visualizations:**

    *   **Sine/Polynomial Function:** *Currently fails due to error.*
    *   **Rosenbrock Function (2D):**
        `[Insert GIF for KNN on RosenbrockFunction here - e.g., examples/results/Rosenbrock/KNN/gifs_and_images/rosenbrock_knn_learning.gif]`
        *Comment:* The contour plot visualization is generated. While interpreting the approximation contours for KNN can be tricky, the metrics plot clearly shows the MSE decreasing over epochs as more data points are added, demonstrating learning.

---

### 4. Particle Swarm Optimization (PSO)

*   **Description:** A population-based stochastic optimization technique inspired by the social behavior of bird flocking or fish schooling. Particles "fly" through the search space, adjusting their velocity based on their own best-known position (`cognitive_weight`) and the swarm's best-known position (`social_weight`).
*   **Implementation Details:**
    *   Maintains a swarm of particles, each with a position and velocity.
    *   Updates positions and velocities based on inertia, personal best, and global best.
    *   Primarily used here to find the function's minimum value (best fitness).
    *   Key parameters: `num_particles`, `inertia_weight`, `cognitive_weight`, `social_weight`.
*   **Performance Analysis:**
    *   **Successes:** Extremely effective and rapid at *optimization* for the simple 1D functions (Sine, Polynomial) and the 2D Rosenbrock function. The "Best Fitness" metric converges to the optimal (or near-optimal) value almost immediately, often within the first few epochs.
    *   **Challenges/Clarifications:** **Not a direct approximator:** PSO, as implemented here, focuses on finding the *input* that yields the best *output*. It doesn't inherently build a model to predict outputs across the entire input range like GA or QL. **Plotting Artifact:** The visualization attempts to call `algorithm.predict()` on PSO. Since this method isn't suitably defined for PSO in this context, the error handling in the visualization script likely causes it to plot a flat line of zeros ( `y_pred = np.zeros_like(x_inputs)`). **Therefore, the red dashed line in the PSO GIFs does *not* represent a meaningful function approximation but is an artifact of the plotting code's fallback.** The key result to observe for PSO is the rapid convergence of the "Best Fitness" metric.
*   **Pros:** Efficient for optimization tasks, generally fewer parameters to tune than GA, simple concept.
*   **Cons:** Primarily an optimizer, not a direct function approximator in this setup, can converge prematurely to local optima. Visualization requires careful interpretation (focus on metrics, ignore the approximation line artifact).

*   **Example Visualizations:**

    *   **Sine Function:**
        `[Insert GIF for PSO on SineFunction here - e.g., examples/results/Sine/PSO/gifs_and_images/sine_pso_learning.gif]`
        *Comment:* Note the "Best Fitness" metric converges almost instantly to the minimum value (-1.0). **Ignore the red dashed line**, it is a plotting artifact (likely zeros) due to `predict()` incompatibility, not a function approximation.

    *   **Polynomial Function:**
        `[Insert GIF for PSO on PolynomialFunction here - e.g., examples/results/Polynomial/PSO/gifs_and_images/polynomial_pso_learning.gif]`
        *Comment:* Similar to Sine, "Best Fitness" quickly finds the minimum. **Ignore the red dashed approximation line.**

    *   **Rosenbrock Function (2D):**
        `[Insert GIF for PSO on RosenbrockFunction here - e.g., examples/results/Rosenbrock/PSO/gifs_and_images/rosenbrock_pso_learning.gif]`
        *Comment:* "Best Fitness" shows rapid convergence towards the minimum value (0.0). The 2D contour plot may not be meaningful for the same reasons as the 1D plots. Focus on the metric graph.

---

## Known Issues & Limitations

1.  **KNN 1D Visualization Error:** The `create_visualization` function currently throws a dimension mismatch error when attempting to visualize `KNearestNeighbors` on 1D functions (Sine, Polynomial). The KNN learning process for 2D (Rosenbrock) appears functional based on metrics.
2.  **PSO Visualization Artifact:** The `predict` method is not implemented for `ParticleSwarmOptimization` in a way compatible with the visualization script's expectation for function approximation plotting. The script's error handling defaults to plotting zeros for the approximation line, which is misleading. The relevant result for PSO is the "Best Fitness" metric.

## Future Work

*   Fix the KNN 1D visualization error.
*   Implement a more meaningful visualization for PSO, perhaps showing particle positions or deriving an approximation based on particle density/best positions.
*   Add more algorithms (e.g., Simulated Annealing, other Neural Network approaches, different RL algorithms).
*   Implement more complex benchmark functions.
*   Conduct systematic hyperparameter tuning studies for the implemented algorithms.
*   Improve error handling and reporting in the visualization script. 