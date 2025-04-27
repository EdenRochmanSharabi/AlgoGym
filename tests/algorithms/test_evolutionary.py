import pytest
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

import sys
import os
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from algogym.algorithms.evolutionary import GeneticAlgorithm, SimpleNN
from algogym.algorithms.base import BaseAlgorithm
from algogym.functions.base import BaseFunction # Import needed for type hinting
from algogym.functions.examples import PolynomialFunction
from algogym.data import FunctionSampler # Correct import path

# --- Test SimpleNN --- (Basic tests as it's a helper class)

def test_simple_nn_init():
    nn = SimpleNN(input_size=2, hidden_size=5, output_size=1)
    assert nn.W1.shape == (2, 5)
    assert nn.b1.shape == (1, 5)
    assert nn.W2.shape == (5, 1)
    assert nn.b2.shape == (1, 1)

def test_simple_nn_predict():
    nn = SimpleNN(input_size=1, hidden_size=2, output_size=1)
    # Predict shape
    x = np.array([[0.5], [0.1]])
    preds = nn.predict(x)
    assert preds.shape == (2, 1)

def test_simple_nn_weights():
    nn = SimpleNN(input_size=1, hidden_size=2, output_size=1)
    initial_weights = [arr.copy() for arr in nn.get_weights()]
    flat_weights = nn.get_flat_weights()
    assert flat_weights.shape == (1*2 + 1*2 + 2*1 + 1*1,)
    
    # Modify flat weights and set back
    new_flat_weights = flat_weights + 1.0
    nn.set_flat_weights(new_flat_weights)
    
    # Check if internal weights changed
    modified_weights = nn.get_weights()
    for initial, modified in zip(initial_weights, modified_weights):
        assert not np.array_equal(initial, modified)
        np.testing.assert_allclose(modified, initial + 1.0)
        
    # Check setting weights directly
    nn.set_weights(initial_weights)
    np.testing.assert_allclose(nn.get_flat_weights(), flat_weights)

# --- Test GeneticAlgorithm ---

@pytest.fixture
def default_ga_config():
    return {
        "population_size": 10, # Small for testing
        "generations": 5,     # Small for testing
        "mutation_rate": 0.1,
        "mutation_strength": 0.1,
        "crossover_rate": 0.7,
        "tournament_size": 3,
        "hidden_layer_size": 5,
        "elitism_count": 1, # Add elitism for realistic count checks
        "verbose": False # Keep tests quiet
    }

@pytest.fixture
def dummy_data():
    X = np.random.rand(20, 1) * 10 - 5 # 20 samples, 1 feature
    y = X**2 # Simple quadratic target
    return X, y

def test_ga_init(default_ga_config):
    ga = GeneticAlgorithm(config=default_ga_config)
    assert isinstance(ga, BaseAlgorithm)
    assert ga.config["population_size"] == 10
    assert ga.config["generations"] == 5
    assert ga.population == []
    assert ga.best_individual is None
    assert ga.best_fitness == float('inf')

def test_ga_init_validation_errors():
    with pytest.raises(ValueError, match="Missing required config key: population_size"):
        GeneticAlgorithm(config={}) # Missing keys
    with pytest.raises(ValueError, match="population_size must be a positive integer"):
        GeneticAlgorithm(config={"population_size": 0, "generations": 1, "mutation_rate": 0.1, "hidden_layer_size": 5})
        
# Mocking internal methods for train test to avoid full run
@patch.object(GeneticAlgorithm, '_initialize_population', return_value=None)
@patch.object(GeneticAlgorithm, '_evaluate_population', return_value=None)
@patch.object(GeneticAlgorithm, '_tournament_selection')
@patch.object(GeneticAlgorithm, '_crossover')
@patch.object(GeneticAlgorithm, '_mutate', return_value=None)
def test_ga_train_execution_flow(
    mock_mutate, mock_crossover, mock_selection, mock_evaluate, mock_initialize, 
    default_ga_config, dummy_data):
    
    X, y = dummy_data
    ga = GeneticAlgorithm(config=default_ga_config)
    
    # Mock return values for selection and crossover
    mock_individual = SimpleNN(input_size=1, hidden_size=5, output_size=1)
    mock_selection.return_value = mock_individual
    # Ensure crossover returns two *distinct* mock individuals for population replacement test
    mock_crossover.return_value = (MagicMock(spec=SimpleNN), MagicMock(spec=SimpleNN))

    # Mock the population property after initialization
    mock_population_list = [MagicMock(spec=SimpleNN) for _ in range(default_ga_config["population_size"])]
    # IMPORTANT: Mock fitness attribute on each mock individual for evaluation/selection
    for ind in mock_population_list:
        type(ind).fitness = PropertyMock(return_value=np.random.rand())
        
    type(ga).population = PropertyMock(return_value=mock_population_list) 
    
    # Mock best fitness update to prevent issues if evaluate doesn't run fully
    type(ga).best_fitness = PropertyMock(return_value=0.1) # Set a reasonable fitness
    # Mock _input_dim and _output_dim as they are set during population init/eval
    type(ga)._input_dim = PropertyMock(return_value=1)
    type(ga)._output_dim = PropertyMock(return_value=1)
    # Mock best_individual for elitism to work
    mock_best = MagicMock(spec=SimpleNN)
    type(mock_best).fitness = PropertyMock(return_value=0.05)
    type(ga).best_individual = PropertyMock(return_value=mock_best)

    # Train should run without errors
    ga.train(X_data=X, y_data=y)

    # Assertions
    mock_initialize.assert_called_once()
    # Evaluate called once initially for whole population, then once per generation
    assert mock_evaluate.call_count == 1 + default_ga_config["generations"] 
    
    # Calculate number of new individuals per generation (pop_size - elitism_count)
    new_inds_per_gen = default_ga_config["population_size"] - default_ga_config["elitism_count"]
    # Calculate number of pairs needed to generate new individuals
    pairs_needed = (new_inds_per_gen + 1) // 2 # Integer division, ceiling
    
    # Selection called twice per pair needed 
    assert mock_selection.call_count == default_ga_config["generations"] * pairs_needed * 2
    # Crossover called once per pair
    assert mock_crossover.call_count == default_ga_config["generations"] * pairs_needed
    # Mutate called twice per pair (for each child created), but only if crossover happened
    # This test assumes crossover_rate = 1 for simplicity of counting mocks
    # A more precise test would need to inspect mock_crossover calls based on rate.
    assert mock_mutate.call_count >= default_ga_config["generations"] * pairs_needed # At least one child per pair, potentially 2
    assert mock_mutate.call_count <= default_ga_config["generations"] * pairs_needed * 2

    # Restore mocked property if needed elsewhere
    del type(ga).population
    del type(ga).best_fitness
    del type(ga)._input_dim
    del type(ga)._output_dim
    del type(ga).best_individual # Restore best_individual
    
# Patch FunctionSampler.load_data using string path to avoid import issues
@patch('algogym.data.FunctionSampler.load_data', return_value=(np.random.rand(30, 1), np.random.rand(30, 1))) 
def test_ga_train_with_target_function(mock_load_data, default_ga_config):
    """Test that train can run by sampling a target function."""
    func = PolynomialFunction(a=1, b=0, c=0, domain=(-2, 2)) # y = x^2
    config = {**default_ga_config, "train_samples": 30} # Specify samples needed
    ga = GeneticAlgorithm(config=config)
    
    # Run train - focus is on execution without error, not convergence
    # This will run the actual GA loop, but on a small scale
    ga.train(target_function=func)
    
    # Check FunctionSampler was instantiated correctly and load_data called
    # We expect FunctionSampler(func, 30) to be created internally
    # Can't easily assert internal instantiation, but assert load_data call
    mock_load_data.assert_called_once() 
    
    assert ga._input_dim == 1
    assert ga._output_dim == 1
    assert ga.best_individual is not None
    assert ga.best_fitness < float('inf') # Fitness should have improved

def test_ga_train_value_error(default_ga_config):
    ga = GeneticAlgorithm(config=default_ga_config)
    with pytest.raises(ValueError, match="requires either \(X_data, y_data\) or target_function"):
        ga.train() # No data or function
        
def test_ga_predict_before_train(default_ga_config):
     ga = GeneticAlgorithm(config=default_ga_config)
     with pytest.raises(RuntimeError, match="Algorithm has not been trained yet"):
         ga.predict(np.array([1.0]))
         
def test_ga_predict_after_train(default_ga_config, dummy_data):
    """Test predict runs and returns correct shape after training."""
    X, y = dummy_data
    # Reduce generations/pop for faster actual training run in this test
    config = {**default_ga_config, "generations": 2, "population_size": 5, "elitism_count": 0} 
    ga = GeneticAlgorithm(config=config)
    ga.train(X_data=X, y_data=y)
    
    # Predict single point
    x_single = np.array([0.5])
    y_pred_single = ga.predict(x_single)
    # Check if the result is a scalar or a 1-element array
    assert isinstance(y_pred_single, (np.number, np.ndarray))
    if isinstance(y_pred_single, np.ndarray):
        assert y_pred_single.shape == () or y_pred_single.shape == (1,) # Scalar or (1,) for 1D output
    
    # Predict batch
    x_batch = np.array([[0.1], [-0.2], [0.3]])
    y_pred_batch = ga.predict(x_batch)
    assert y_pred_batch.shape == (3, 1) # (N, output_dim)
    
    # Predict with wrong input dim
    x_wrong_dim = np.array([[0.1, 0.2]]) # Input dim = 2
    with pytest.raises(ValueError, match="Input batch dimension mismatch. Expected 1, got 2"):
        ga.predict(x_wrong_dim)

def test_ga_repr(default_ga_config):
    config = {**default_ga_config, "extra_param": "value"}
    ga = GeneticAlgorithm(config=config)
    repr_str = repr(ga)
    assert "GeneticAlgorithm" in repr_str
    assert "population_size=10" in repr_str
    assert "generations=5" in repr_str
    assert "extra_param='value'" in repr_str 