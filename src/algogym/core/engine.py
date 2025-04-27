import time
from typing import Dict, Any, Tuple, Callable
import numpy as np

from algogym.functions import BaseFunction
from algogym.algorithms import BaseAlgorithm
from algogym.data import BaseDataLoader, FunctionSampler
from algogym.evaluation import mean_squared_error # Default metric

class ExperimentEngine:
    """
    Orchestrates the process of running an approximation experiment.

    Takes a function (or data), an algorithm, runs the training/approximation,
    and evaluates the result using specified metrics.
    """
    def __init__(self, 
                 target_function: BaseFunction | None = None,
                 data_loader: BaseDataLoader | None = None,
                 algorithm: BaseAlgorithm | None = None,
                 metrics: Dict[str, Callable[[np.ndarray, np.ndarray], float]] | None = None,
                 train_samples: int = 1000, # Used if target_function given but no data_loader
                 test_samples: int = 500): # Used for evaluation if function available
                 
        if target_function is None and data_loader is None:
            raise ValueError("Either target_function or data_loader must be provided.")
        if algorithm is None:
            raise ValueError("An algorithm instance must be provided.")
            
        self.target_function = target_function
        self.data_loader = data_loader
        self.algorithm = algorithm
        self.metrics = metrics if metrics is not None else {"mse": mean_squared_error}
        self.train_samples = train_samples
        self.test_samples = test_samples
        
        self.results: Dict[str, Any] = {}

    def _get_data(self) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Gets training and testing data."""
        X_train, y_train, X_test, y_test = None, None, None, None
        
        if self.target_function:
            # If function is given, we prioritize sampling from it for train/test
            print(f"Sampling training data ({self.train_samples} points) from {self.target_function!r}...")
            train_sampler = FunctionSampler(self.target_function, self.train_samples)
            X_train, y_train = train_sampler.load_data()
            
            print(f"Sampling test data ({self.test_samples} points) from {self.target_function!r}...")
            test_sampler = FunctionSampler(self.target_function, self.test_samples)
            X_test, y_test = test_sampler.load_data()
        elif self.data_loader:
            print(f"Loading data using {self.data_loader!r}...")
            X_train, y_train = self.data_loader.load_data()
            # If we only have data, we can't easily generate separate test data
            # unless the loader supports splitting, or we sample from the training data.
            # For now, we'll test on the training data if no function is available.
            print("Warning: No target function provided. Evaluating on training data.")
            X_test, y_test = X_train, y_train
            
        if X_train is None:
             raise RuntimeError("Failed to obtain training data.")
             
        return X_train, y_train, X_test, y_test

    def run(self) -> Dict[str, Any]:
        """Runs the full experiment: data loading, training, evaluation."""
        start_time = time.time()
        self.results = {"start_time": start_time}
        
        try:
            # 1. Get Data
            X_train, y_train, X_test, y_test = self._get_data()
            self.results["data_load_time"] = time.time() - start_time
            self.results["train_data_shape"] = X_train.shape
            self.results["test_data_shape"] = X_test.shape if X_test is not None else None
            
            # 2. Train Algorithm
            print(f"Training algorithm {self.algorithm!r}...")
            train_start_time = time.time()
            self.algorithm.train(target_function=self.target_function, X_data=X_train, y_data=y_train)
            self.results["train_time"] = time.time() - train_start_time
            print(f"Training finished in {self.results['train_time']:.2f} seconds.")

            # 3. Evaluate Algorithm
            if X_test is not None and y_test is not None:
                print("Evaluating algorithm...")
                eval_start_time = time.time()
                try:
                    y_pred = self.algorithm.predict(X_test)
                    
                    # Ensure shapes match for metrics
                    if y_pred.shape != y_test.shape:
                        print(f"Warning: Shape mismatch during evaluation. y_test: {y_test.shape}, y_pred: {y_pred.shape}. Attempting to reshape y_pred.")
                        try:
                            y_pred = y_pred.reshape(y_test.shape)
                        except ValueError as e:
                            print(f"Error reshaping y_pred: {e}. Skipping metric calculation.")
                            self.results["evaluation_error"] = str(e)
                            y_pred = None # Prevent metric calculation
                            
                    eval_scores = {}
                    if y_pred is not None:
                        for name, metric_func in self.metrics.items():
                            try:
                                eval_scores[name] = metric_func(y_test, y_pred)
                            except Exception as e:
                                print(f"Error calculating metric '{name}': {e}")
                                eval_scores[name] = None
                                
                    self.results["evaluation_scores"] = eval_scores
                except Exception as e:
                    print(f"Experiment failed: {e}")
                    self.results["error"] = str(e)
                    self.results["evaluation_scores"] = None
                finally:
                    self.results["evaluation_time"] = time.time() - eval_start_time
                    if "evaluation_scores" in self.results and self.results["evaluation_scores"]:
                        print(f"Evaluation scores: {self.results['evaluation_scores']}")
            else:
                print("Skipping evaluation as no test data is available.")
                self.results["evaluation_scores"] = None

        except Exception as e:
            print(f"Experiment failed: {e}")
            self.results["error"] = str(e)
            import traceback
            self.results["traceback"] = traceback.format_exc()

        finally:
            end_time = time.time()
            self.results["end_time"] = end_time
            self.results["total_time"] = end_time - start_time
            print(f"Experiment finished in {self.results['total_time']:.2f} seconds.")
            
        return self.results

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target={self.target_function or self.data_loader!r}, algorithm={self.algorithm!r})" 