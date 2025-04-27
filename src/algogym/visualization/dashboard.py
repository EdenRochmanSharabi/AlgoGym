import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import ipywidgets as widgets
from IPython.display import display, clear_output

from algogym.functions import BaseFunction
from algogym.algorithms import BaseAlgorithm
from algogym.core.engine import ExperimentEngine
from .base import BaseVisualizer
from .function_viz import FunctionVisualizer


class InteractiveDashboard:
    """
    Interactive dashboard for AlgoGym experiments.
    
    Provides an interactive interface for visualizing and comparing function
    approximation algorithms, displaying metrics and experimenting with parameters.
    """
    
    def __init__(self, 
                config: Dict[str, Any] = None,
                visualizer: Optional[BaseVisualizer] = None):
        """
        Initialize the interactive dashboard.
        
        Args:
            config (Dict[str, Any], optional): Configuration parameters for the dashboard.
            visualizer (BaseVisualizer, optional): Visualizer instance to use for plotting.
                If None, a FunctionVisualizer with default settings is created.
        """
        # Dashboard configuration
        self.config = config if config is not None else {}
        self._validate_config()
        
        # Initialize visualizer
        self.visualizer = visualizer if visualizer is not None else FunctionVisualizer()
        
        # Experiment components
        self.functions: Dict[str, BaseFunction] = {}
        self.algorithms: Dict[str, BaseAlgorithm] = {}
        self.experiment_results: Dict[str, Dict[str, Any]] = {}
        
        # UI components
        self._init_ui_components()
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        default_config = {
            'output_height': '500px',
            'sidebar_width': '30%',
            'main_width': '70%',
            'default_title': 'AlgoGym Interactive Dashboard',
        }
        
        # Update default with provided config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def _init_ui_components(self) -> None:
        """Initialize UI components."""
        # Function selection
        self.function_dropdown = widgets.Dropdown(
            description='Function:',
            disabled=True,
            layout=widgets.Layout(width='90%')
        )
        
        # Algorithm selection
        self.algorithm_dropdown = widgets.Dropdown(
            description='Algorithm:',
            disabled=True,
            layout=widgets.Layout(width='90%')
        )
        
        # Action buttons
        self.run_button = widgets.Button(
            description='Run Experiment',
            disabled=True,
            button_style='primary',
            tooltip='Run the selected function with the selected algorithm',
            icon='play'
        )
        
        self.compare_button = widgets.Button(
            description='Compare',
            disabled=True,
            button_style='info',
            tooltip='Compare selected algorithms on the same function',
            icon='random'
        )
        
        # Output area
        self.output_area = widgets.Output(
            layout=widgets.Layout(
                height=self.config['output_height'],
                border='1px solid #ddd'
            )
        )
        
        # Tabs for different views
        self.tabs = widgets.Tab(
            children=[
                widgets.VBox([self.output_area]),  # Visualization tab
                widgets.Output(),  # Metrics tab
                widgets.Output(),  # Parameters tab
            ]
        )
        self.tabs.set_title(0, 'Visualization')
        self.tabs.set_title(1, 'Metrics')
        self.tabs.set_title(2, 'Parameters')
        
        # Set up event handlers
        self.function_dropdown.observe(self._on_function_change, names='value')
        self.algorithm_dropdown.observe(self._on_algorithm_change, names='value')
        self.run_button.on_click(self._on_run_click)
        self.compare_button.on_click(self._on_compare_click)
        
        # Main layout
        self.sidebar = widgets.VBox([
            widgets.HTML(f"<h3>{self.config['default_title']}</h3>"),
            widgets.HTML("<h4>Selection</h4>"),
            self.function_dropdown,
            self.algorithm_dropdown,
            widgets.HBox([self.run_button, self.compare_button]),
            widgets.HTML("<h4>Parameters</h4>"),
            # Parameter widgets will be added dynamically
        ], layout=widgets.Layout(width=self.config['sidebar_width']))
        
        self.main_area = widgets.VBox([
            self.tabs
        ], layout=widgets.Layout(width=self.config['main_width']))
        
        self.dashboard = widgets.HBox([self.sidebar, self.main_area])
    
    def register_function(self, name: str, function: BaseFunction) -> None:
        """
        Register a function for use in the dashboard.
        
        Args:
            name (str): Name to display for the function.
            function (BaseFunction): Function instance.
        """
        self.functions[name] = function
        
        # Update function dropdown
        self.function_dropdown.options = list(self.functions.keys())
        self.function_dropdown.disabled = len(self.functions) < 1
    
    def register_algorithm(self, name: str, algorithm: BaseAlgorithm) -> None:
        """
        Register an algorithm for use in the dashboard.
        
        Args:
            name (str): Name to display for the algorithm.
            algorithm (BaseAlgorithm): Algorithm instance.
        """
        self.algorithms[name] = algorithm
        
        # Update algorithm dropdown
        self.algorithm_dropdown.options = list(self.algorithms.keys())
        self.algorithm_dropdown.disabled = len(self.algorithms) < 1
        
        # Enable run button if both function and algorithm are available
        self._update_button_states()
    
    def _update_button_states(self) -> None:
        """Update the state of action buttons based on current selections."""
        if (self.function_dropdown.value is not None and 
            self.algorithm_dropdown.value is not None):
            self.run_button.disabled = False
        else:
            self.run_button.disabled = True
        
        # Enable compare button if at least two algorithms are registered
        self.compare_button.disabled = len(self.algorithms) < 2
    
    def _on_function_change(self, change) -> None:
        """Handle function selection change."""
        if change['new'] is None:
            return
        
        with self.tabs.children[0].children[0]:
            clear_output(wait=True)
            function = self.functions[change['new']]
            
            try:
                fig = self.visualizer.visualize_function(
                    function=function,
                    title=f"Function: {change['new']}",
                    show=False
                )
                plt.show()
            except Exception as e:
                print(f"Error visualizing function: {e}")
        
        self._update_button_states()
    
    def _on_algorithm_change(self, change) -> None:
        """Handle algorithm selection change."""
        self._update_button_states()
    
    def _on_run_click(self, b) -> None:
        """Handle run button click."""
        function_name = self.function_dropdown.value
        algorithm_name = self.algorithm_dropdown.value
        
        if function_name is None or algorithm_name is None:
            return
        
        function = self.functions[function_name]
        algorithm = self.algorithms[algorithm_name]
        
        with self.tabs.children[0].children[0]:
            clear_output(wait=True)
            print(f"Running experiment: {function_name} with {algorithm_name}...")
            
            # Create and run experiment
            engine = ExperimentEngine(
                target_function=function,
                algorithm=algorithm
            )
            
            try:
                results = engine.run()
                self.experiment_results[f"{function_name}_{algorithm_name}"] = results
                
                # Visualize results
                fig = self.visualizer.visualize_function(
                    function=function,
                    approximation=algorithm,
                    title=f"{function_name} with {algorithm_name}",
                    show=False
                )
                plt.show()
                
                # Show metrics in the Metrics tab
                with self.tabs.children[1]:
                    clear_output(wait=True)
                    if "evaluation_scores" in results and results["evaluation_scores"]:
                        plt.figure(figsize=(8, 4))
                        scores = results["evaluation_scores"]
                        plt.bar(list(scores.keys()), list(scores.values()))
                        plt.title(f"Evaluation Metrics: {function_name} with {algorithm_name}")
                        plt.ylabel("Value")
                        plt.xlabel("Metric")
                        plt.show()
                        
                        # Print additional experiment info
                        print(f"Training time: {results.get('training_time', 'N/A'):.4f} seconds")
                        print(f"Evaluation time: {results.get('evaluation_time', 'N/A'):.4f} seconds")
                        print(f"Total experiment time: {results.get('total_time', 'N/A'):.4f} seconds")
                    else:
                        print("No evaluation metrics available.")
                
            except Exception as e:
                print(f"Error running experiment: {e}")
                import traceback
                print(traceback.format_exc())
    
    def _on_compare_click(self, b) -> None:
        """Handle compare button click."""
        function_name = self.function_dropdown.value
        
        if function_name is None:
            return
        
        function = self.functions[function_name]
        
        with self.tabs.children[0].children[0]:
            clear_output(wait=True)
            print(f"Comparing algorithms on {function_name}...")
            
            algorithms_list = []
            algorithm_names = []
            
            # Run experiments for each algorithm
            for algo_name, algorithm in self.algorithms.items():
                print(f"Running {algo_name}...")
                
                # Create and run experiment
                engine = ExperimentEngine(
                    target_function=function,
                    algorithm=algorithm
                )
                
                try:
                    results = engine.run()
                    self.experiment_results[f"{function_name}_{algo_name}"] = results
                    algorithms_list.append(algorithm)
                    algorithm_names.append(algo_name)
                except Exception as e:
                    print(f"Error running {algo_name}: {e}")
            
            # Compare algorithms
            if len(algorithms_list) >= 2:
                try:
                    # Visualize comparison
                    fig = self.visualizer.visualize_comparison(
                        functions=[function] * len(algorithms_list),
                        approximations=algorithms_list,
                        labels=algorithm_names,
                        title=f"Algorithm Comparison on {function_name}",
                        show=False
                    )
                    plt.show()
                    
                    # Show comparison metrics in the Metrics tab
                    with self.tabs.children[1]:
                        clear_output(wait=True)
                        
                        # Collect metrics from all experiments
                        metrics = {}
                        for i, algo_name in enumerate(algorithm_names):
                            exp_key = f"{function_name}_{algo_name}"
                            if exp_key in self.experiment_results:
                                results = self.experiment_results[exp_key]
                                if "evaluation_scores" in results and results["evaluation_scores"]:
                                    for metric_name, value in results["evaluation_scores"].items():
                                        if metric_name not in metrics:
                                            metrics[metric_name] = []
                                        # Ensure all algorithms have an entry
                                        while len(metrics[metric_name]) < i:
                                            metrics[metric_name].append(None)
                                        metrics[metric_name].append(value)
                        
                        # Plot metrics comparison
                        if metrics:
                            fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
                            if len(metrics) == 1:
                                axes = [axes]
                                
                            for i, (metric_name, values) in enumerate(metrics.items()):
                                ax = axes[i]
                                ax.bar(algorithm_names[:len(values)], values)
                                ax.set_title(f"{metric_name}")
                                ax.set_xlabel("Algorithm")
                                ax.set_ylabel("Value")
                                ax.tick_params(axis='x', rotation=45)
                            
                            plt.tight_layout()
                            plt.show()
                        else:
                            print("No evaluation metrics available for comparison.")
                
                except Exception as e:
                    print(f"Error comparing algorithms: {e}")
                    import traceback
                    print(traceback.format_exc())
            else:
                print("Need at least two algorithms for comparison.")
    
    def display(self) -> None:
        """Display the dashboard."""
        display(self.dashboard)
        
        # Show initial instructions
        with self.output_area:
            clear_output(wait=True)
            print("Select a function and algorithm, then click 'Run Experiment'.")
            print("To compare multiple algorithms, select a function and click 'Compare'.") 