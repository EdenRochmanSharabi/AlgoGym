import numpy as np

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Squared Error (MSE).

    Args:
        y_true (np.ndarray): Ground truth target values. Shape (n_samples, output_dim).
        y_pred (np.ndarray): Predicted values. Shape (n_samples, output_dim).

    Returns:
        float: The calculated MSE.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return float(np.mean((y_true - y_pred)**2))

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Mean Absolute Error (MAE).

    Args:
        y_true (np.ndarray): Ground truth target values. Shape (n_samples, output_dim).
        y_pred (np.ndarray): Predicted values. Shape (n_samples, output_dim).

    Returns:
        float: The calculated MAE.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    return float(np.mean(np.abs(y_true - y_pred)))

# Add other metrics as needed (e.g., R-squared, etc.) 