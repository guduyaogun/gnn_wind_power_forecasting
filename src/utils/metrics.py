import numpy as np

def mae(y, y_hat):
    """
    Mean Absolute Error:
    - Not differentiable
    - Less prone to outliers than MSE
    - Smooth interpretation, as scale aligns with variable scale
    - Gives no information about direction of the error (over- or underestimation)
    """
    return np.mean(abs(y - y_hat))


def mse(y, y_hat):
    """
    Mean Squared Error:
    - Differentiable, so good for optimization
    - Due to squaring, more prone to outliers
    - Interpretation has to be done with square factor in mind
    - Gives no information about direction of the error (over- or underestimation)
    """
    return np.mean((y - y_hat)**2)


def rmse(y, y_hat):
    """
    Root Mean Squared Error:
    - Differentiable, so good for optimization
    - Less prone to outliers than MSE
    - Smooth interpretation, as scale aligns with variable scale
    - Gives no information about direction of the error (over- or underestimation)
    """
    return np.sqrt(np.mean((y - y_hat)**2))


def nrmse(y, y_hat, normalizing_method = "range"):
    """
    Normalized Root Mean Squared Error:
    - Differentiable, so good for optimization
    - Less prone to outliers than MSE
    - Good to compare performance on different datasets with different scales
    - Gives no information about direction of the error (over- or underestimation)
    """
    if normalizing_method == "mean":
        nrmse_res = np.sqrt(np.mean((y - y_hat)**2)) / (np.mean(y))
    else:
        nrmse_res = np.sqrt(np.mean((y - y_hat)**2)) / (np.max(y) - np.min(y))

    return nrmse_res


def mape(y, y_hat, use_eps=False):
    """
    Mean Absolute Percentage Error:
    - Not differentiable.
    - Has an intuitive interpretation (relative error).
    - Cannot be used if there are (close-to) zero-values.
      One can adjust when adding a small eps in the denominator.
    - Puts a heavier penalty on negative errors.
    """
    if use_eps:
        eps = 1e-07
    else:
        eps = 0

    return np.mean(np.abs((y_hat - y) / (y + eps)))


def mspe(y, y_hat, use_eps=False):
    """
    Mean Squared Percentage Error:
    - Differentiable.
    - Interpretation has to be adjusted, because of the square.
    - Cannot be used if there are (close-to) zero-values.
      One can adjust when adding a small eps in the denominator.
    """
    if use_eps:
        eps = 1e-07
    else:
        eps = 0

    return np.mean(((y_hat - y) / (y + eps))**2)
