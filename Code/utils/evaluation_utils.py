import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, make_scorer


def get_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "msle": mean_squared_log_error(y_true, y_pred),
        "rmsle": mean_squared_log_error(y_true, y_pred, squared=False),
        "r2": r2_score(y_true, y_pred)
    }


def get_cross_validate_scorer():
    return {
        'MSE': make_scorer(mean_squared_error),
        'MSLE': make_scorer(mean_squared_log_error),
        'RMSE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))),
        'RMSLE': make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_log_error(y_true, y_pred))),
        'R2-score': make_scorer(r2_score)
    }
