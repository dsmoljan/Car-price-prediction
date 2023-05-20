import numpy as np


class mean_value_regressor():
    def __init__(self):
        self.data_mean = None
        pass

    def fit(self, y_train: np.ndarray):
        self.data_mean = np.mean(y_train)

    def predict(self, y_test):
        if self.data_mean is None: raise RuntimeError("Error! Please fit your model first before making a prediction!")
        return np.full_like(y_test, self.data_mean)
