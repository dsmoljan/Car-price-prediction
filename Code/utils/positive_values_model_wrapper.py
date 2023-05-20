# from sklearn.base import BaseEstimator

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class positive_values_model_wrapper(BaseEstimator, RegressorMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        predictions = self.estimator.predict(X)
        rounded_predictions = np.maximum(predictions, 0)  # Round negative values to zero
        predictions[predictions < 0] = 0
        return predictions






#
#
# class positive_values_model_wrapper(BaseEstimator):
#     """
#     A very simple model wrapper, which ensures that, when using model.predict() all the negative values
#     get rounded to 0. Fitting of the model is simply passed to the base model, along any kwagrs.
#     """
#     def __init__(self, model_, *args, **kwargs):
#         self.model_ = model_
#
#         # kwargs depend on the model used, so assign them whatever they are
#         for key, value in kwargs.items():
#             setattr(self, key, value)
#
#     def predict(self, y_test):
#         y_pred = self.model_.predict(y_test)
#         y_pred[y_pred < 0] = 0
#         return y_pred
#
#     # method taken from https://ploomber.io/blog/sklearn-custom/
#     # some models implement custom methods. Anything that is not implemented here
#     # will be delegated to the underlying model. There is one condition we have
#     # to cover: if the underlying estimator has class attributes they won't
#     # be accessible until we fit the model (since we instantiate the model there)
#     # to fix it, we try to look it up attributes in the instance, if there
#     # is no instance, we look up the class. More info here:
#     # https://scikit-learn.org/stable/developers/develop.html#estimator-types
#     def __getattr__(self, key):
#         if key != 'model_':
#             if hasattr(self, 'model_'):
#                 return getattr(self.model_, key)
#             else:
#                 return getattr(self.est_class, key)
#         else:
#             raise AttributeError(
#                 "'{}' object has no attribute 'model_'".format(type(self).__name__))