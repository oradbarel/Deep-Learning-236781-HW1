import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.model_selection import ParameterGrid, cross_validate


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")
        y_pred = np.matmul(X, self.weights_)
        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)
        lamda = np.identity(X.shape[1], dtype=X.dtype)
        lamda[0][0] = 0
        w_opt = np.matmul(np.linalg.inv(((np.matmul(np.transpose(X),X) / X.shape[0]) + (self.reg_lambda * lamda))), (np.matmul(np.transpose(X), y) / X.shape[0]))
        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    features_df = df[feature_names] if feature_names else df.drop(target_name, axis=1)
    y_pred = model.fit_predict(features_df.to_numpy(), df[target_name].to_numpy())
    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)
        bias_term = np.ones((X.shape[0],1))
        xb = np.hstack((bias_term,X))
        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)
        polynomial = PolynomialFeatures(2)
        X_transformed = polynomial.fit_transform(np.delete(X, [1,3], 1))
        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """
    correlations =  {}
    for column in df.columns:
        if column != target_feature:
            correlation = df[column].corr(df[target_feature])
            correlations[column] = (abs(correlation), correlation)
    top_n_features = sorted(correlations, key=correlations.get, reverse=True)[:n]
    top_n_corr = [correlations[i][1] for i in top_n_features]
    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """
    mse = np.mean(np.square(np.subtract(y, y_pred)))
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """
    numerator = np.square(np.subtract(y,y_pred))
    denominator = np.square(np.subtract(y, np.mean(y)))
    r2 = 1 - (np.sum(numerator) / np.sum(denominator))
    return r2


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range
):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """
    parameter_grid = list(ParameterGrid({'bostonfeaturestransformer__degree': list(degree_range), 'linearregressor__reg_lambda': list(lambda_range)}))
    scores = []
    for parameters in parameter_grid:
        model.set_params(**parameters)
        current_score = cross_validate(model, X=X, y=y, cv=k_folds)
        scores.append(np.mean(current_score["test_score"]))
    best_params = parameter_grid[np.argmax(scores)]
    return best_params
