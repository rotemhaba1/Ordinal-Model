import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances

class PLSCorePoints(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, percent=0.9):
        self.n_components = n_components
        self.percent = percent
        self.pls = PLSRegression(n_components=n_components)
        self.selected_indices_ = None

    def _select_core_points(self, X, y):
        core_indices = []
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y

        for label in np.unique(y_series):
            X_label = X[y_series == label]
            center = X_label.mean(axis=0).values.reshape(1, -1)
            dists = pairwise_distances(X_label, center).flatten()
            threshold = np.quantile(dists, self.percent)
            keep_idx = X_label[dists <= threshold].index
            core_indices.extend(keep_idx)

        return sorted(core_indices)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns
        else:
            self.columns_ = [f'col_{i}' for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=self.columns_)

        core_idx = self._select_core_points(X, y)
        self.selected_indices_ = core_idx
        self.pls.fit(X.loc[core_idx], y.loc[core_idx])
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(self.pls.transform(X), index=X.index,
                                columns=[f'pls_{i+1}' for i in range(self.n_components)])
        else:
            return self.pls.transform(X)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)