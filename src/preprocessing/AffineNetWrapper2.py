import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import pairwise_distances

class SimpleMLPTransform:
    def __init__(self, hidden_layer_sizes=(64,), max_iter=200):
        self.model = None
        self.dim1 = None
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.fitted = False

    def get_core_points(self, X, y, percent=0.9):
        core_indices = []
        for label in y.unique():
            X_label = X[y == label]
            center = X_label.mean().values.reshape(1, -1)
            dists = pairwise_distances(X_label, center).flatten()
            threshold = np.quantile(dists, percent)
            keep_idx = X_label[dists <= threshold].index
            core_indices.extend(keep_idx)
        return core_indices

    def fit(self, X_anchor: pd.DataFrame, y_anchor: pd.Series,
                  X_subject: pd.DataFrame, y_subject: pd.Series):
        self.dim1 = min(X_anchor.shape[1], X_subject.shape[1])
        X_anchor = X_anchor.iloc[:, :self.dim1]
        X_subject = X_subject.iloc[:, :self.dim1]

        # Keep only core points
        anchor_core_idx = self.get_core_points(X_anchor, y_anchor, percent=0.9)
        subject_core_idx = self.get_core_points(X_subject, y_subject, percent=0.9)
        X_anchor = X_anchor.loc[anchor_core_idx]
        y_anchor = y_anchor.loc[anchor_core_idx]
        X_subject = X_subject.loc[subject_core_idx]
        y_subject = y_subject.loc[subject_core_idx]

        p_list, q_list = [], []
        for label in y_anchor.unique():
            anchor_idx = y_anchor[y_anchor == label].index
            subject_idx = y_subject[y_subject == label].index
            n = min(len(anchor_idx), len(subject_idx))
            if n == 0:
                continue
            p_list.append(X_anchor.loc[anchor_idx].iloc[:n].to_numpy())
            q_list.append(X_subject.loc[subject_idx].iloc[:n].to_numpy())

        if len(p_list) == 0:
            raise ValueError("No matching labels found between anchor and subject.")

        p = np.vstack(p_list)
        q = np.vstack(q_list)

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            activation='relu',
            solver='adam',
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.1
        )

        self.model.fit(q, p)
        self.fitted = True
        return True

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        X_transformed = self.model.predict(X.iloc[:, :self.dim1])
        return pd.DataFrame(X_transformed, index=X.index, columns=[f'lda_{i+1}' for i in range(self.dim1)])

    def fit_transform(self, X_anchor, y_anchor, X_subject, y_subject):
        self.fit(X_anchor, y_anchor, X_subject, y_subject)
        return self.transform(X_subject)