import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class PCALDATransform:
    def __init__(self, n_pca_components=50, n_lda_components=2):
        self.n_pca_components = n_pca_components
        self.n_lda_components = n_lda_components
        self.pipeline = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        max_pca = min(X.shape[0], X.shape[1])
        pca_components = min(self.n_pca_components, max_pca)

        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_components)),
            ('lda', LDA(n_components=self.n_lda_components))
        ])
        self.pipeline.fit(X, y)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        transformed = self.pipeline.transform(X)
        return pd.DataFrame(transformed, index=X.index, columns=[f'lda_{i+1}' for i in range(transformed.shape[1])])

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)
