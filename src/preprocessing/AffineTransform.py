import numpy as np
import pandas as pd
from scipy import linalg
import random
import matplotlib.pyplot as plt


class SimpleAffineTransform:
    def __init__(self):
        self.dim1 = None
        self.A = None
        self.b = None
        self.fitted = False

    def fit(self, X_anchor: pd.DataFrame, y_anchor: pd.Series,
                  X_subject: pd.DataFrame, y_subject: pd.Series):
        """
        Fit an affine transform from subject → anchor using all available data.
        """
        # Determine dimensions
        self.dim1 = min(X_anchor.shape[1], X_subject.shape[1])
        X_anchor = X_anchor.iloc[:, :self.dim1]
        X_subject = X_subject.iloc[:, :self.dim1]

        # Create matching pairs (take all available data)
        # Match by class label
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

        # Add ones column to q for affine transform
        ones = np.ones((q.shape[0], 1))
        Q_aug = np.hstack([q, ones])  # shape (n, dim+1)

        try:
            coeffs, _, _, _ = linalg.lstsq(Q_aug, p)  # Least Squares
            self.A = coeffs[:-1].T
            self.b = coeffs[-1]
            self.fitted = True
            return True
        except Exception as e:
            print(f"Affine fitting failed: {e}")
            self.fitted = False
            return False

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted affine transform to new data.
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        X = X.iloc[:, :self.dim1].to_numpy()
        transformed = X @ self.A.T + self.b
        return pd.DataFrame(transformed, columns=[f'lda_{i+1}' for i in range(self.dim1)])

    def fit_transform(self, X_anchor, y_anchor, X_subject, y_subject):
        self.fit(X_anchor, y_anchor, X_subject, y_subject)
        return self.transform(X_subject)

    def plot_affine_alignment(self, X_anchor: pd.DataFrame, y_anchor: pd.Series,
                                   X_subject: pd.DataFrame, y_subject: pd.Series,
                                   X_subject_test: pd.DataFrame, y_subject_test: pd.Series,
                                   title: str = 'Affine Alignment'):

        plt.figure(figsize=(9, 7))

        unique_labels = sorted(
            set(y_anchor.unique()).union(set(y_subject.unique())).union(set(y_subject_test.unique())))
        colors = plt.cm.tab10.colors
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

        for label in unique_labels:
            mask_anchor = (y_anchor.values == label)
            plt.scatter(X_anchor.iloc[mask_anchor, 0],
                        X_anchor.iloc[mask_anchor, 1],
                        marker='o', color=color_map[label], label=f'Anchor - Class {label}', alpha=0.8)

        for label in unique_labels:
            mask_subject = (y_subject.values == label)
            plt.scatter(X_subject.iloc[mask_subject, 0],
                        X_subject.iloc[mask_subject, 1],
                        marker='s', color=color_map[label], label=f'Subject - Class {label}', alpha=0.8)

        for label in unique_labels:
            mask_subject_test = (y_subject_test.values == label)
            plt.scatter(X_subject_test.iloc[mask_subject_test, 0],
                        X_subject_test.iloc[mask_subject_test, 1],
                        marker='*', color=color_map[label], label=f'Subject Test - Class {label}', alpha=0.8)

        plt.legend()
        plt.title(title)
        plt.xlabel("LDA 1")
        plt.ylabel("LDA 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_before_after(self,
                          X_anchor: pd.DataFrame, y_anchor: pd.Series,
                          X_subject_test_before: pd.DataFrame, y_subject_test_before: pd.Series,
                          X_subject_test_after: pd.DataFrame, y_subject_test_after: pd.Series,
                          X_subject_train_after: pd.DataFrame, y_subject_train_after: pd.Series,
                          title: str = 'Affine Alignment'):

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        unique_labels = sorted(
            set(y_anchor.unique())
            .union(set(y_subject_test_before.unique()))
            .union(set(y_subject_test_after.unique()))
            .union(set(y_subject_train_after.unique()))
        )
        colors = plt.cm.tab10.colors
        color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

        # --- Plot 1: Anchor vs Subject Before ---
        ax = axes[0]
        for label in unique_labels:
            mask_anchor = (y_anchor.values == label)
            mask_subject_before = (y_subject_test_before.values == label)

            ax.scatter(X_anchor.iloc[mask_anchor, 0],
                       X_anchor.iloc[mask_anchor, 1],
                       marker='o', facecolors='none', edgecolors=color_map[label], label=f'Anchor - Class {label}',
                       alpha=0.8)

            ax.scatter(X_subject_test_before.iloc[mask_subject_before, 0],
                       X_subject_test_before.iloc[mask_subject_before, 1],
                       marker='s', color=color_map[label], label=f'Before - Class {label}', alpha=0.8)

        ax.set_title('Anchor vs Subject Test Before')
        ax.set_xlabel("LDA 1")
        ax.set_ylabel("LDA 2")
        ax.grid(True)
        ax.legend()

        # --- Plot 2: Anchor vs Subject Test After ---
        ax = axes[1]
        for label in unique_labels:
            mask_anchor = (y_anchor.values == label)
            mask_subject_after = (y_subject_test_after.values == label)

            ax.scatter(X_anchor.iloc[mask_anchor, 0],
                       X_anchor.iloc[mask_anchor, 1],
                       marker='o', facecolors='none', edgecolors=color_map[label], label=f'Anchor - Class {label}',
                       alpha=0.8)

            ax.scatter(X_subject_test_after.iloc[mask_subject_after, 0],
                       X_subject_test_after.iloc[mask_subject_after, 1],
                       marker='*', color=color_map[label], label=f'Test After - Class {label}', alpha=0.8)

        ax.set_title('Anchor vs Subject Test After')
        ax.set_xlabel("LDA 1")
        ax.set_ylabel("LDA 2")
        ax.grid(True)
        ax.legend()

        # --- Plot 3: Anchor vs Subject Train After ---
        ax = axes[2]
        for label in unique_labels:
            mask_anchor = (y_anchor.values == label)
            mask_subject_train_after = (y_subject_train_after.values == label)

            ax.scatter(X_anchor.iloc[mask_anchor, 0],
                       X_anchor.iloc[mask_anchor, 1],
                       marker='o', facecolors='none', edgecolors=color_map[label], label=f'Anchor - Class {label}',
                       alpha=0.8)

            ax.scatter(X_subject_train_after.iloc[mask_subject_train_after, 0],
                       X_subject_train_after.iloc[mask_subject_train_after, 1],
                       marker='^', color=color_map[label], label=f'Train After - Class {label}', alpha=0.8)

        ax.set_title('Anchor vs Subject Train After')
        ax.set_xlabel("LDA 1")
        ax.set_ylabel("LDA 2")
        ax.grid(True)
        ax.legend()

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()




