
import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance_ratio = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)

        # Calculate feature means
        self.mean = np.mean(X, axis=0)

        # Center the data
        X_centered = X - self.mean

        # Calculate covariance matrix
        covariance = (X_centered.T @ X_centered) / (len(X) - 1)

        # Find eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # Sort eigenvalues in descending order
        indices = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]

        # Select the top K eigenvectors
        self.components = eigenvectors[:, :self.n_components]

        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)

        self.explained_variance_ratio = (
            eigenvalues[:self.n_components] / total_variance
        )

        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)

        X_centered = X - self.mean

        return X_centered @ self.components

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
