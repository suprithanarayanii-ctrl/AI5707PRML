"""
Synthetic 2-class dataset generator using Gaussian Mixture Models (GMMs).

For each team (batch) 1..N_TEAMS:
  - each class is a 16-D GMM with random number of components, random means, random covariances
  - data is sampled and shuffled (NO train/val/test split - students do that themselves)
  - the 16-D data is saved as-is; PCA (fit on the full data) is used only to make the 2-D plot
  - saved as:
        SyntheticData/team{n}/data.pkl        -> {"x": (N, 16), "y": (N,)}
        SyntheticData/team{n}/data_plot.png   -> PCA 2-D scatter
"""

import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ----------------------------- config ---------------------------------------
N_TEAMS = 22
N_CLASSES = 2
N_FEATURES = 16
SAMPLES_PER_CLASS = 10000
COMPONENTS_RANGE = (2, 4)  # each class gets 4-7 Gaussian components (high is exclusive)
MEAN_RANGE = (-20, 20)     # random means drawn uniformly from this box
COV_SCALE = 4              # controls how "big" the random covariances are
BASE_SEED = 42
OUT_DIR = "./SyntheticData"
# -----------------------------------------------------------------------------


def random_spd_cov(rng, d, scale):
    """Random symmetric positive-definite covariance matrix."""
    A = rng.normal(size=(d, d)) * scale
    return A @ A.T + 0.3 * np.eye(d)  # + eps*I guarantees positive definite


def make_random_gmm(rng):
    """Return a dict describing a random GMM."""
    k = rng.integers(*COMPONENTS_RANGE)
    weights = rng.dirichlet(np.ones(k) * 2.0)
    means = rng.uniform(*MEAN_RANGE, size=(k, N_FEATURES))
    covs = np.stack([random_spd_cov(rng, N_FEATURES, COV_SCALE) for _ in range(k)])
    return {"weights": weights, "means": means, "covs": covs}


def sample_gmm(rng, gmm, n):
    """Draw n samples from a GMM."""
    k = len(gmm["weights"])
    comps = rng.choice(k, size=n, p=gmm["weights"])
    X = np.empty((n, N_FEATURES))
    for j in range(k):
        idx = np.where(comps == j)[0]
        if len(idx):
            X[idx] = rng.multivariate_normal(gmm["means"][j], gmm["covs"][j], size=len(idx))
    return X


def generate_team_data(team_id):
    rng = np.random.default_rng(BASE_SEED + team_id)  # reproducible, unique per team

    Xs, ys = [], []
    for c in range(N_CLASSES):
        gmm = make_random_gmm(rng)
        Xs.append(sample_gmm(rng, gmm, SAMPLES_PER_CLASS))
        ys.append(np.full(SAMPLES_PER_CLASS, c))

    X = np.vstack(Xs)
    y = np.concatenate(ys)

    perm = rng.permutation(len(y))  # shuffle so classes aren't ordered
    return {"x": X[perm], "y": y[perm]}


def plot_team_data(data, team_id, path):
    colors = ["tab:blue", "tab:red"]
    X, y = data["x"], data["y"]

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(N_CLASSES):
        ax.scatter(Z[y == c, 0], Z[y == c, 1], s=4, alpha=0.35,
                   c=colors[c], label=f"Class {c}")
    ax.set_title(f"Team {team_id}: {N_FEATURES}-D GMM data, PCA to 2-D (n={len(y)})")
    ax.set_xlabel(f"PC1 ({evr[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({evr[1]:.1%} var)")
    ax.grid(alpha=0.3)
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    for n in range(1, N_TEAMS + 1):
        team_dir = os.path.join(OUT_DIR, f"team{n}")
        os.makedirs(team_dir, exist_ok=True)

        data = generate_team_data(n)

        with open(os.path.join(team_dir, "data.pkl"), "wb") as f:
            pickle.dump(data, f)

        plot_team_data(data, n, os.path.join(team_dir, "data_plot.png"))
        print(f"team{n}: saved (x={data['x'].shape}, y={data['y'].shape})")


if __name__ == "__main__":
    main()