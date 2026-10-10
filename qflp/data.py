"""Benchmark classification tasks for the trained-parameter experiments.

Ported from XanaduAI/qml-benchmarks (Apache-2.0), the generators behind Bowles, Ahmed and Schuld,
"Better than classical? The subtle art of benchmarking quantum machine learning models" (arXiv
2403.07059), with the paper's settings. The only change is an explicit random generator in place of
numpy's global state, so a dataset is fixed by its seed.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

N_SAMPLES = 300
TEST_SIZE = 0.2
# task name -> two-curves Fourier degree; None is the linearly separable task.
TASKS = {"linear": None, "curves2": 2, "curves10": 10}


def linearly_separable(n_samples: int, n_features: int, margin: float, rng: np.random.Generator):
    """Uniform points in [-1, 1]^d outside a margin around the hyperplane normal to (1, ..., 1)."""
    w = np.ones(n_features)
    x = 2 * rng.random((2 * n_samples, n_features)) - 1
    x = x[np.abs(x @ w) > margin][:n_samples]
    y = np.where(x @ w > 0, -1, 1)
    return x, y


def _fourier_series(t: float, coeffs: np.ndarray, degree: int, noise: float, rng):
    scaling = np.pi
    res = coeffs[0, 0] + coeffs[0, 1]
    for frequency in range(1, degree + 1):
        res += coeffs[frequency, 0] * np.cos(frequency * scaling * t)
        res += coeffs[frequency, 1] * np.sin(frequency * scaling * t)
    return res + rng.normal(scale=noise)


def two_curves(n_samples, n_features, degree, offset, noise, rng: np.random.Generator):
    """Two 1-d curves embedded in d dimensions by the same Fourier series, one shifted by
    `offset`, then standardized."""
    coeffs = rng.uniform(size=(n_features, degree + 1, 2))
    coeffs = coeffs / np.linalg.norm(coeffs)
    half = n_samples // 2
    curves = []
    for shift in (0.0, offset):
        pts = np.array(
            [
                [_fourier_series(t, coeffs[i], degree, noise, rng) for i in range(n_features)]
                for t in rng.random(half)
            ]
        )
        curves.append(pts + shift)
    x = np.r_[curves[0], curves[1]]
    y = np.array([-1] * half + [1] * half)
    return StandardScaler().fit_transform(x), y


def make_task(task: str, n_features: int, seed: int):
    """Train/test split of a task, features min-max scaled on the training split to [0, π], the
    attack's input domain."""
    rng = np.random.default_rng([seed, n_features, list(TASKS).index(task)])
    degree = TASKS[task]
    if degree is None:
        x, y = linearly_separable(N_SAMPLES, n_features, 0.02 * n_features, rng)
    else:
        x, y = two_curves(N_SAMPLES, n_features, degree, 1 / (2 * degree), 0.01, rng)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=int(rng.integers(2**31)), stratify=y
    )
    scaler = MinMaxScaler(feature_range=(0, np.pi)).fit(x_tr)
    return scaler.transform(x_tr), scaler.transform(x_te), y_tr.astype(float), y_te.astype(float)
