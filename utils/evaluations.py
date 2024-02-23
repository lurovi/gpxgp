import math
import numpy as np


def compute_linear_scaling(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    slope: float = np.cov(y, p)[0, 1] / float(np.var(p) + 1e-12)
    intercept: float = np.mean(y) - slope * np.mean(p)
    return slope, intercept


def linear_scale_predictions(p: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    slope: float = np.core.umath.clip(slope, -1e+20, 1e+20)
    intercept: float = np.core.umath.clip(intercept, -1e+20, 1e+20)
    p: np.ndarray = intercept + np.core.umath.clip(slope * p, -1e+20, 1e+20)
    p = np.core.umath.clip(p, -1e+20, 1e+20)
    return p


def optionally_linear_scale_predictions(y: np.ndarray, p: np.ndarray, linear_scaling: bool = False, slope: float = None, intercept: float = None) -> np.ndarray:
    if linear_scaling:
        slope, intercept = compute_linear_scaling(y, p)
        p: np.ndarray = linear_scale_predictions(p, slope=slope, intercept=intercept)
    else:
        if slope is not None and intercept is not None:
            p: np.ndarray = linear_scale_predictions(p, slope=slope, intercept=intercept)
    return p


def mean_squared_error(y: np.ndarray, p: np.ndarray, linear_scaling: bool = False, slope: float = None, intercept: float = None) -> float:
    p: np.ndarray = optionally_linear_scale_predictions(y=y, p=p, linear_scaling=linear_scaling, slope=slope, intercept=intercept)
    diff: np.ndarray = np.core.umath.clip(p - y, -1e+20, 1e+20)
    diff = np.core.umath.clip(np.square(diff), -1e+20, 1e+20)
    s: float = diff.sum()
    if s > 1e+20:
        s = 1e+20
    s = s / float(len(y))
    if s > 1e+20:
        s = 1e+20
    return s


def root_mean_squared_error(y: np.ndarray, p: np.ndarray, linear_scaling: bool = False, slope: float = None, intercept: float = None) -> float:
    s: float = mean_squared_error(y=y, p=p, linear_scaling=linear_scaling, slope=slope, intercept=intercept)
    s = math.sqrt(s)
    if s > 1e+20:
        s = 1e+20
    return s

