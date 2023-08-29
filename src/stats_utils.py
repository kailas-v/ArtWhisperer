import numpy as np
import scipy.stats as stats

def calculate_wilson_interval(values, confidence_level=0.95):
    n = np.size(values)
    p = np.mean(values)

    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    denominator = 1 + z**2 / n

    center = (p + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p * (1 - p) / n + z**2 / (4 * n**2)) / denominator)

    lower_bound = center - margin
    upper_bound = center + margin

    return lower_bound, upper_bound

def bootstrap_mean_interval(values, confidence_level=0.95, n_bootstrap=100, seed=0):
    rng = np.random.RandomState(seed)
    bootstrapped_means = [
        np.mean(rng.choice(values, size=len(values), replace=True))
            for _ in range(n_bootstrap)
    ]
    return np.quantile(bootstrapped_means, [(1-confidence_level)/2, 1-(1-confidence_level)/2])