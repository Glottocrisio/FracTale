import pandas as pd
import numpy as np
from scipy import stats
from numpy.random import default_rng
from hurst import compute_Hc

def fbm(n, H):
    """Generate fractional Brownian motion."""
    rng = default_rng()
    t = np.arange(n)
    dB = rng.normal(0, 1, n)
    
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            C[i, j] = C[j, i] = 0.5 * (abs(i-j+1)**(2*H) - 2*abs(i-j)**(2*H) + abs(i-j-1)**(2*H))
    
    L = np.linalg.cholesky(C)
    return np.dot(L, dB)

def calculate_hurst(time_series):
    if len(time_series) > 100:
        try:
            H, _, _ = compute_Hc(time_series)
            return H
        except:
            return np.nan
    else:
        return np.nan  # Return NaN for short series

def ks_test(series1, series2):
    return stats.ks_2samp(series1, series2).statistic

def analyze_tale(tale_data):
    metrics = ['num_words', 'num_clauses', 'dep_distance', 'Eventfulness', 'I']
    
    results = {}
    for metric in metrics:
        time_series = tale_data[metric].values
        results[f'{metric}_hurst'] = calculate_hurst(time_series)
    
    # Perform KS test between all pairs of metrics
    for i, metric1 in enumerate(metrics):
        for metric2 in metrics[i+1:]:
            ks_stat = ks_test(tale_data[metric1], tale_data[metric2])
            results[f'ks_{metric1}_{metric2}'] = ks_stat
    
    return results

def calculate_hurst_and_ks(series):
    """Calculate Hurst exponent and KS statistic."""
    H = calculate_hurst(series)
    fBm = fbm(len(series), H)
    _, p_value = stats.kstest(series, fBm)
    return H, p_value

# Load the data
df = pd.read_csv('grimm_sentences_metrics_en.csv', sep=';')

# Select numerical columns
numerical_columns = ['num_words', 'num_clauses', 'dep_distance', 'Eventfulness', 'I']

results = {}

for column in numerical_columns:
    series = df[column].values
    H, p_value = calculate_hurst_and_ks(series)
    results[column] = {'Hurst': H, 'KS_p_value': p_value}

for column, metrics in results.items():
    print(f"{column}:")
    print(f"  Hurst exponent: {metrics['Hurst']:.4f}")
    print(f"  KS test p-value: {metrics['KS_p_value']:.4f}")
    print()
