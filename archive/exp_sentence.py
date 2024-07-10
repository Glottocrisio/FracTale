
import numpy as np
import pandas as pd
from scipy import stats
from numpy.random import RandomState
from hurst import compute_Hc

def fbm(n, H, seed=None):
    """Generate fractional Brownian motion."""
    rng = RandomState(seed)
    t = np.arange(n)
    dt = t[1] - t[0]
    dB = rng.randn(n) * np.sqrt(dt)
    B = np.cumsum(dB)
    
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = 0.5 * (np.abs(i - j + 1)**(2*H) + np.abs(i - j - 1)**(2*H) - 2*np.abs(i - j)**(2*H))
    
    L = np.linalg.cholesky(C)
    fBm = np.dot(L, B)
    return fBm

def analyze_tale(tale_data):
    """Analyze self-similarity for a single tale."""
    series = ['num_words', 'num_clauses', 'I']
    results = {}
    
    # Calculate KS statistic between series
    ks_between_series = {}
    for i, series1 in enumerate(series):
        for j, series2 in enumerate(series[i+1:], start=i+1):
            data1 = pd.to_numeric(tale_data[series1], errors='coerce').dropna().values
            data2 = pd.to_numeric(tale_data[series2], errors='coerce').dropna().values
            
            if len(data1) > 1 and len(data2) > 1:
                ks_stat, _ = stats.kstest(stats.zscore(data1), stats.zscore(data2))
                ks_between_series[f"{series1}_vs_{series2}"] = ks_stat
            else:
                ks_between_series[f"{series1}_vs_{series2}"] = None
    
    results['ks_between_series'] = ks_between_series
    
    for col in series:
        data = pd.to_numeric(tale_data[col], errors='coerce').dropna().values
        n = len(data)
        
        if n < 2:
            results[col] = {
                'best_H': None,
                'min_ks_stat': None,
                'hurst_exponent': None
            }
            continue
        
        # Calculate Hurst exponent
        H, _, _ = compute_Hc(data)
        
        # Generate fBm with different Hurst parameters
        hurst_params = np.arange(0.1, 1.0, 0.1)
        ks_stats = []
        
        for H_fbm in hurst_params:
            fBm_series = fbm(n, H_fbm)
            ks_stat, _ = stats.kstest(stats.zscore(data), stats.zscore(fBm_series))
            ks_stats.append(ks_stat)
        
        # Find the Hurst parameter with the lowest KS statistic
        best_H = hurst_params[np.argmin(ks_stats)]
        min_ks_stat = np.min(ks_stats)
        
        results[col] = {
            'best_H': best_H,
            'min_ks_stat': min_ks_stat,
            'hurst_exponent': H
        }
    
    return results

# Read the CSV file
df = pd.read_csv('grimm_sentences_metrics_en.csv', delimiter=';', decimal=',')

# Convert numeric columns to float
numeric_cols = ['num_words', 'num_clauses', 'dep_distance', 'Eventfulness', 'I']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Group by tale_id
tales = df.groupby('tale_id')

# Analyze each tale
all_results = {}

for tale_id, tale_data in tales:
    all_results[tale_id] = analyze_tale(tale_data)

# Print results
for tale_id, results in all_results.items():
    print(f"Tale {tale_id}:")
    print("  KS statistics between series:")
    for pair, ks_stat in results['ks_between_series'].items():
        print(f"    {pair}: {ks_stat:.4f}" if ks_stat is not None else f"    {pair}: N/A")
    for series, stats in results.items():
        if series != 'ks_between_series':
            print(f"  {series}:")
            print(f"    Best Hurst parameter: {stats['best_H']:.2f}" if stats['best_H'] is not None else "    Best Hurst parameter: N/A")
            print(f"    Minimum KS statistic: {stats['min_ks_stat']:.4f}" if stats['min_ks_stat'] is not None else "    Minimum KS statistic: N/A")
            print(f"    Hurst exponent: {stats['hurst_exponent']:.4f}" if stats['hurst_exponent'] is not None else "    Hurst exponent: N/A")
    print()