import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from numpy.random import RandomState
from hurst import compute_Hc
import matplotlib.pyplot as plt
import seaborn as sns
import Functions as func

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

def estimate_hurst(data):
    """Estimate Hurst exponent for short time series."""
    n = len(data)
    if n < 2:
        return None
    
    lags = range(2, n // 2)
    tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
    
    num_lags = max(5, int(0.2 * len(lags)))
    m = np.polyfit(np.log(lags[-num_lags:]), np.log(tau[-num_lags:]), 1)
    hurst = m[0] / 2
    return hurst

def rescaled_range(data, lag):
    """Calculate rescaled range for given lag."""
    y = pd.Series(data)
    y_mean = y.rolling(lag).mean()
    y_std = y.rolling(lag).std()
    z = (y - y_mean) / y_std
    r = z.rolling(lag).max() - z.rolling(lag).min()
    s = y.rolling(lag).std()
    rs = r / s
    return rs.dropna().mean()

def rs_analysis(data):
    """Perform Rescaled Range Analysis."""
    lags = np.logspace(0, np.log10(len(data)//2), 20).astype(int)
    rs_values = [rescaled_range(data, lag) for lag in lags]
    return lags, rs_values

def analyze_tale(tale_data):
    """Analyze self-similarity for a single tale."""
    series = ['num_words', 'num_clauses', 'dep_distance','Eventfulness','I']
    results = {}
    
    ks_between_series = {}
    for i, series1 in enumerate(series):
        for j, series2 in enumerate(series[i+1:], start=i+1):
            data1 = pd.to_numeric(tale_data[series1], errors='coerce').dropna().values
            data2 = pd.to_numeric(tale_data[series2], errors='coerce').dropna().values
            
            if len(data1) > 1 and len(data2) > 1:
                try:
                    ks_stat, _ = stats.kstest(stats.zscore(data1), stats.zscore(data2))
                    ks_between_series[f"{series1}_vs_{series2}"] = ks_stat
                except Exception as e:
                    ks_between_series[f"{series1}_vs_{series2}"] = None
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
                'hurst_exponent': None,
                'rs_hurst': None,
                'rs_lags': None,
                'rs_values': None
            }
            continue
        if n >= 100:
            H, _, _ = compute_Hc(data)
        else:
            try:
                H = func.hurst_exponent(data)
            except Exception as e:
                H = estimate_hurst(data)
        
        if n < 2:  
            results[col] = {
                'best_H': None,
                'min_ks_stat': None,
                'hurst_exponent': H,
                'rs_hurst': rs_hurst,
                'rs_lags': rs_lags.tolist(),
                'rs_values': rs_values
            }
            continue
        
        rs_lags, rs_values = rs_analysis(data)
        rs_hurst, _ = np.polyfit(np.log(rs_lags), np.log(rs_values), 1)
        
        hurst_params = np.arange(0.1, 1.0, 0.1)
        ks_stats = []
        
        for H in hurst_params:
            fBm_series = fbm(n, H)
            try:
                ks_stat, _ = stats.kstest(stats.zscore(data), stats.zscore(fBm_series))
            except Exception as e:
                ks_stat = None
            ks_stats.append(ks_stat)
        
        try:
            best_H = hurst_params[np.argmin(ks_stats)]
            min_ks_stat = np.min(ks_stats)
        except Exception as e:
            best_H = None
            min_ks_stat = None
        
        results[col] = {
            'best_H': best_H,
            'min_ks_stat': min_ks_stat,
            'hurst_exponent': H,
            'rs_hurst': rs_hurst,
            'rs_lags': rs_lags.tolist(),
            'rs_values': rs_values
        }
    
    return results

languages = ['en', 'de', 'es', 'it']

for lang in languages:
    df = pd.read_csv(f'grimm_sentences_metrics_{lang}.csv', delimiter=';', decimal=',')

    # Convert numeric columns to float
    numeric_cols = ['num_words', 'num_clauses', 'dep_distance', 'Eventfulness', 'I']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    tales = df.groupby('tale_id')


    # Analyze each tale
    all_results = {}

    for tale_id, tale_data in tales:
        all_results[tale_id] = analyze_tale(tale_data)
    
    # Export results to CSV
    csv_data = []
    for tale_id, results in all_results.items():
        for series, stats in results.items():
            if series != 'ks_between_series':
                csv_data.append({
                    'tale_id': tale_id,
                    'series': series,
                    'best_H': stats['best_H'],
                    'min_ks_stat': stats['min_ks_stat'],
                    'hurst_exponent': stats['hurst_exponent'],
                    'rs_hurst': stats['rs_hurst']
                })
        for pair, ks_stat in results['ks_between_series'].items():
            csv_data.append({
                'tale_id': tale_id,
                'series': pair,
                'ks_stat': ks_stat
            })

    df_results = pd.DataFrame(csv_data)
    df_results.to_csv(f'tale_analysis_results_{lang}.csv', index=False)

#The direct computation of the Hurst exponent will be performed on the dimension related series 
#per every language, not per every tale.

