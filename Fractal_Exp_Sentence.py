import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from numpy.random import RandomState
from hurst import compute_Hc
import matplotlib.pyplot as plt
import seaborn as sns

def fbm(n, H, seed=None):
    """Generate fractional Brownian motion."""
    rng = RandomState(seed)
    t = np.arange(n)
    dt = t[1] - t[0]
    dB = rng.randn(n) * np.sqrt(dt)
    B = np.cumsum(dB)
    
    # Compute the covariance matrix
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = 0.5 * (np.abs(i - j + 1)**(2*H) + np.abs(i - j - 1)**(2*H) - 2*np.abs(i - j)**(2*H))
    
    # Compute the Cholesky decomposition
    L = np.linalg.cholesky(C)
    
    # Generate fBm
    fBm = np.dot(L, B)
    return fBm

def estimate_hurst(data):
    """Estimate Hurst exponent for short time series."""
    n = len(data)
    if n < 2:
        return None
    
    # Calculate the array of the variances of the difference
    lags = range(2, n // 2)
    tau = [np.sqrt(np.std(np.subtract(data[lag:], data[:-lag]))) for lag in lags]
    
    # Use the last 20% of lags for the fit, or at least 5 lags
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
                ks_stat, _ = stats.kstest(stats.zscore(data1), stats.zscore(data2))
                ks_between_series[f"{series1}_vs_{series2}"] = ks_stat
            else:
                ks_between_series[f"{series1}_vs_{series2}"] = None
    
    results['ks_between_series'] = ks_between_series

    for col in series:
        # Convert data to numeric and remove any non-numeric values
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
            H = estimate_hurst(data)
        
        if n < 2:  # Skip if not enough data points
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
        
        # Generate fBm with different Hurst parameters
        hurst_params = np.arange(0.1, 1.0, 0.1)
        ks_stats = []
        
        for H in hurst_params:
            fBm_series = fbm(n, H)
            ks_stat, _ = stats.kstest(stats.zscore(data), stats.zscore(fBm_series))
            ks_stats.append(ks_stat)
        
        # Find the Hurst parameter with the lowest KS statistic
        best_H = hurst_params[np.argmin(ks_stats)]
        min_ks_stat = np.min(ks_stats)
        
        results[col] = {
            'best_H': best_H,
            'min_ks_stat': min_ks_stat,
            'hurst_exponent': H,
            'rs_hurst': rs_hurst,
            'rs_lags': rs_lags.tolist(),
            'rs_values': rs_values
        }
    
    return results


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
df_results.to_csv('tale_analysis_results.csv', index=False)



# Group by tale_id
tales = df.groupby('tale_id')

# Analyze each tale
all_results = {}

for tale_id, tale_data in tales:
    all_results[tale_id] = analyze_tale(tale_data)

# Assign beauty scores
beauty_ranks = {
    1: 10, 2: 9.84, 3: 9.68, 4: 9.52, 5: 9.36,
    6: 9.2, 7: 9.04, 8: 8.88, 9: 8.72, 10: 8.56,
    11: 8.4, 12: 8.24, 13: 8.08, 14: 7.92, 15: 7.76,
    16: 7.6, 17: 7.44, 18: 7.28, 19: 7.12, 20: 6.96,
    21: 6.8, 22: 6.64, 23: 6.48, 24: 6.32, 25: 6.16
}

# Create a DataFrame with all scores
scores_data = []
for tale_id, results in all_results.items():
    tale_scores = {
        'beauty_score': beauty_ranks[tale_id],
        'hurst_num_words': results['num_words']['hurst_exponent'],
        'hurst_num_clauses': results['num_clauses']['hurst_exponent'],
        'hurst_I': results['I']['hurst_exponent'],
        'rs_hurst_num_words': results['num_words']['rs_hurst'],
        'rs_hurst_num_clauses': results['num_clauses']['rs_hurst'],
        'rs_hurst_I': results['I']['rs_hurst']
    }
    scores_data.append(tale_scores)

scores_df = pd.DataFrame(scores_data)

# Calculate Spearman correlation
correlation_matrix = scores_df.rank().corr(method='spearman')

# Create a heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Spearman Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Print the correlation matrix
print("Spearman Correlation Matrix:")
print(correlation_matrix)

# Find the best performing self-similarity score
best_score_column = correlation_matrix['beauty_score'].abs().idxmax()
best_correlation = correlation_matrix.loc['beauty_score', best_score_column]

print(f"\nBest performing self-similarity score: {best_score_column}")
print(f"Correlation with beauty score: {best_correlation:.4f}")

# Scatter plot of beauty score vs best performing self-similarity score
plt.figure(figsize=(10, 6))
plt.scatter(scores_df[best_score_column], scores_df['beauty_score'])
plt.xlabel(best_score_column)
plt.ylabel('Beauty Score')
plt.title(f'Beauty Score vs {best_score_column}')
plt.tight_layout()
plt.savefig('beauty_vs_best_score.png')
plt.close()

# Visualization
plt.figure(figsize=(12, 8))
for tale_id, results in all_results.items():
    for series, stats in results.items():
        if series != 'ks_between_series' and stats['rs_lags'] is not None:
            plt.loglog(stats['rs_lags'], stats['rs_values'], label=f'Tale {tale_id} - {series}')

plt.xlabel('Lag')
plt.ylabel('R/S Statistic')
plt.title('Rescaled Range Analysis')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('rs_analysis_plot.png')
plt.close()

# Heatmap of Hurst exponents
hurst_data = {(tale_id, series): stats['hurst_exponent'] 
              for tale_id, results in all_results.items() 
              for series, stats in results.items() 
              if series != 'ks_between_series'}
df_hurst = pd.DataFrame(hurst_data, index=['Hurst Exponent']).T.unstack()

plt.figure(figsize=(12, 8))
sns.heatmap(df_hurst, annot=True, cmap='viridis')
plt.title('Hurst Exponents Across Tales and Series')
plt.tight_layout()
plt.savefig('hurst_exponent_heatmap.png')
plt.close()

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
