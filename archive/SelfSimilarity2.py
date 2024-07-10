import pandas as pd
import numpy as np
from hurst import compute_Hc
from scipy import stats
import matplotlib.pyplot as plt

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

# def process_csv(file_path):
#     df = pd.read_csv(file_path)
    
#     # Separate tale metrics and sentence metrics
#     #tale_metrics = df[df['sentence'].isna()]
#     #sentence_metrics = df[df['sentence'].notna()]
    
#     sentence_results = []
    
#     for tale_id in sentence_metrics['tale_id'].unique():
#         tale_data = sentence_metrics[sentence_metrics['tale_id'] == tale_id]
#         results = analyze_tale(tale_data)
#         results['tale_id'] = tale_id
#         tale_results.append(results)
    
#     return pd.DataFrame(tale_results)

def plot_hurst_distribution(results, metric):
    plt.figure(figsize=(10, 6))
    plt.hist(results[f'{metric}_hurst'], bins=20)
    plt.title(f'Distribution of Hurst Exponent for {metric}')
    plt.xlabel('Hurst Exponent')
    plt.ylabel('Frequency')
    plt.savefig(f'hurst_{metric}_distribution.png')
    plt.close()

# Main execution
csv_file = 'grimm_sentences_metrics_en.csv'  # Replace with your CSV file path
#results = process_csv(csv_file)
# Load the data
df = pd.read_csv('grimm_sentences_metrics_en.csv', sep=';')

# Select numerical columns
numerical_columns = ['num_words', 'num_clauses', 'dep_distance', 'Eventfulness', 'I']

results = {}

for column in numerical_columns:
    series = df[column].values
    H, p_value = analyze_tale(series)
    
    #results[column] = {'Hurst': H, 'KS_p_value': p_value}

for column, metrics in results.items():
    print(f"{column}:")
    print(f"  Hurst exponent: {metrics['Hurst']:.4f}")
    print(f"  KS test p-value: {metrics['KS_p_value']:.4f}")
    print()
# Save results to CSV
results.to_csv('sentences_self_similarity_analysis.csv', index=False)

# Plot Hurst exponent distributions
for metric in ['dep_distance', 'Eventfulness', 'I']:
    plot_hurst_distribution(results, metric)

print("Analysis complete. Results saved to 'tale_self_similarity_analysis.csv'")
print("Hurst exponent distribution plots saved as PNG files.")
