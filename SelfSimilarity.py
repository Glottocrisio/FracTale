import pandas as pd
import numpy as np
from hurst import compute_Hc
from scipy import stats
import matplotlib.pyplot as plt

def calculate_hurst(time_series):
    H, _, _ = compute_Hc(time_series)
    return H

def ks_test(series1, series2):
    return stats.ks_2samp(series1, series2).statistic

def analyze_tale(tale_data):
    metrics = ['dep_distance', 'Eventfulness', 'I']
    
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

def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Print basic information about the dataframe
    print(df.info())
    print("\nFirst few rows:")
    print(df.head())
    
    # Calculate basic statistics for numerical columns
    numerical_columns = ['dep_distance', 'Eventfulness', 'I']
    print("\nBasic statistics:")
    print(df[numerical_columns].describe())
    
    # Group by tale_id and calculate mean for each metric
    tale_means = df.groupby('tale_id')[numerical_columns].mean()
    print("\nMean values for each tale:")
    print(tale_means)

    return df, tale_means

# Main execution
csv_file = 'grimm_sentences_metrics_en.csv'  # Replace with your CSV file path
df, tale_means = process_csv(csv_file)

print("\nAnalysis complete.")
# Save results to CSV
