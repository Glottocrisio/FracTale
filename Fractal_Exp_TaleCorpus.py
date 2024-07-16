import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.random import RandomState
from scipy import optimize
import os
import csv 



def calculate_D_h(row):
    a = float(row['avg_sentences/episode'])
    b = float(row['avg_clauses/sentence'])
    c = float(row['avg_words/clause'])
    D_h_a = np.log(3) / np.log(1 / a)
    D_h_b = np.log(3) / np.log(1 / b)
    D_h_c = np.log(3) / np.log(1 / c)
    D_h = (D_h_a + D_h_b + D_h_c) / 3
    return D_h

# Example usage
# N = 3
# h = 0.1
# D_h = calculate_D_h(N, h)
# print(f"The calculated fractal dimension D_h is: {D_h}")

def equation(d, a, b, c):
    return a**d + b**d + c**d - 1

def calculate_d(row):
    a = float(row['avg_sentences/episode'])/10
    b = float(row['avg_clauses/sentence'])/10
    c = float(row['avg_words/clause'])/10

    try:
        # Ensure initial bracket [0, 10] has function values with different signs
        if equation(0, a, b, c) * equation(10, a, b, c) >= 0:
            print(f"Cannot find a valid root for a={a}, b={b}, c={c}")
            return np.nan

        result = optimize.root_scalar(equation, args=(a, b, c), bracket=[0, 10], method='secant')
        return result.root if result.converged else np.nan
    except Exception as e:
        print(f"Error in calculation: {e}")
        return np.nan

def calculate_average_homotethy(row):
    a = float(row['avg_sentences/episode'])
    b = float(row['avg_clauses/sentence'])
    c = float(row['avg_words/clause'])

    return (a+b+c)/3

def insert_fractality_csv(input_file, output_file):
    df = pd.read_csv(input_file, delimiter=';')
    df['Average_homotethy'] = df.apply(calculate_average_homotethy, axis=1)
    df['D_h'] = df.apply(calculate_D_h, axis=1)
    df['D'] = df.apply(calculate_d, axis=1)
    df.to_csv(output_file, sep=';', index=False)

    print(f"Results saved to {output_file}")
    
# corpora_metrics = ['grimm_tales_metrics_en.csv', 'grimm_tales_metrics_de.csv', 'grimm_tales_metrics_es.csv', 'grimm_tales_metrics_it.csv']

# for corpus_metrics in corpora_metrics:
#     insert_fractality_csv(corpus_metrics, corpus_metrics.replace('.csv', '_fractality.csv'))


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

def analyze_corpus(df, columns_to_analyze):
    all_results = {}
    ks_between_series = {}
    
    for i, series1 in enumerate(columns_to_analyze):
        for j, series2 in enumerate(columns_to_analyze[i+1:], start=i+1):
            data1 = pd.to_numeric(df[series1], errors='coerce').dropna().values
            data2 = pd.to_numeric(df[series2], errors='coerce').dropna().values
            
            if len(data1) > 1 and len(data2) > 1:
                ks_stat, _ = stats.kstest(stats.zscore(data1), stats.zscore(data2))
                ks_between_series[f"{series1}_vs_{series2}"] = ks_stat
            else:
                ks_between_series[f"{series1}_vs_{series2}"] = None
    
    for col in columns_to_analyze:
        data = pd.to_numeric(df[col], errors='coerce').dropna().values
        
        # Generate fBm with different Hurst parameters
        n = len(data)
        hurst_params = np.arange(0.1, 1.0, 0.1)
        ks_stats = []
        
        for H_fbm in hurst_params:
            fBm_series = fbm(n, H_fbm)
            ks_stat, _ = stats.kstest(stats.zscore(data), stats.zscore(fBm_series))
            ks_stats.append(ks_stat)
        
        # Find the Hurst parameter with the lowest KS statistic
        best_H = hurst_params[np.argmin(ks_stats)]
        min_ks_stat = np.min(ks_stats)
        
        all_results[col] = {
            'best_H': best_H,
            'min_ks_stat': min_ks_stat
        }
    
    all_results['ks_between_series'] = ks_between_series
    print(all_results)
    return all_results

languages = ['en']#, 'de', 'es', 'it']
for lang in languages:
    df = pd.read_csv(f'grimm_tales_metrics_{lang}_fractality.csv', delimiter=';', decimal='.')

    columns_to_analyze = ['num_episodes', 'num_sentences', 'num_clauses', 'num_words', 'avg_sentences/episode', 'avg_clauses/sentence', 'avg_words/clause',
                          'avg_episode_length', 'avg_sentence_length', 'avg_clause_length',
                          'avg_dep_distance_clause', 'avg_dep_distance_sentence', 'avg_dep_distance_episode',
                          'Average_Eventfulness', 'Average_I', 'CLI', 'Average_homotethy', 'D_h'] #, 'CLI'] The Coleman-Liau index is reliable only for the English language

    # Analyze metrics
    all_results = analyze_corpus(df, columns_to_analyze)

    beauty_ranks = {
        1: 10, 2: 9.84, 3: 9.68, 4: 9.52, 5: 9.36,
        6: 9.2, 7: 9.04, 8: 8.88, 9: 8.72, 10: 8.56,
        11: 8.4, 12: 8.24, 13: 8.08, 14: 7.92, 15: 7.76,
        16: 7.6, 17: 7.44, 18: 7.28, 19: 7.12, 20: 6.96,
        21: 6.8, 22: 6.64, 23: 6.48, 24: 6.32, 25: 6.16
    }
    df['beauty_score'] = df['tale_id'].map(beauty_ranks)

    scores_data = []
    for col, results in all_results.items():
        if col != 'ks_between_series':
            scores_data.append({
                'metric': col,
                'best_H': results['best_H'],
                'min_ks_stat': results['min_ks_stat']
            })
    scores_df = pd.DataFrame(scores_data)

    correlation_matrix = df[columns_to_analyze + ['beauty_score']].corr(method='spearman')

    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    if lang == 'de':
        plt.title('Spearman Correlation Matrix for German Folktales')
    elif lang == 'es':
        plt.title('Spearman Correlation Matrix for Spanish Folktales')
    elif lang == 'it':
        plt.title('Spearman Correlation Matrix for Italian Folktales')
    plt.tight_layout()
    plt.savefig(f'correlation_matrix_def_{lang}.png')
    plt.close()

    print("Spearman Correlation Matrix:")
    print(correlation_matrix)


    # Find the best performing metric (excluding beauty_score)
    best_metric = correlation_matrix.loc[columns_to_analyze, 'beauty_score'].abs().idxmax()
    best_correlation = correlation_matrix.loc[best_metric, 'beauty_score']

    print(f"\nBest performing metric: {best_metric}")
    print(f"Correlation with beauty score: {best_correlation:.4f}")

    # Scatter plot of beauty score vs best performing metric
    plt.figure(figsize=(10, 6))
    plt.scatter(df[best_metric], df['beauty_score'])
    plt.xlabel(best_metric)
    plt.ylabel('Beauty Score')
    plt.title(f'Beauty Score vs {best_metric}')
    plt.tight_layout()
    plt.savefig(f'beauty_vs_best_metric_{lang}.png')
    plt.close()

    csv_data = []
    for _, row in df.iterrows():
        tale_data = {
            'tale_id': row['tale_id'],
            'beauty_score': row['beauty_score']
        }
        for col in columns_to_analyze:
            tale_data[f'{col}_value'] = row[col]
            tale_data[f'{col}_best_H'] = all_results[col]['best_H']
            tale_data[f'{col}_min_ks_stat'] = all_results[col]['min_ks_stat']
    
        for pair, ks_stat in all_results['ks_between_series'].items():
            tale_data[f'ks_stat_{pair}'] = ks_stat
    
        csv_data.append(tale_data)

    df_results = pd.DataFrame(csv_data)
    #df_results.to_csv(f'corpus_analysis_results_{lang}.csv', index=False)

    print("\nResults:")
    for col, results in all_results.items():
        if col != 'ks_between_series':
            print(f"{col}:")
            print(f"  Best H (fBm): {results['best_H']:.4f}")
            print(f"  Min KS statistic: {results['min_ks_stat']:.4f}")

    print("\nKS statistics between series:")
    for pair, ks_stat in all_results['ks_between_series'].items():
        print(f"  {pair}: {ks_stat:.4f}" if ks_stat is not None else f"  {pair}: N/A")
        

