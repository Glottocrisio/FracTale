import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def generate_spearman_correlation_matrices(csv_file, output_dir='correlation_matrices'):
    # Create output directory if it doesn't exist
    #output_dir = output_dir+f"\\{csv_file[-40:]}".replace(".csv","")
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    df = pd.read_csv(csv_file)
    
    parameters = [
        'avg_dep_distance_sentence',
        'avg_dep_distance_episode',
        'Average_Eventfulness',
        'Average_I',
        'CLI',
        'D_h',
        'avg_homothety',
        'hurst_dep_distance',
        'hurst_eventfulness',
        'hurst_I',
        'Fractal Dimension (Sentences)',
        'Fractal Dimension (Episodes)'
    ]
    
    # Get the first 5 tales
    tales = df.head(5)
    
    # Create a sliding window to compute correlations
    window_size = 5  # Using a window size of 5 for better correlation estimation
    
    # Set up the plotting style
    #plt.style.use('seaborn')
    
    for idx, current_tale in tales.iterrows():
        tale_id = f"Tale {int(current_tale['tale_id'])}"
        
        matrix = np.zeros((len(parameters), len(parameters)))
        
        # Calculate Spearman correlations using a sliding window around the current tale
        start_idx = max(0, idx - window_size // 2)
        end_idx = min(len(df), idx + window_size // 2 + 1)
        window_data = df.iloc[start_idx:end_idx]
        
        for i, param1 in enumerate(parameters):
            for j, param2 in enumerate(parameters):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    correlation, _ = stats.spearmanr(
                        window_data[param1],
                        window_data[param2],
                        nan_policy='omit'
                    )
                    matrix[i][j] = correlation if not np.isnan(correlation) else 0
        
        corr_df = pd.DataFrame(matrix, index=parameters, columns=parameters)
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(corr_df, 
                   annot=True,  
                   cmap='RdBu_r',  
                   vmin=-1, 
                   vmax=1,
                   center=0,
                   fmt='.2f',  
                   square=True,  
                   cbar_kws={'label': 'Spearman Correlation'})
        

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        

        plt.title(f'Spearman Correlation Matrix for {tale_id}', pad=20)
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, f'spearman_correlation_matrix_tale_{int(current_tale["tale_id"])}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Generated Spearman correlation matrix for {tale_id}")

if __name__ == "__main__":
    
    folder = "C:\\Users\\Palma\\Desktop\\PHD\\FracTale\\"  

    files = [f"{folder}grimm_unified_metrics_en_ugly_complete.csv",
             f"{folder}grimm_unified_metrics_de_complete.csv",
             f"{folder}europeana_unified_metrics_de_complete.csv",
             f"{folder}grimm_unified_metrics_es_complete.csv",
             f"{folder}grimm_unified_metrics_es_ugly_complete.csv",
             f"{folder}grimm_unified_metrics_it_complete.csv",
             f"{folder}grimm_unified_metrics_it_ugly_complete.csv",
             f"{folder}random_unified_metrics_complete.csv",
             f"{folder}europeana_unified_metrics_en_complete.csv",
             f"{folder}grimm_unified_metrics_de_ugly_complete.csv",
             
             ]

    for file in files:

        if file == f"{folder}grimm_unified_metrics_en_ugly_complete.csv":
            generate_spearman_correlation_matrices(file, output_dir=f"{folder}correlation_matrices\\grimm_unified_metrics_en_ugly_complete")
        elif file == f"{folder}grimm_unified_metrics_de_complete.csv":
            generate_spearman_correlation_matrices(file, output_dir=f"{folder}correlation_matrices\\grimm_unified_metrics_de_complete")
        elif file == f"{folder}europeana_unified_metrics_de_complete.csv":
            generate_spearman_correlation_matrices(file, output_dir=f"{folder}correlation_matrices\\europeana_unified_metrics_de_complete")
        elif file == f"{folder}grimm_unified_metrics_es_complete.csv":
            generate_spearman_correlation_matrices(file, output_dir=f"{folder}correlation_matrices\\grimm_unified_metrics_es_complete")
        elif file == f"{folder}grimm_unified_metrics_es_ugly_complete.csv":
            generate_spearman_correlation_matrices(file, output_dir=f"{folder}correlation_matrices\\grimm_unified_metrics_es_ugly_complete")
        elif file == f"{folder}grimm_unified_metrics_it_complete.csv":
            generate_spearman_correlation_matrices(file, output_dir=f"{folder}correlation_matrices\\grimm_unified_metrics_it_complete")
        elif file == f"{folder}grimm_unified_metrics_it_ugly_complete.csv":
            generate_spearman_correlation_matrices(file, output_dir=f"{folder}correlation_matrices\\grimm_unified_metrics_it_ugly_complete")
        elif file == f"{folder}random_unified_metrics_complete.csv":
            generate_spearman_correlation_matrices(file, output_dir=f"{folder}correlation_matrices\\random_unified_metrics_complete")
        elif file == f"{folder}europeana_unified_metrics_en_complete.csv":
            generate_spearman_correlation_matrices(file, output_dir=f"{folder}correlation_matrices\\europeana_unified_metrics_en_complete")
        elif file == f"{folder}grimm_unified_metrics_de_ugly_complete.csv":
            generate_spearman_correlation_matrices(file, output_dir=f"{folder}correlation_matrices\\grimm_unified_metrics_de_ugly_complete")
    print("\nAll Spearman correlation matrices have been generated!")