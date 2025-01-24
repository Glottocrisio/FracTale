import pandas as pd
import numpy as np
from scipy import stats

def calculate_D_h(row):
    """Calculate D_h metric for homothety."""
    try:
        a = float(row['avg_episode_length'])
        b = float(row['avg_sentence_length'])
        c = float(row['avg_clause_length'])
        D_h_a = np.log(3) / np.log(1 / a)
        D_h_b = np.log(3) / np.log(1 / b)
        D_h_c = np.log(3) / np.log(1 / c)
        D_h = (D_h_a + D_h_b + D_h_c) / 3
        return D_h
    except Exception as e:
        print(f"Error calculating D_h for tale {row['tale_id']}: {e}")
        return np.nan

def calculate_average_homothety(row):
    """Calculate average homothety."""
    try:
        d = float(row['avg_episode_length'])*float(row['num_episodes'])/(float(row['num_words']))
        a = float(row['avg_clause_length'])/d
        b = float(row['avg_sentence_length'])/float(row['avg_episode_length'])
        c = float(row['avg_clause_length'])/float(row['avg_sentence_length'])
        return (a + b + c) / 3
    except Exception as e:
        print(f"Error calculating average homothety for tale {row['tale_id']}: {e}")
        return np.nan

def calculate_hurst(data):
    """Calculate Hurst exponent using R/S analysis."""
    try:
        data = np.array(data)
        # Remove any NaN values
        data = data[~np.isnan(data)]
        
        n = len(data)
        if n < 20:  # Not enough data points
            return np.nan
        
        log_rs, log_k = [], []
        for k in range(2, int(np.log2(n)) + 1):
            size = n // k
            if size < 2:
                break
            
            segments = [data[i * size : (i + 1) * size] for i in range(k)]
            rs_values = []
            
            for seg in segments:
                if len(seg) > 1 and np.std(seg) > 0:
                    mean_adj = seg - np.mean(seg)
                    cumsum = np.cumsum(mean_adj)
                    r = np.max(cumsum) - np.min(cumsum)
                    s = np.std(seg)
                    rs_values.append(r/s)
            
            if rs_values:
                log_rs.append(np.log(np.mean(rs_values)))
                log_k.append(np.log(size))
        
        if len(log_k) > 1:
            hurst, *_ = stats.linregress(log_k, log_rs)
            return hurst
        return np.nan
    except Exception as e:
        print(f"Error calculating Hurst exponent: {e}")
        return np.nan

def split_file(input_file):
    """Split the input file into two parts based on headers."""
    try:
        
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        separator_idx = None
        for i, line in enumerate(lines):
            if line.strip().startswith('tale_id,episode_id'):
                separator_idx = i
                break
            elif not line.strip():  # empty line
                separator_idx = i
                break
        
        if separator_idx is None:
            print(f"Warning: Could not find section separator in {input_file}")
            return None, None
        
        temp_file1 = input_file.replace('.csv', '') + '_tales.csv'
        temp_file2 = input_file.replace('.csv', '') + '_sentences.csv'
        
        with open(temp_file1, 'w', encoding='utf-8') as f:
            f.writelines(lines[:separator_idx])
        
        with open(temp_file2, 'w', encoding='utf-8') as f:
            f.writelines(lines[separator_idx:])
        
        return temp_file1, temp_file2
    
    except Exception as e:
        print(f"Error splitting file: {e}")
        return None, None

def process_data(input_file, output_file):
    """Main function to process the data and calculate all metrics."""
    try:
        # Split the input file into two parts
        #metrics_file, sequences_file = split_file(input_file)
        #if metrics_file is None or sequences_file is None:
        #    raise Exception("Failed to split input file")

        if input_file[-7:] == 'les.csv':
            print("Processing metrics and calculating homothety...")
            df_metrics = pd.read_csv(input_file)
    
            required_columns = ['avg_episode_length', 'avg_sentence_length', 'avg_clause_length']
            if all(column in df_metrics.columns for column in required_columns):
                df_metrics['D_h'] = df_metrics.apply(calculate_D_h, axis=1)
                df_metrics['avg_homothety'] = df_metrics.apply(calculate_average_homothety, axis=1)
            else:
                print(f"Error: Missing one or more required columns in {input_file}")
                df_metrics['D_h'] = np.nan
                df_metrics['avg_homothety'] = np.nan

            # Save results to a new Excel file
            #output_excel_file = input_file.rsplit('.', 1)[0] + '_processed.csv'
            df_metrics.to_csv(output_file, index=False)
            print(f"Results successfully saved to {output_file}")
        
        else:
            print("Processing sequences and calculating Hurst exponents...")
            #df_metrics = pd.read_csv(input_file)
            df_sequences = pd.read_csv(input_file, delimiter=',')
        
            tale_metrics = []
            for tale_id in df_sequences['tale_id'].unique():
                tale_sequences = df_sequences[df_sequences['tale_id'] == tale_id]
            
                metrics = {
                    'tale_id': tale_id,
                    'hurst_dep_distance': calculate_hurst(tale_sequences['dep_distance']),
                    'hurst_eventfulness': calculate_hurst(tale_sequences['Eventfulness']),
                    'hurst_I': calculate_hurst(tale_sequences['I'])
                }
                tale_metrics.append(metrics)
        
            tale_metrics_df = pd.DataFrame(tale_metrics)
            final_df = pd.merge(df_sequences, tale_metrics_df, on='tale_id', how='left')
        
            print("Saving results...")
            final_df.to_csv(output_file, index=False)
            print(f"Results successfully saved to {output_file}")
        
            # Clean up temporary files
            #import os
            #os.remove(metrics_file)
            #os.remove(sequences_file)
        
        # Print summary statistics
        #print("\nSummary Statistics:")
       # print(final_df[['D_h', 'avg_homothety', 
                     #  'hurst_dep_distance', 'hurst_eventfulness', 
                     #  'hurst_I']].describe())
        
    except Exception as e:
        print(f"Error processing data: {e}")

def merge_metrics(tales_file, sentences_file, fractal_file, output_file):
    """
    Merge metrics from three input files.
    
    Parameters:
    tales_file: CSV with tale-level metrics including homothety metrics
    sentences_file: CSV with sentence-level metrics and Hurst exponents
    fractal_file: CSV with fractal dimensions
    output_file: Path for the output CSV file
    """
    try:
        print("Reading input files...")
        df_tales = pd.read_csv(tales_file)
        df_sentences = pd.read_csv(sentences_file)
        df_fractal = pd.read_csv(fractal_file)
        
        hurst_columns = ['tale_id', 'hurst_dep_distance', 'hurst_eventfulness', 'hurst_I']
        df_hurst = df_sentences[hurst_columns].drop_duplicates(subset=['tale_id'])
        
        df_fractal['tale_id'] = df_fractal['Tale'].str.extract('(\d+)').astype(float)
        
        print("Merging datasets...")
        df_merged = pd.merge(df_tales, df_hurst, on='tale_id', how='left')
        
        df_merged = pd.merge(
            df_merged,
            df_fractal[[
                'tale_id',
                'Fractal Dimension (Letters)',
                'Fractal Dimension (Words)',
                'Fractal Dimension (Clauses)',
                'Fractal Dimension (Sentences)',
                'Fractal Dimension (Episodes)'
            ]],
            on='tale_id',
            how='left'
        )
        
        required_columns = [
            'tale_id', 'num_episodes', 'num_sentences', 'num_clauses', 'num_words',
            'avg_episode_length', 'avg_sentence_length', 'avg_clause_length',
            'avg_dep_distance_clause', 'avg_dep_distance_sentence', 'avg_dep_distance_episode',
            'Average_Eventfulness', 'Average_I', 'CLI', 'D_h', 'avg_homothety', 
            'hurst_dep_distance', 'hurst_eventfulness', 'hurst_I',
            'Fractal Dimension (Letters)', 'Fractal Dimension (Words)',
            'Fractal Dimension (Clauses)', 'Fractal Dimension (Sentences)',
            'Fractal Dimension (Episodes)'
        ]
        
        missing_columns = set(required_columns) - set(df_merged.columns)
        if missing_columns:
            print(f"Warning: Missing columns in output: {missing_columns}")
            for col in missing_columns:
                df_merged[col] = np.nan
        
        df_merged = df_merged[required_columns]
        
        print("Saving results...")
        df_merged.to_csv(output_file, index=False)
        print(f"Results successfully saved to {output_file}")
        
        print("\nOutput file statistics:")
        print(f"Number of tales processed: {len(df_merged)}")
        print("\nColumns with missing values:")
        missing_vals = df_merged.isnull().sum()
        print(missing_vals[missing_vals > 0])
        
    except Exception as e:
        print(f"Error processing files: {e}")




if __name__ == "__main__":
    #List of input files to process
    input_files = [
        #'grimm_unified_metrics_en_sentences.csv',
        #'grimm_unified_metrics_de_sentences.csv',
        #'grimm_unified_metrics_es_sentences.csv',
        #'grimm_unified_metrics_it_sentences.csv',
        #'grimm_unified_metrics_en_ugly_sentences.csv',
        #'grimm_unified_metrics_de_ugly_sentences.csv',
        #'grimm_unified_metrics_es_ugly_sentences.csv',
        #'grimm_unified_metrics_it_ugly_sentences.csv',
        #'europeana_unified_metrics_de_sentences.csv',
        'europeana_unified_metrics_en_sentences.csv',
        #'random_unified_metrics_sentences.csv',
    #     'grimm_unified_metrics_en_tales.csv',
    #     'grimm_unified_metrics_de_tales.csv',
    #     'grimm_unified_metrics_es_tales.csv',
    #     'grimm_unified_metrics_it_tales.csv',
         #'grimm_unified_metrics_en_ugly_tales.csv',
         #'grimm_unified_metrics_de_ugly_tales.csv',
         #'grimm_unified_metrics_es_ugly_tales.csv',
         #'grimm_unified_metrics_it_ugly_tales.csv',
    #     'europeana_unified_metrics_de_tales.csv',
         'europeana_unified_metrics_en_tales.csv',
    #     'random_unified_metrics_tales.csv'
    ]
    
    for input_file in input_files:
        try:
            output_file = input_file.rsplit('.', 1)[0] + 'hh.' + input_file.rsplit('.', 1)[1]
            
            print(f"\nProcessing {input_file}...")
            process_data(input_file, output_file)
            print(f"Completed processing {input_file} -> {output_file}\n")
            
        except Exception as e:
            print(f"Error processing {input_file}: {e}\n")

#if __name__ == "__main__":
    folder = "C:\\Users\\Palma\\Desktop\\PHD\\FracTale\\text_analysis_results\\"
    # tales_files = [f'{folder}grimm_unified_metrics_en_taleshh.csv', f'{folder}grimm_unified_metrics_it_taleshh.csv', f'{folder}grimm_unified_metrics_es_taleshh.csv', f'{folder}grimm_unified_metrics_en_ugly_taleshh.csv', f'{folder}grimm_unified_metrics_it_ugly_taleshh.csv', f'{folder}grimm_unified_metrics_es_ugly_taleshh.csv', f'{folder}europeana_unified_metrics_en_taleshh.csv', f'{folder}europeana_unified_metrics_de_taleshh.csv', f'{folder}random_unified_metrics_de_taleshh.csv']
    # sentences_files = [f'{folder}grimm_unified_metrics_en_sentenceshh.csv', f'{folder}grimm_unified_metrics_it_sentenceshh.csv', f'{folder}grimm_unified_metrics_es_sentenceshh.csv']
    # fractal_files = [f'{folder}grimm_tales_en_results.csv', f'{folder}grimm_tales_it_results.csv', f'{folder}grimm_tales_es_results.csv']
    # output_files = [f'{folder}grimm_unified_metrics_en_complete.csv', f'{folder}grimm_unified_metrics_it_complete.csv', f'{folder}grimm_unified_metrics_es_complete.csv']
    tales_files = [
    # f'{folder}grimm_unified_metrics_en_taleshh.csv',
    # f'{folder}grimm_unified_metrics_it_taleshh.csv',
    # f'{folder}grimm_unified_metrics_es_taleshh.csv',
    # f'{folder}grimm_unified_metrics_en_ugly_taleshh.csv',
    # f'{folder}grimm_unified_metrics_it_ugly_taleshh.csv',
    # f'{folder}grimm_unified_metrics_es_ugly_taleshh.csv',
    # f'{folder}europeana_unified_metrics_en_taleshh.csv',
    # f'{folder}europeana_unified_metrics_de_taleshh.csv',
    # f'{folder}random_unified_metrics_taleshh.csv',
    f'{folder}europeana_unified_metrics_en_taleshh.csv',
    #f'{folder}grimm_unified_metrics_de_ugly_taleshh.csv'
]

    sentences_files = [
        # f'{folder}grimm_unified_metrics_en_sentenceshh.csv',
        # f'{folder}grimm_unified_metrics_it_sentenceshh.csv',
        # f'{folder}grimm_unified_metrics_es_sentenceshh.csv',
        # f'{folder}grimm_unified_metrics_en_ugly_sentenceshh.csv',
        # f'{folder}grimm_unified_metrics_it_ugly_sentenceshh.csv',
        # f'{folder}grimm_unified_metrics_es_ugly_sentenceshh.csv',
        # f'{folder}europeana_unified_metrics_en_sentenceshh.csv',
        # f'{folder}europeana_unified_metrics_de_sentenceshh.csv',
        # f'{folder}random_unified_metrics_sentenceshh.csv',
        f'{folder}europeana_unified_metrics_en_sentenceshh.csv',
        #f'{folder}grimm_unified_metrics_de_ugly_sentenceshh.csv'
    ]

    fractal_files = [
        # f'{folder}grimm_tales_en_results.csv',
        # f'{folder}grimm_tales_it_results.csv',
        # f'{folder}grimm_tales_es_results.csv',
        # f'{folder}ugly_en_grimm_tales_results.csv',
        # f'{folder}ugly_it_grimm_tales_results.csv',
        # f'{folder}ugly_es_grimm_tales_results.csv',
        # f'{folder}europeana_stories_en_results.csv',
        # f'{folder}europeana_stories_de_results.csv',
        # f'{folder}random_tales_results.csv',
        f'{folder}europeana_stories_en_results.csv',
        #f'{folder}ugly_de_grimm_tales_results.csv'
    ]

    output_files = [
        # f'{folder}grimm_unified_metrics_en_complete.csv',
        # f'{folder}grimm_unified_metrics_it_complete.csv',
        # f'{folder}grimm_unified_metrics_es_complete.csv',
        # f'{folder}grimm_unified_metrics_en_ugly_complete.csv',
        # f'{folder}grimm_unified_metrics_it_ugly_complete.csv',
        # f'{folder}grimm_unified_metrics_es_ugly_complete.csv',
        # f'{folder}europeana_unified_metrics_en_complete.csv',
        # f'{folder}europeana_unified_metrics_de_complete.csv',
        # f'{folder}random_unified_metrics_complete.csv',
        f'{folder}europeana_unified_metrics_en_complete.csv',
        #f'{folder}grimm_unified_metrics_de_ugly_complete.csv'
    ]
    i = 0

    while i < len(tales_files):
        tales_file = tales_files[i]
        sentences_file = sentences_files[i]
        fractal_file = fractal_files[i]
        output_file = output_files[i]
        merge_metrics(tales_file, sentences_file, fractal_file, output_file)
        i = i + 1