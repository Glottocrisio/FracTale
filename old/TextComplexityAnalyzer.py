import numpy as np
from typing import List, Dict, Tuple, Union
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy import stats
import pywt
import math
import seaborn as sns
import re

class TextComplexityAnalyzer:
    def __init__(self):
        self.features = ['letter_length', 'word_length', 'clause_length', 'sentence_length', 'episode_length']
        self.scales = [2**i for i in range(3, 8)]
        self.distance_methods = {
            'linear': lambda d: 1/d if d != 0 else 0,
            'exponential': lambda d: 1/(d*d) if d != 0 else 0,
            'binary': lambda d: 1 if d <= 3 else 0
        }
        
    from typing import Dict

    def extract_features(self, text: str) -> Dict[str, np.ndarray]:
        """Extract numerical features from text."""
        # Splitting into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
    
        # Splitting into words
        words = text.split()
    
        # Splitting into clauses using any punctuation except ".", "?", or "!"
        clauses = re.split(r'[;:,()\[\]{}]', text)
        clauses = [clause.strip() for clause in clauses if clause.strip()]
    
        # Splitting into episodes using "\n\n"
        episodes = [e.strip() for e in text.split('\n\n') if e.strip()]
    
        # Extracting individual letters
        letters = [char for char in text if char.isalpha()]

        features = {
            'word_length': np.array([len(word) for word in words]),
            'sentence_length': np.array([len(sent.split()) for sent in sentences]),
            'clause_length': np.array([len(clause.split()) for clause in clauses]),
            'episode_length': np.array([len(episode.split()) for episode in episodes]),
            'letter_count': np.array([ord(letter) for letter in letters])
        }
        return features


    def calculate_moran_i(self, data: np.ndarray, 
                         distance_method: str = 'linear') -> Tuple[float, np.ndarray]:
        """Calculate Moran's I statistic."""
        n = len(data)
        
        # Calculate distance weights matrix
        weights = np.zeros((n, n))
        distance_func = self.distance_methods[distance_method]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    weights[i,j] = distance_func(abs(i-j))
        
        # Calculate Moran's I
        mean = np.mean(data)
        numerator = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    numerator += weights[i,j] * (data[i] - mean) * (data[j] - mean)
        
        denominator = np.sum((data - mean) ** 2)
        w_sum = np.sum(weights)
        
        if w_sum == 0 or denominator == 0:
            return 0, weights
            
        I = (n / w_sum) * (numerator / denominator)
        return I, weights
        
    def calculate_entropy(self, data: np.ndarray, bins: int = 20) -> float:
        """Calculate Shannon entropy of the data."""
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def multiscale_entropy(self, data: np.ndarray, max_scale: int = 20) -> List[float]:
        """Calculate entropy at multiple scales."""
        entropies = []
        for scale in range(1, max_scale + 1):
            coarsed = np.array([np.mean(data[i:i+scale]) 
                              for i in range(0, len(data)-scale+1, scale)])
            entropies.append(self.calculate_entropy(coarsed))
        return entropies
    
    def fft_analysis(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Perform FFT analysis on the data."""
        n = len(data)
        fft_vals = fft(data)
        freqs = np.fft.fftfreq(n)
        
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        amplitudes = np.abs(fft_vals)[pos_mask]
        
        return freqs, amplitudes
    
    def wavelet_analysis(self, data: np.ndarray, 
                        wavelet: str = 'db1', 
                        level: int = 5) -> Dict[str, np.ndarray]:
        """Perform wavelet-based multiscale analysis."""
        coeffs = pywt.wavedec(data, wavelet, level=level)
        energies = [np.sum(np.square(c)) for c in coeffs]
        
        stats_dict = {}
        for i, c in enumerate(coeffs):
            stats_dict[f'level_{i}_mean'] = np.mean(np.abs(c))
            stats_dict[f'level_{i}_std'] = np.std(c)
            stats_dict[f'level_{i}_energy'] = energies[i]
            
        return stats_dict
    
    def hurst_exponent(self, data: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Calculate Hurst exponent using R/S analysis."""
        data = np.array(data)
        n = len(data)
        max_k = int(np.floor(np.log2(n)))
        R_S = []
        k_values = []
        
        for k in range(2, max_k+1):
            size = int(np.floor(n/k))
            if size < 2:
                break
                
            R_s_values = []
            for i in range(k):
                start = i * size
                end = (i + 1) * size
                subseries = data[start:end]
                
                mean = np.mean(subseries)
                std = np.std(subseries)
                if std == 0:
                    continue
                    
                Z = np.cumsum(subseries - mean)
                R = np.max(Z) - np.min(Z)
                S = std
                
                if S > 0:
                    R_s_values.append(R/S)
            
            if R_s_values:
                R_S.append(np.mean(R_s_values))
                k_values.append(size)
        
        if len(k_values) > 1:
            log_k = np.log10(k_values)
            log_rs = np.log10(R_S)
            hurst, _, _, _, _ = stats.linregress(log_k, log_rs)
        else:
            hurst = np.nan
            
        return hurst, np.array(k_values), np.array(R_S)
    
    def analyze_text(self, text: str) -> Dict[str, Union[float, Dict, List]]:
        """Perform comprehensive analysis of text complexity."""
        features = self.extract_features(text)
        results = {}
        
        for feature_name, feature_data in features.items():
            moran_results = {method: self.calculate_moran_i(feature_data, method)
                           for method in self.distance_methods.keys()}
            
            feature_results = {
                'moran': moran_results,
                'entropy': self.calculate_entropy(feature_data),
                'multiscale_entropy': self.multiscale_entropy(feature_data),
                #'fft': self.fft_analysis(feature_data),
                #'wavelet': self.wavelet_analysis(feature_data),
                'hurst': self.hurst_exponent(feature_data)
            }
            results[feature_name] = feature_results
            
        return results
    
    def plot_results(self, results: Dict[str, Dict], feature_name: str):
        """Plot analysis results for a specific feature."""
        feature_results = results[feature_name]
        
        fig = plt.figure(figsize=(20, 15))
        
        # Plot 1: Multiscale Entropy
        ax1 = plt.subplot2grid((3, 3), (0, 0))
        ax1.plot(feature_results['multiscale_entropy'])
        ax1.set_title('Multiscale Entropy')
        ax1.set_xlabel('Scale')
        ax1.set_ylabel('Entropy')
        
        # Plot 2: FFT
        # ax2 = plt.subplot2grid((3, 3), (0, 1))
        # freqs, amps = feature_results['fft']
        # ax2.plot(freqs, amps)
        # ax2.set_title('Fourier Analysis')
        # ax2.set_xlabel('Frequency')
        # ax2.set_ylabel('Amplitude')
        
        # # Plot 3: Wavelet Energies
        # ax3 = plt.subplot2grid((3, 3), (0, 2))
        # wavelet_results = feature_results['wavelet']
        # energies = [wavelet_results[f'level_{i}_energy'] 
        #            for i in range(len(wavelet_results)//3)]
        # ax3.bar(range(len(energies)), energies)
        # ax3.set_title('Wavelet Energy by Level')
        # ax3.set_xlabel('Decomposition Level')
        # ax3.set_ylabel('Energy')
        
        # Plot 4: Hurst Analysis
        ax4 = plt.subplot2grid((3, 3), (1, 0))
        hurst, k_values, rs_values = feature_results['hurst']
        ax4.loglog(k_values, rs_values, 'b.')
        ax4.set_title(f'R/S Analysis (H={hurst:.3f})')
        ax4.set_xlabel('log(k)')
        ax4.set_ylabel('log(R/S)')
        
        # Plot 5-7: Moran's I for different distance methods
        # for i, method in enumerate(['linear', 'exponential', 'binary']):
        #     ax = plt.subplot2grid((3, 3), (1, i%3))
        #     moran_i, weights = feature_results['moran'][method]
        #     sns.heatmap(weights, ax=ax, cmap='YlOrRd')
        #     ax.set_title(f"Moran's I - {method}\nI={moran_i:.3f}")
        
        # plt.tight_layout()
        # return fig

# Example usage
if __name__ == "__main__":

    


    # Create analyzer instance
    analyzer = TextComplexityAnalyzer()
    corpora = ['grimm_tales_en.txt', 'grimm_tales_es.txt', 'grimm_tales_de.txt', 'grimm_tales_it.txt',]
    for corpus in corpora:
        with open(corpus, 'r', encoding='iso-8859-1') as file:
            content = file.read()
            tales = content.split('--------------------------------------------------\n\n')

        for tale in tales:
            results = analyzer.analyze_text(tale)

            # Plot results
            fig = analyzer.plot_results(results, 'episode_length')
            plt.show()

            # Print numerical results
            for feature_name, feature_results in results.items():
                print(f"\nResults for {feature_name}:")
                print(f"Entropy: {feature_results['entropy']:.3f}")
                print("Moran's I values:")
                for method, (moran_i, _) in feature_results['moran'].items():
                    print(f"  {method}: {moran_i:.3f}")
                print(f"Hurst exponent: {feature_results['hurst'][0]:.3f}")