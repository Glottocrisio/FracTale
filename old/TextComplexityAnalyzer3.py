import numpy as np
from typing import List, Dict, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.fft import fft
from scipy import stats
import re
import os


class TextComplexityAnalyzer:
    def __init__(self):
        self.features = [
            "letter_length",
            "word_length",
            "clause_length",
            "sentence_length",
            "episode_length",
        ]
        self.distance_methods = {
            "linear": lambda d: 1 / d if d != 0 else 0,
            "exponential": lambda d: 1 / (d * d) if d != 0 else 0,
            "binary": lambda d: 1 if d <= 3 else 0,
        }

    def extract_features(self, text: str) -> Dict[str, np.ndarray]:
        """Extract numerical features from text."""
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        words = text.split()
        clauses = re.split(r"[;:,()\[\]{}]", text)
        clauses = [clause.strip() for clause in clauses if clause.strip()]
        episodes = [e.strip() for e in text.split("\n\n") if e.strip()]
        letters = [char for char in text if char.isalpha()]

        features = {
            "word_length": np.array([len(word) for word in words]),
            "sentence_length": np.array([len(sent.split()) for sent in sentences]),
            "clause_length": np.array([len(clause.split()) for clause in clauses]),
            "episode_length": np.array([len(episode.split()) for episode in episodes]),
            "letter_count": np.array([ord(letter) for letter in letters]),
        }
        return features

    def calculate_moran_i(self, data: np.ndarray, distance_method: str = "linear") -> float:
        """Calculate Moran's I statistic."""
        n = len(data)
        if n < 2:
            return 0.0  # Handle small data gracefully
        mean = np.mean(data)

        # Create a distance matrix and apply the chosen distance method
        dist_matrix = cdist(np.arange(n).reshape(-1, 1), np.arange(n).reshape(-1, 1), metric="cosine")
        weights = np.vectorize(self.distance_methods[distance_method])(dist_matrix)

        # Calculate Moran's I
        numerator = np.sum(weights * np.outer(data - mean, data - mean))
        denominator = np.sum((data - mean) ** 2)
        w_sum = np.sum(weights)

        if w_sum == 0 or denominator == 0:
            return 0.0

        return (n / w_sum) * (numerator / denominator)

    def hurst_exponent(self, data: np.ndarray) -> float:
        """Calculate Hurst exponent using R/S analysis."""
        n = len(data)
        if n < 20:  # Not enough data points
            return np.nan

        log_rs, log_k = [], []
        for k in range(2, int(np.log2(n)) + 1):
            size = n // k
            if size < 2:
                break
            segments = [data[i * size : (i + 1) * size] for i in range(k)]
            rs_values = [
                (np.max(np.cumsum(seg - np.mean(seg))) - np.min(np.cumsum(seg - np.mean(seg)))) / np.std(seg)
                for seg in segments
                if len(seg) > 1 and np.std(seg) > 0
            ]
            if rs_values:
                log_rs.append(np.log(np.mean(rs_values)))
                log_k.append(np.log(size))

        if len(log_k) > 1:
            hurst, _, _, _, _ = stats.linregress(log_k, log_rs)
            return hurst
        return np.nan

    def calculate_entropy(self, data: np.ndarray, bins: int = 20) -> float:
        """Calculate Shannon entropy of the data."""
        if len(data) == 0:
            return 0.0
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

    # def analyze_text(self, text: str, output_dir: str, corpus_name: str) -> None:
    #     """Analyze text and save results to CSV and plots."""
    #     features = self.extract_features(text)
    #     results = []

    #     os.makedirs(output_dir, exist_ok=True)
    #     plot_dir = os.path.join(output_dir, "plots")
    #     os.makedirs(plot_dir, exist_ok=True)

    #     for feature_name, data in features.items():
    #         # Calculate metrics
    #         moran_values = {method: self.calculate_moran_i(data, method) for method in self.distance_methods.keys()}
    #         entropy = self.calculate_entropy(data)
    #         hurst = self.hurst_exponent(data)
    #         mul_entropy = self.multiscale_entropy(data)

    #         # Save results
    #         results.append({
    #             "Feature": feature_name,
    #             "Entropy": entropy,
    #             "Hurst": hurst,
    #             "Mul_entropy" : mul_entropy,
    #             **moran_values
    #         })

    #         # Plot and save results
    #         plt.figure(figsize=(10, 6))
    #         plt.plot(data, label=feature_name, alpha=0.7)
    #         plt.title(f"{feature_name} - Data Plot")
    #         plt.xlabel("Index")
    #         plt.ylabel("Value")
    #         plt.legend()
    #         plt.grid()
    #         plot_path = os.path.join(plot_dir, f"{corpus_name}_{feature_name}.png")
    #         plt.savefig(plot_path)
    #         plt.close()

    #         for i, method in enumerate(['linear', 'exponential', 'binary']):
    # ax = plt.subplot2grid((3, 3), (1, i % 3))

    def analyze_corpus(self, corpus: str, output_dir: str) -> None:
            """Analyze a corpus and save results to a CSV and plots."""
            try:
                with open(corpus, "r", encoding="iso-8859-1") as file:
                    content = file.read()
                    tales = content.split("--------------------------------------------------\n\n")

                # Prepare results storage
                corpus_results = []

                # Create directories
                os.makedirs(output_dir, exist_ok=True)
                plot_dir = os.path.join(output_dir, "plots")
                os.makedirs(plot_dir, exist_ok=True)

                # Process each tale
                for tale_num, tale in enumerate(tales, start=1):
                    if not tale.strip():
                        continue
                    features = self.extract_features(tale)
                    tale_results = {"Tale Number": tale_num}

                    for feature_name, data in features.items():
                        # Calculate metrics
                        #moran_values = {f"Moran_linear": self.calculate_moran_i(data)}
                                       #for method in self.distance_methods.keys()}
                        #entropy = self.calculate_entropy(data)
                        hurst = self.hurst_exponent(data)
                        mul_entropy = self.multiscale_entropy(data)

                        # Add results to dictionary
                        tale_results.update({
                            #f"{feature_name}_Entropy": entropy,
                            f"{feature_name}_Hurst": hurst,
                            #f"{feature_name}_Mul_entropy" : mul_entropy
                            f"{feature_name}_Moran_linear": self.calculate_moran_i(data)
                        })

                        # # Plot and save
                        # plt.figure(figsize=(10, 6))
                        # plt.plot(data, label=feature_name, alpha=0.7)
                        # plt.title(f"{feature_name} - Tale {tale_num}")
                        # plt.xlabel("Index")
                        # plt.ylabel("Value")
                        # plt.legend()
                        # plt.grid()
                        # plot_path = os.path.join(plot_dir, f"{os.path.basename(corpus)}_tale_{tale_num}_{feature_name}.png")
                        # plt.savefig(plot_path)
                        # plt.close()

                    # Append tale results to corpus results
                    print("tale processed")
                    corpus_results.append(tale_results)
                    if tale_num ==5:
                        break
                # Save results for the corpus to a CSV
                results_df = pd.DataFrame(corpus_results)
                output_csv_path = os.path.join(output_dir, f"{os.path.basename(corpus)}_results_jensenshannon.csv")
                results_df.to_csv(output_csv_path, index=False)
                print(f"Results for {corpus} saved to {output_csv_path}")

            except Exception as e:
                print(f"Error processing {corpus}: {e}")


if __name__ == "__main__":
    analyzer = TextComplexityAnalyzer()
    corpora = ["grimm_tales_es.txt", "random_tales.txt", "europeana_stories_de.txt", "europeana_stories_en.txt"]
    output_dir = "text_analysis_results"

    for corpus in corpora:
        analyzer.analyze_corpus(corpus, output_dir)

# if __name__ == "__main__":
#     analyzer = TextComplexityAnalyzer()
#     corpora = ["grimm_tales_en.txt", "grimm_tales_es.txt", "grimm_tales_de.txt", "grimm_tales_it.txt"]
#     output_dir = "text_analysis_results"

#     for corpus in corpora:
#         try:

#             corpus_results = []
#             with open(corpus, "r", encoding="iso-8859-1") as file:
#                 content = file.read()
#                 tales = content.split("--------------------------------------------------\n\n")

#             for i, tale in enumerate(tales):
#                 if i < 4:
#                     continue
#                 if tale.strip():
#                     corpus_name = f"{os.path.splitext(os.path.basename(corpus))[0]}_tale_{i + 1}"
#                     analyzer.analyze_text(tale, output_dir, corpus_name)
#         except Exception as e:
#             print(f"Error processing {corpus}: {e}")

#                 # Save CSV
#         results_df = pd.DataFrame(results)
#         results_csv_path = os.path.join(output_dir, f"{corpus_name}_results.csv")
#         results_df.to_csv(results_csv_path, index=False)
#         print(f"Results saved to {results_csv_path}")

