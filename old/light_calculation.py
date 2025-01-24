import numpy as np
import re
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import unicodedata
import chardet

class MultilingualHierarchicalDecomposition:
    def __init__(self):
        self.levels = ['episodes', 'sentences', 'words', 'letters']
        self.sentence_endings = r'[.!?;]["\']?\s+'
        
        # Using Unicode escape sequences instead of direct special characters
        self.special_chars = {
            'german': '\u00e4\u00f6\u00fc\u00df\u00c4\u00d6\u00dc',  # äöüßÄÖÜ
            'italian': '\u00e0\u00e8\u00e9\u00ec\u00ed\u00ee\u00f2\u00f3\u00f9\u00fa\u00c0\u00c8\u00c9\u00cc\u00cd\u00ce\u00d2\u00d3\u00d9\u00da'  # àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ
        }

    @staticmethod
    def detect_encoding(file_path: str) -> str:
        """Detect the encoding of a file."""
        with open(file_path, 'rb') as file:
            raw_data = file.read()
        result = chardet.detect(raw_data)
        return result['encoding']

    @staticmethod
    def read_file_with_encoding(file_path: str) -> str:
        """Read file with appropriate encoding."""
        # Try common encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        # First try detected encoding
        detected_encoding = MultilingualHierarchicalDecomposition.detect_encoding(file_path)
        if detected_encoding:
            encodings.insert(0, detected_encoding)
        
        # Try each encoding
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not read file with any of the attempted encodings: {encodings}")
        
    def normalize_text(self, text: str) -> str:
        """Normalize Unicode text while preserving special characters."""
        text = unicodedata.normalize('NFKD', text)
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch == '\n')
        return text
        
    def split_into_episodes(self, text: str) -> List[str]:
        """Split text into episodes based on double newlines."""
        episodes = text.split('\n\n')
        return [ep.strip() for ep in episodes if ep.strip()]
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with support for multiple languages."""
        text = re.sub(r'(Mr\.|Mrs\.|Dr\.|Prof\.|etc\.|vs\.|Fig\.|St\.)', r'\1<stop>', text)
        sentences = re.split(self.sentence_endings, text)
        sentences = [s.replace('<stop>', '.').strip() for s in sentences]
        return [s for s in sentences if s]
    
    def split_into_words(self, text: str) -> List[str]:
        """Split text into words, handling special characters."""
        text = text.lower()
        special_chars = ''.join(chars for chars in self.special_chars.values())
        pattern = fr"[a-z{special_chars}]+(?:[''][a-z{special_chars}]+)*"
        words = re.findall(pattern, text)
        return [w for w in words if w]
    
    def get_letters(self, text: str) -> List[str]:
        """Get all letters including special characters."""
        special_chars = ''.join(chars for chars in self.special_chars.values())
        pattern = f'[a-zA-Z{special_chars}]'
        return [c for c in text.lower() if re.match(pattern, c)]
    
    def count_elements_at_levels(self, text: str) -> Dict[str, int]:
        """Count elements at each hierarchical level."""
        text = self.normalize_text(text)
        counts = {}
        counts['episodes'] = len(self.split_into_episodes(text))
        counts['sentences'] = len(self.split_into_sentences(text))
        counts['words'] = len(self.split_into_words(text))
        counts['letters'] = len(self.get_letters(text))
        return counts
    
    def calculate_sizes(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Calculate average size of elements at each level."""
        sizes = {}
        total_letters = counts['letters']
        for level in self.levels:
            if counts[level] > 0:
                sizes[level] = total_letters / counts[level]
            else:
                sizes[level] = 0
        return sizes
    
    def calculate_fractal_dimension(self, text: str) -> Tuple[float, Dict, plt.Figure]:
        """Calculate fractal dimension using hierarchical decomposition."""
        counts = self.count_elements_at_levels(text)
        sizes = self.calculate_sizes(counts)
        
        x_data = []
        y_data = []
        valid_levels = []
        
        for level in self.levels:
            if sizes[level] > 0 and counts[level] > 0:
                x_data.append(np.log(sizes[level]))
                y_data.append(np.log(counts[level]))
                valid_levels.append(level)
        
        coeffs = np.polyfit(x_data, y_data, 1)
        fractal_dim = -coeffs[0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(x_data, y_data, color='blue', label='Data points')
        ax.plot(x_data, np.polyval(coeffs, x_data), 'r--', 
                label=f'Fit (D = {fractal_dim:.4f})')
        
        ax.set_xlabel('log(Size)')
        ax.set_ylabel('log(Count)')
        ax.set_title('Hierarchical Decomposition Analysis')
        ax.legend()
        ax.grid(True)
        
        for i, level in enumerate(valid_levels):
            ax.annotate(level, (x_data[i], y_data[i]), 
                       xytext=(5, 5), textcoords='offset points')
        
        return fractal_dim, counts, fig

# Example usage
if __name__ == "__main__":
    try:
        hd = MultilingualHierarchicalDecomposition()
        file_path = 'rotkapp.txt'
        text = hd.read_file_with_encoding(file_path)
        dimension, counts, fig = hd.calculate_fractal_dimension(text)
        
        print(f"Fractal Dimension: {dimension:.4f}")
        print("\nCounts at each level:")
        for level, count in counts.items():
            print(f"{level}: {count}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")