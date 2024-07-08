import nltk
from nltk import ngrams, FreqDist
import os
from collections import defaultdict
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.parse import DependencyGraph
import re
import math

def calculateProppNgrams(n, sequences, sign):
    
    all_ngrams = []
    for sequence in sequences:
        ngram_list = list(ngrams(sequence, n))
        all_ngrams.extend(ngram_list)

        freq_dist = FreqDist(all_ngrams)

    total_ngrams = len(all_ngrams)
    ngrams_with_sign = sum(freq for ngram, freq in freq_dist.items() if sign in ngram)

    # Likelihood of an n-gram containing 'S'
    likelihood = ngrams_with_sign / total_ngrams

    return total_ngrams, ngrams_with_sign, likelihood


nltk.download('punkt')

def dependency_distance(doc):
    with open(file_path, 'r', encoding='iso-8859-1') as file:
        content = file.read()
    total_distance = 0
    for token in content:
        if token.dep_ != "ROOT":
            distance = abs(token.i - token.head.i)
            total_distance += distance
    return total_distance

def calculate_DD(tale_text):
    # Tokenize the tale into sentences
    sentences = nltk.sent_tokenize(tale_text)
    
    # Initialize metrics
    metrics = defaultdict(float)
    metrics['num_sentences'] = len(sentences)
    metrics['num_episodes'] = len(tale_text.split('--------------------------------------------------\n\n'))
    
    for sentence in sentences:
        doc = nlp(sentence)
        
        # Count clauses, words, and letters
        clauses = [sent for sent in doc.sents]
        words = [token for token in doc if not token.is_punct]
        letters = sum(len(word.text) for word in words)
        
        metrics['num_clauses'] += len(clauses)
        metrics['num_words'] += len(words)
        metrics['num_letters'] += letters
        
        # Calculate dependency distance
        metrics['total_dep_distance'] += dependency_distance(doc)
    
    # Calculate averages
    
    return metrics

def process_files(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                tale_text = file.read()
                results[filename] = analyze_tale(tale_text)
    return results

# Usage
#directory = "path/to/your/tales/directory"
#analysis_results = process_files(directory)

# Print or save results as needed
# for filename, metrics in analysis_results.items():
#     print(f"Analysis for {filename}:")
#     for metric, value in metrics.items():
#         print(f"{metric}: {value}")
#     print("\n")
    


def coleman_liau_index(text):
    words = len(re.findall(r'\w+', text))
    sentences = len(re.findall(r'\w+[.!?]', text)) or 1  # Ensure at least 1 sentence
    letters = sum(c.isalpha() for c in text)
    
    L = (letters / words) * 100
    S = (sentences / words) * 100
    
    return 0.0588 * L - 0.296 * S - 15.8

def process_file_coleman_liau_index(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    tales = content.split('--------------------------------------------------\n\n')
    
    for i, tale in enumerate(tales, 1):
        if tale.strip():  # Ignore empty tales
            index = coleman_liau_index(tale)
            print(f"Tale {i}: Coleman-Liau Index = {index:.2f}")

# Usage
#file_path = 'grimm_tales_en.txt'
#process_file_coleman_liau_index(file_path)

# Assuming metrics['total_dep_distance'] is already calculated

# metrics['#E'] = metrics['num_episodes']
# metrics['#S'] = metrics['num_sentences']
# metrics['#C'] = metrics['num_clauses']
# metrics['#W'] = metrics['num_words']

# metrics['AEL'] = metrics['#S'] / metrics['#E'] if metrics['#E'] > 0 else 0
# metrics['ASL'] = metrics['#W'] / metrics['#S'] if metrics['#S'] > 0 else 0
# metrics['ACL'] = metrics['#W'] / metrics['#C'] if metrics['#C'] > 0 else 0

# metrics['ADDc'] = metrics['total_dep_distance'] / metrics['#C'] if metrics['#C'] > 0 else 0
# metrics['ADDs'] = metrics['total_dep_distance'] / metrics['#S'] if metrics['#S'] > 0 else 0
# metrics['ADDe'] = metrics['total_dep_distance'] / metrics['#E'] if metrics['#E'] > 0 else 0

# metrics['#Ev'] = metrics['#E'] / metrics['#S'] if metrics['#S'] > 0 else 0
# metrics['I'] = metrics['ADDc'] * metrics['#Ev']

# # • #E: Amount of Episodes in the Tale.
# # • #S: Amount of sentences in a Tale.
# # • #C: Amount of clauses in a tale.
# # • #W: Amount of words in a tale.
# # • AEL: Average Episode length.
# # • ASL: Average Sentence length.
# # • ACL: Average clause length.
# # • ADDc: Average Dependency Distance per clause.
# # • ADDs: Average Dependency Distance per sentence.
# # • ADDe: Average Dependency Distance per episode.
# # • #Ev: Amount of Propp functions (episodes) per sentence.
# # • I: ADD * #P.
# # • CLI: Coleman-Liau Index per Tal

# # Coleman-Liau Index calculation
# # Assuming you have the number of letters and characters available
# L = (metrics['num_letters'] / metrics['#W']) * 100 if metrics['#W'] > 0 else 0
# S = (metrics['#S'] / metrics['#W']) * 100 if metrics['#W'] > 0 else 0
# metrics['CLI'] = 0.0588 * L - 0.296 * S - 15.8