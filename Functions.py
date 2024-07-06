import nltk
from nltk import ngrams, FreqDist
import os
import spacy
from collections import defaultdict

def calculateProppNgrams(n, sequences, sign):
    
    all_ngrams = []
    for sequence in sequences:
        ngram_list = list(ngrams(sequence, n))
        all_ngrams.extend(ngram_list)

        freq_dist = FreqDist(all_ngrams)

    # Calculate the likelihood of a sequence containing the sign 'S'
    total_ngrams = len(all_ngrams)
    ngrams_with_sign = sum(freq for ngram, freq in freq_dist.items() if sign in ngram)

    # Likelihood of an n-gram containing 'S'
    likelihood = ngrams_with_sign / total_ngrams

    return total_ngrams, ngrams_with_sign, likelihood


nltk.download('punkt')
nlp = spacy.load("de_core_news_sm")

def calculate_dependency_distance(doc):
    total_distance = 0
    for token in doc:
        if token.dep_ != "ROOT":
            distance = abs(token.i - token.head.i)
            total_distance += distance
    return total_distance

def analyze_tale(tale_text):
    # Tokenize the tale into sentences
    sentences = nltk.sent_tokenize(tale_text)
    
    # Initialize metrics
    metrics = defaultdict(float)
    metrics['num_sentences'] = len(sentences)
    metrics['num_episodes'] = 0  # You'll need to implement episode detection
    
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
        metrics['total_dep_distance'] += calculate_dependency_distance(doc)
    
    # Calculate averages
    metrics['avg_word_length'] = metrics['num_letters'] / metrics['num_words']
    metrics['avg_clause_length'] = metrics['num_words'] / metrics['num_clauses']
    metrics['avg_sentence_length'] = metrics['num_words'] / metrics['num_sentences']
    metrics['avg_episode_length'] = metrics['num_sentences'] / metrics['num_episodes'] if metrics['num_episodes'] > 0 else 0
    metrics['ADD'] = metrics['total_dep_distance'] / metrics['num_clauses']
    metrics['P'] = metrics['num_episodes'] / metrics['num_sentences']
    metrics['I'] = metrics['ADD'] * metrics['P']
    metrics['W'] = metrics['avg_sentence_length']
    metrics['O'] = metrics['I'] / metrics['W']
    metrics['F'] = metrics['ADD'] + metrics['P']
    metrics['A'] = metrics['F'] / metrics['W']
    
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
directory = "path/to/your/tales/directory"
analysis_results = process_files(directory)

# Print or save results as needed
for filename, metrics in analysis_results.items():
    print(f"Analysis for {filename}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print("\n")