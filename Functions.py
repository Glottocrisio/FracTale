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


# for lang, path in languages.items():
#     print(f"Scraping {lang.upper()} fairy tales...")
#     tale_list = ts.get_fairy_tales(base_url + path)
#     tales = []
        
#     for title in tale_list:
#         # if title == "Cinderella":
#         #     title = "aschenputtel"
#         #url = base_url + path + "//" + title
#         print(f"Scraping: {title}")
#         full_title, content = ts.scrape_tale(title[1], lang)
#         tales.append((full_title, content))
        
#     print(f"Saving {lang.upper()} fairy tales...")
#     ts.save_tales(tales, lang)
        
#     print(f"Finished processing {lang.upper()} fairy tales.\n")


# # Usage
# input_file = 'C:/Users/Palma/Desktop//PHD/FracTale/grimm_tales_en.txt'
# output_file = 'annotated_fairy_tales_with_improved_lda.txt'
# func.process_tales_file(input_file, output_file)

def count_clauses(sentence):
    total_clauses = 0
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    verb_count = sum(1 for word, tag in pos_tags if tag.startswith('VB'))
    total_clauses += max(1, verb_count)  # Ensure at least one clause per sentence
    return total_clauses



nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def simple_dependency_parse(sentence):
    # Tokenize and POS tag the sentence
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)
    
    # Create a simple dependency graph
    graph = DependencyGraph()
    for i, (word, pos) in enumerate(pos_tags):
        graph.add_node({
            'address': i+1,
            'word': word,
            'lemma': word.lower(),
            'ctag': pos,
            'tag': pos,
            'feats': '',
            'head': None,
            'deps': {},
            'rel': None
        })
    
    # Simple rules for dependency
    root = None
    for i, (word, pos) in enumerate(pos_tags):
        if pos.startswith('VB'):  # Verb as root
            root = i + 1
            break
    if root is None and len(pos_tags) > 0:
        root = 1  # First word as root if no verb found
    
    graph.nodes[root]['rel'] = 'ROOT'
    
    for i, (word, pos) in enumerate(pos_tags):
        if i + 1 != root:
            graph.nodes[i+1]['head'] = root
            graph.nodes[root]['deps'].setdefault('dep', []).append(i+1)
    
    return graph

def calculate_dependency_distance(graph):
    total_distance = 0
    for node in graph.nodes.values():
        if node['head'] is not None:
            total_distance += abs(node['address'] - node['head'])
    return total_distance

def process_text_DD(text):
    #with open(text, 'r', encoding='iso-8859-1') as file:
        #content = file.read()
    sentences = sent_tokenize(text)
    results = []
    
    for i, sentence in enumerate(sentences, 1):
        graph = simple_dependency_parse(sentence)
        distance = calculate_dependency_distance(graph)
        results.append((i, sentence, distance))
    
    return results

with open('grimm_tales_en.txt', 'r', encoding='iso-8859-1') as file:
    content = file.read()
    tales = content.split('--------------------------------------------------\n\n')

    for i, tale in enumerate(tales, 1):
        if tale.strip():
            print(f"Tale {i}:")
            results = process_text_DD(tale)
            for i, sentence, distance in results:
                print(f"Sentence {i}: '{sentence}'")
                print(f"Dependency Distance: {distance}\n")


def simple_dependency_distance(sentence):
    graph = simple_dependency_parse(sentence)
    dd = calculate_dependency_distance(graph)
    return dd

def eventfulness(episode):
    pass