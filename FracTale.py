import TaleScraping as ts
import Functions as func
import nltk
from nltk import ngrams, FreqDist
import os
from collections import defaultdict
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.parse import DependencyGraph
import csv


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


file_path = 'grimm_tales_en.txt'
func.process_file_coleman_liau_index(file_path)



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

def process_text(text):
    with open(text, 'r', encoding='iso-8859-1') as file:
        content = file.read()
    sentences = sent_tokenize(content)
    results = []
    
    for i, sentence in enumerate(sentences, 1):
        graph = simple_dependency_parse(sentence)
        distance = calculate_dependency_distance(graph)
        results.append((i, sentence, distance))
    
    return results


results = process_text(file_path)

for i, sentence, distance in results:
    print(f"Sentence {i}: '{sentence}'")
    print(f"Dependency Distance: {distance}\n")




def simple_dependency_distance(sentence):
    words = word_tokenize(sentence)
    total_distance = sum(i for i in range(len(words)))
    return total_distance / len(words) if words else 0

def process_tale(tale):
    episodes = tale.split('\n\n')
    sentences = [sent for episode in episodes for sent in sent_tokenize(episode)]
    words = [word for sentence in sentences for word in word_tokenize(sentence)]
    
    # Simplified clause counting (assuming one clause per sentence)
    num_clauses = len(sentences)
    
    metrics = {
        'num_episodes': len(episodes),
        'num_sentences': len(sentences),
        'num_clauses': num_clauses,
        'num_words': len(words),
        'avg_episode_length': len(words) / len(episodes) if episodes else 0,
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'avg_clause_length': len(words) / num_clauses if num_clauses else 0,
    }
    
    total_dep_distance = sum(simple_dependency_distance(sent) for sent in sentences)
    metrics['avg_dep_distance_clause'] = total_dep_distance / num_clauses if num_clauses else 0
    metrics['avg_dep_distance_sentence'] = total_dep_distance / len(sentences) if sentences else 0
    metrics['avg_dep_distance_episode'] = total_dep_distance / len(episodes) if episodes else 0
    metrics['episodes_per_sentence'] = len(episodes) / len(sentences) if sentences else 0
    metrics['I'] = metrics['avg_dep_distance_sentence'] * metrics['episodes_per_sentence']
    
    sentence_metrics = []
    for sentence in sentences:
        dep_distance = simple_dependency_distance(sentence)
        sentence_metrics.append({
            'sentence': sentence,
            'dep_distance': dep_distance,
            'I': dep_distance * metrics['episodes_per_sentence']
        })
    
    return metrics, sentence_metrics

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        tales = file.read().split('--------------------------------------------------\n\n')
    
    all_tale_metrics = []
    all_sentence_metrics = []
    
    for i, tale in enumerate(tales, 1):
        if tale.strip():
            tale_metrics, sentence_metrics = process_tale(tale)
            tale_metrics['tale_id'] = i
            all_tale_metrics.append(tale_metrics)
            for sm in sentence_metrics:
                sm['tale_id'] = i
            all_sentence_metrics.extend(sentence_metrics)
    
    return all_tale_metrics, all_sentence_metrics

def export_to_csv(tale_metrics, sentence_metrics, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Write tale metrics
        tale_fieldnames = ['tale_id', 'num_episodes', 'num_sentences', 'num_clauses', 'num_words',
                           'avg_episode_length', 'avg_sentence_length', 'avg_clause_length',
                           'avg_dep_distance_clause', 'avg_dep_distance_sentence', 'avg_dep_distance_episode',
                           'episodes_per_sentence', 'I']
        writer = csv.DictWriter(csvfile, fieldnames=tale_fieldnames)
        writer.writeheader()
        for tm in tale_metrics:
            writer.writerow(tm)
        
        # Write a separator
        writer.writerow({field: '' for field in tale_fieldnames})
        
        # Write sentence metrics
        sentence_fieldnames = ['tale_id', 'sentence', 'dep_distance', 'I']
        writer = csv.DictWriter(csvfile, fieldnames=sentence_fieldnames)
        writer.writeheader()
        for sm in sentence_metrics:
            writer.writerow(sm)

# Usage
input_file = 'grimm_tales_en.txt'
output_file = 'output_metrics.csv'

tale_metrics, sentence_metrics = process_file(input_file)
export_to_csv(tale_metrics, sentence_metrics, output_file)