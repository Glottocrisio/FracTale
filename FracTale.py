import TaleScraping as ts
import Functions as func
import nltk
from nltk import ngrams, FreqDist
import os
from collections import defaultdict
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.parse import DependencyGraph
import csv



def process_tale(tale):
    episodes = tale.split('\n\n')[1:]
    sentences = [sent for episode in episodes for sent in sent_tokenize(episode)]
    words = [word for sentence in sentences for word in word_tokenize(sentence)]
    num_clauses = 0
    
    # Simplified clause counting (assuming one clause per sentence)
    for sentence in sentences:
        num_clauses += func.count_clauses(sentence)
    
    metrics = {
        'num_episodes': len(episodes),
        'num_sentences': len(sentences),
        'num_clauses': num_clauses,
        'num_words': len(words),
        'avg_episode_length': len(words) / len(episodes) if episodes else 0,
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'avg_clause_length': len(words) / num_clauses if num_clauses else 0,
    }
    
    total_dep_distance = sum(func.simple_dependency_distance(sent) for sent in sentences)
    metrics['avg_dep_distance_clause'] = total_dep_distance / num_clauses if num_clauses else 0
    metrics['avg_dep_distance_sentence'] = total_dep_distance / len(sentences) if sentences else 0
    metrics['avg_dep_distance_episode'] = total_dep_distance / len(episodes) if episodes else 0
    metrics['episodes_per_sentence'] = len(episodes) / len(sentences) if sentences else 0
    metrics['I'] = metrics['avg_dep_distance_sentence'] * metrics['episodes_per_sentence']
    metrics['CLI'] = func.coleman_liau_index(tale)
    sentence_metrics = []
    for sentence in sentences:
        dep_distance = func.simple_dependency_distance(sentence)
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
                           'episodes_per_sentence', 'I', 'CLI']
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
#input_file = 'grimm_tales_en.txt'
#output_file = 'output_metrics_2.csv'

tale_metrics, sentence_metrics = process_file('grimm_tales_en.txt')
export_to_csv(tale_metrics, sentence_metrics, 'grimm_tales_metrics_en.csv')

tale_metrics, sentence_metrics = process_file('grimm_tales_fi.txt')
export_to_csv(tale_metrics, sentence_metrics, 'grimm_tales_metrics_fi.csv')

# tale_metrics, sentence_metrics = process_file('grimm_tales_de.txt')
# export_to_csv(tale_metrics, sentence_metrics, 'grimm_tales_metrics_de.csv')

# tale_metrics, sentence_metrics = process_file('grimm_tales_es.txt')
# export_to_csv(tale_metrics, sentence_metrics, 'grimm_tales_metrics_es.csv')

# tale_metrics, sentence_metrics = process_file('grimm_tales_it.txt')
# export_to_csv(tale_metrics, sentence_metrics, 'grimm_tales_metrics_it.csv')



