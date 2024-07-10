import TaleScraping as ts
import Functions as func
import nltk
from nltk import ngrams, FreqDist
import os
from collections import defaultdict
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.parse import DependencyGraph
import csv



def process_tale(tale, lang):
    episodes = tale.split('\n\n')[1:]
    sentences = [sent for episode in episodes for sent in sent_tokenize(episode)]
    words = [word for sentence in sentences for word in word_tokenize(sentence)]
    num_clauses = 0
    sentences_with_info = []
    
    for sentence in sentences:
        num_clauses += func.count_clauses_u(sentence, lang)
        
    for episode_id, episode in enumerate(episodes, 1):
        episode_sentences = sent_tokenize(episode)
        try:
            sentence_weight = 1 / (len(episode_sentences))
        except Exception as e:
            sentence_weight = 1 / (len(episode_sentences) + 1)
        for sentence in episode_sentences:
            sentences_with_info.append({
                'sentence': sentence,
                'episode_id': episode_id,
                'num_clauses' : func.count_clauses_u(sentence, lang),
                'num_words': len(word_tokenize(sentence)),
                'sentence_weight': sentence_weight
            })
            words.extend(word_tokenize(sentence))
            num_clauses += func.count_clauses(sentence)
    
    metrics = {
        'num_episodes': len(episodes) - 1,
        'num_sentences': len(sentences),
        'num_clauses': num_clauses,
        'num_words': len(words),
        'avg_episode_length': len(words) / len(episodes) if episodes else 0,
        'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
        'avg_clause_length': len(words) / num_clauses if num_clauses else 0,
    }
    

    total_dep_distance = func.process_text_DD(tale)[0][2] 
    metrics['avg_dep_distance_clause'] = total_dep_distance / num_clauses if num_clauses else 0
    metrics['avg_dep_distance_sentence'] = total_dep_distance / len(sentences) if sentences else 0
    metrics['avg_dep_distance_episode'] = total_dep_distance / len(episodes) if episodes else 0
    metrics['Average_Eventfulness'] = len(episodes) / len(sentences) if sentences else 0
    metrics['Average_I'] = metrics['avg_dep_distance_sentence'] * metrics['Average_Eventfulness']
    metrics['CLI'] = func.coleman_liau_index(tale)
    
    sentence_metrics = []
    # for sentence in sentences:
    #     dep_distance = func.simple_dependency_distance(sentence)
    #     eventfulness = func.eventfulness(sentence)
    #     sentence_metrics.append({
    #         'sentence': sentence,
    #         'dep_distance': dep_distance,
    #         'Eventfulness': eventfulness,
    #         'I': dep_distance * eventfulness
    #     })
    
    for sentence_info in sentences_with_info:
        dep_distance = func.simple_dependency_distance(sentence_info['sentence'])
        #eventfulness = func.eventfulness(sentence_info['sentence'])
        informativeness = (dep_distance * float(sentence_info['sentence_weight']))
        sentence_metrics.append({
            'episode_id': sentence_info['episode_id'],
            'sentence': sentence_info['sentence'],
            'num_words': sentence_info['num_words'],
            'num_clauses': sentence_info['num_clauses'],
            'dep_distance': dep_distance,
            'Eventfulness': sentence_info['sentence_weight'],
            'I': informativeness

        })
    
    return metrics, sentence_metrics

def process_file(file_path, lang):
    with open(file_path, 'r', encoding='utf-8') as file:
        tales = file.read().split('--------------------------------------------------\n\n')
    
    all_tale_metrics = []
    all_sentence_metrics = []
    
    for i, tale in enumerate(tales, 1):
        if tale.strip():
            tale_metrics, sentence_metrics = process_tale(tale, lang)
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
                           'Average_Eventfulness', 'Average_I', 'CLI']
        writer = csv.DictWriter(csvfile, fieldnames=tale_fieldnames)
        writer.writeheader()
        for tm in tale_metrics:
            writer.writerow(tm)
        
        # Write a separator
        writer.writerow({field: '' for field in tale_fieldnames})
        
        # Write sentence metrics
        sentence_fieldnames = ['tale_id', 'episode_id', 'sentence', 'num_words', 'num_clauses', 'dep_distance', 'Eventfulness', 'I']
        writer = csv.DictWriter(csvfile, fieldnames=sentence_fieldnames)
        writer.writeheader()
        for sm in sentence_metrics:
            writer.writerow(sm)

# Usage
#input_file = 'grimm_tales_en.txt'
#output_file = 'output_metrics_2.csv'

tale_metrics, sentence_metrics = process_file('grimm_tales_en.txt', lang = 'en')
export_to_csv(tale_metrics, sentence_metrics, 'grimm_unified_metrics_en.csv')

# tale_metrics, sentence_metrics = process_file('grimm_tales_fi.txt', lang = 'fi')
# export_to_csv(tale_metrics, sentence_metrics, 'grimm_unified_metrics_fi.csv')

tale_metrics, sentence_metrics = process_file('grimm_tales_de.txt', lang = 'de')
export_to_csv(tale_metrics, sentence_metrics, 'grimm_unified_metrics_de.csv')

tale_metrics, sentence_metrics = process_file('grimm_tales_es.txt', lang = 'es')
export_to_csv(tale_metrics, sentence_metrics, 'grimm_unified_metrics_es.csv')

tale_metrics, sentence_metrics = process_file('grimm_tales_it.txt', lang = 'it')
export_to_csv(tale_metrics, sentence_metrics, 'grimm_unified_metrics_it.csv')



