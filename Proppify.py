import os
import json
import numpy as np
from langdetect import detect
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import re
from collections import Counter


print("Loading word vectors...")
word_vectors = api.load("word2vec-google-news-300")

# Load the extended Propp functions
print("Loading Propp functions...")
with open('propp_functions.json', 'r', encoding='utf-8') as f:
    extended_propp_functions = json.load(f)

def get_keyword_embeddings(keywords, word_vectors):
    embeddings = []
    for keyword in keywords:
        if keyword in word_vectors:
            embeddings.append(word_vectors[keyword])
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(word_vectors.vector_size)

print("Creating Propp function embeddings...")
propp_function_embeddings = {
    func: {lang: get_keyword_embeddings(keywords, word_vectors) 
           for lang, keywords in lang_keywords.items()}
    for func, lang_keywords in extended_propp_functions.items()
}

def preprocess_text(text):
    
    text = re.sub(r'[^\w\s]', '', text.lower())
    return [word for word in text.split() if word not in STOPWORDS and len(word) > 2]

def train_lda_model(tales, num_topics=20):
    all_episodes = [episode for tale in tales for episode in tale.split('\n\n')]
    preprocessed_episodes = [preprocess_text(episode) for episode in all_episodes]
    
    dictionary = corpora.Dictionary(preprocessed_episodes)
    dictionary.filter_extremes(no_below=5, no_above=0.5)  # Filter out rare and common words
    corpus = [dictionary.doc2bow(text) for text in preprocessed_episodes]
    
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)
    return lda_model, dictionary

def get_episode_topics(episode, lda_model, dictionary):
    bow = dictionary.doc2bow(preprocess_text(episode))
    return lda_model[bow]

def get_episode_embedding(episode_text, word_vectors):
    words = episode_text.lower().split()
    word_embeddings = [word_vectors[word] for word in words if word in word_vectors]
    return np.mean(word_embeddings, axis=0) if word_embeddings else np.zeros(word_vectors.vector_size)

def annotate_episode(episode_text, lang, lda_model, dictionary):
    episode_embedding = get_episode_embedding(episode_text, word_vectors)
    episode_topics = get_episode_topics(episode_text, lda_model, dictionary)
    
    max_similarity = -1
    best_function = None
    
    for func, lang_embeddings in propp_function_embeddings.items():
        
        embedding_similarity = cosine_similarity([episode_embedding], [lang_embeddings[lang]])[0][0]
        
        
        topic_words = [word for topic_id, _ in episode_topics for word, _ in lda_model.get_topic_terms(topic_id, topn=10)]
        topic_similarity = len(set(topic_words) & set(extended_propp_functions[func][lang])) / len(set(extended_propp_functions[func][lang]))
        
        
        keywords = set(extended_propp_functions[func][lang])
        episode_words = set(preprocess_text(episode_text))
        keyword_presence = len(keywords & episode_words) / len(keywords)
        
        
        combined_similarity = 0.5 * embedding_similarity + 0.3 * topic_similarity + 0.2 * keyword_presence
        
        if combined_similarity > max_similarity:
            max_similarity = combined_similarity
            best_function = func
    
    return best_function


def annotate_tale(tale_text, lda_model, dictionary):
    lang = detect(tale_text)
    if lang not in ['en', 'de', 'it', 'es']:
        lang = 'en'
    
    episodes = tale_text.split('\n\n')
    annotations = [annotate_episode(episode.strip(), lang, lda_model, dictionary) for episode in episodes if episode.strip()]
    
    
    annotation_counts = Counter(annotations)
    most_common = annotation_counts.most_common(1)[0][0]
    
    for i in range(1, len(annotations) - 1):
        if annotations[i] == annotations[i-1] == annotations[i+1] == most_common:
            
            episode = episodes[i].strip()
            second_best = annotate_episode_second_best(episode, lang, lda_model, dictionary, exclude=annotations[i])
            if second_best:
                annotations[i] = second_best
    
    return annotations

def annotate_episode_second_best(episode_text, lang, lda_model, dictionary, exclude):
    
    pass

def process_tales_file(input_file, output_file):
    print(f"Processing tales from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as infile:
        tales = infile.read().split('--------------------------------------------------\n\n')
    
    print("Training LDA model...")
    lda_model, dictionary = train_lda_model(tales)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i, tale in enumerate(tales):
            if tale.strip():
                print(f"Annotating tale {i+1}")
                annotations = annotate_tale(tale, lda_model, dictionary)
                outfile.write(f"Tale {i+1}:\n")
                outfile.write(tale + "\n\n")
                outfile.write("Annotations:\n")
                for j, (episode, annotation) in enumerate(zip(tale.split('\n\n'), annotations)):
                    outfile.write(f"Episode {j+1}: {annotation}\n")
                outfile.write("\n")

    print(f"Annotations written to {output_file}")

# Usage
input_file = 'C:/Users/Palma/Desktop//PHD/FracTale/grimm_tales_en.txt'
output_file = 'annotated_fairy_tales_with_improved_lda.txt'
process_tales_file(input_file, output_file)