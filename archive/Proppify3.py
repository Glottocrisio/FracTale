import os
import json
import numpy as np
from langdetect import detect
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained word vectors
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

def get_episode_embedding(episode_text, word_vectors):
    words = episode_text.lower().split()
    word_embeddings = [word_vectors[word] for word in words if word in word_vectors]
    return np.mean(word_embeddings, axis=0) if word_embeddings else np.zeros(word_vectors.vector_size)

def annotate_episode(episode_text, lang):
    episode_embedding = get_episode_embedding(episode_text, word_vectors)
    
    max_similarity = -1
    best_function = None
    
    for func, lang_embeddings in propp_function_embeddings.items():
        similarity = cosine_similarity([episode_embedding], [lang_embeddings[lang]])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_function = func
    
    return best_function

def annotate_tale(tale_text):
    # Detect language
    lang = detect(tale_text)
    
    # Default to English if language not in our dictionary
    if lang not in ['en', 'de', 'it', 'es']:
        lang = 'en'
    
    # Split the tale into episodes
    episodes = tale_text.split('\n\n')
    
    # Annotate each episode
    annotations = [annotate_episode(episode.strip(), lang) for episode in episodes if episode.strip()]
    
    return annotations

def process_tales_file(input_file, output_file):
    print(f"Processing tales from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        tales = infile.read().split('--------------------------------------------------\n\n')  # Assuming tales are separated by two blank lines
        
        for i, tale in enumerate(tales):
            if tale.strip():  # Skip empty tales
                print(f"Annotating tale {i+1}")
                annotations = annotate_tale(tale)
                outfile.write(f"Tale {i+1}:\n")
                outfile.write(tale + "\n\n")
                outfile.write("Annotations:\n")
                for j, (episode, annotation) in enumerate(zip(tale.split('\n\n'), annotations)):
                    outfile.write(f"Episode {j+1}: {annotation}\n")
                outfile.write("\n")

    print(f"Annotations written to {output_file}")

# Usage
input_file = 'C:/Users/Palma/Desktop//PHD/FracTale/grimm_tales_en.txt'
output_file = 'annotated_fairy_tales3.txt'
process_tales_file(input_file, output_file)
