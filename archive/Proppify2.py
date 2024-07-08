import os
import json
import numpy as np
from langdetect import detect
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained word vectors
print("Loading word vectors...")
word_vectors = api.load("word2vec-google-news-300")

# Load the extended Propp functions (you should replace this with your actual extended dictionary)
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

def annotate_tale(tale_text, threshold=0.3):
    # Detect language
    lang = detect(tale_text)
    
    # Default to English if language not in our dictionary
    if lang not in ['en', 'de', 'it', 'es']:
        lang = 'en'
    
    # Split the tale into words
    words = tale_text.lower().split()
    
    # Get embeddings for each word
    word_embeddings = [word_vectors[word] for word in words if word in word_vectors]
    
    if not word_embeddings:
        return []  # Return empty list if no words have embeddings
    
    # Calculate similarity between each word and each Propp function
    annotations = []
    for i, word_embedding in enumerate(word_embeddings):
        for func, lang_embeddings in propp_function_embeddings.items():
            similarity = cosine_similarity([word_embedding], [lang_embeddings[lang]])[0][0]
            if similarity > threshold:
                annotations.append((i, func, similarity))
    
    # Sort annotations by similarity and remove duplicates
    annotations.sort(key=lambda x: x[2], reverse=True)
    unique_annotations = []
    used_indices = set()
    for index, func, sim in annotations:
        if index not in used_indices:
            unique_annotations.append(func)
            used_indices.add(index)
    
    return unique_annotations

def process_tales_file(input_file, output_file):
    print(f"Processing tales from {input_file}")
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        tales = infile.read().split('--------------------------------------------------\n\n')  # Assuming tales are separated by blank lines
        
        for i, tale in enumerate(tales):
            if tale.strip():  # Skip empty tales
                print(f"Annotating tale {i+1}")
                annotations = annotate_tale(tale)
                outfile.write(f"Tale {i+1}:\n")
                outfile.write(tale + "\n")
                outfile.write("Annotations: " + ", ".join(annotations) + "\n\n")

    print(f"Annotations written to {output_file}")

# Usage
input_file = 'C:/Users/Palma/Desktop//PHD/FracTale/grimm_tales_en.txt'
output_file = 'annotated_fairy_tales.txt'
process_tales_file(input_file, output_file)
