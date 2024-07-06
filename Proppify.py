import os
import re
from collections import defaultdict
from langdetect import detect

propp_functions = {
    'Absentation': {
        'en': ['leave', 'depart', 'go away', 'absent'],
        'de': ['verlassen', 'abreisen', 'weggehen', 'abwesend'],
        'it': ['partire', 'andarsene', 'allontanarsi', 'assente'],
        'es': ['partir', 'irse', 'alejarse', 'ausente']
    },
    'Interdiction': {
        'en': ['forbid', 'warn', 'not allow', 'prohibit'],
        'de': ['verbieten', 'warnen', 'nicht erlauben', 'untersagen'],
        'it': ['vietare', 'avvertire', 'non permettere', 'proibire'],
        'es': ['prohibir', 'advertir', 'no permitir', 'vedar']
    },
    'Violation': {
        'en': ['disobey', 'ignore', 'break rule', 'violate'],
        'de': ['missachten', '\u00fcbertreten', 'ignorieren', 'verletzen'],
        'it': ['disobbedire', 'ignorare', 'infrangere', 'violare'],
        'es': ['desobedecer', 'ignorar', 'infringir', 'violar']
    },
    'Reconnaissance': {
        'en': ['spy', 'investigate', 'inquire', 'reconnaissance'],
        'de': ['ausspionieren', 'erkunden', 'nachforschen', 'auskundschaften'],
        'it': ['spiare', 'investigare', 'indagare', 'ricognizione'],
        'es': ['espiar', 'investigar', 'indagar', 'reconocimiento']
    },
    'Delivery': {
        'en': ['reveal', 'disclose', 'deliver information'],
        'de': ['verraten', 'preisgeben', 'ausliefern'],
        'it': ['rivelare', 'divulgare', 'consegnare informazioni'],
        'es': ['revelar', 'divulgar', 'entregar informaci\u00f3n']
    },
    'Trickery': {
        'en': ['deceive', 'trick', 'fool', 'mislead'],
        'de': ['t\u00e4uschen', 'betr\u00fcgen', 'irref\u00fchren', '\u00fcberlisten'],
        'it': ['ingannare', 'raggirare', 'imbrogliare', 'fuorviare'],
        'es': ['enga\u00f1ar', 'trucar', 'embaucar', 'despistar']
    },
    'Complicity': {
        'en': ['comply', 'help unwittingly', 'be deceived'],
        'de': ['nachgeben', 'unwissentlich helfen', 'get\u00e4uscht werden'],
        'it': ['assecondare', 'aiutare inconsapevolmente', 'essere ingannato'],
        'es': ['cumplir', 'ayudar sin saberlo', 'ser enga\u00f1ado']
    },
    'Villainy or Lacking': {
        'en': ['harm', 'injure', 'lack', 'need'],
        'de': ['schaden', 'verletzen', 'fehlen', 'mangeln'],
        'it': ['danneggiare', 'ferire', 'mancare', 'necessitare'],
        'es': ['da\u00f1ar', 'herir', 'carecer', 'necesitar']
    },
    'Mediation': {
        'en': ['request', 'order', 'send', 'mediate'],
        'de': ['bitten', 'auffordern', 'senden', 'vermitteln'],
        'it': ['richiedere', 'ordinare', 'inviare', 'mediare'],
        'es': ['solicitar', 'ordenar', 'enviar', 'mediar']
    },
    'Beginning Counteraction': {
        'en': ['decide', 'resolve', 'plan', 'counteract'],
        'de': ['entscheiden', 'beschlie\u00dfen', 'planen', 'entgegenwirken'],
        'it': ['decidere', 'risolvere', 'pianificare', 'contrastare'],
        'es': ['decidir', 'resolver', 'planear', 'contrarrestar']
    },
    'Departure': {
        'en': ['set out', 'leave', 'depart', 'go on a journey'],
        'de': ['aufbrechen', 'losziehen', 'abreisen', 'sich auf den Weg machen'],
        'it': ['partire', 'lasciare', 'andarsene', 'mettersi in viaggio'],
        'es': ['partir', 'salir', 'marcharse', 'emprender un viaje']
    },
    'First Function of the Donor': {
        'en': ['test', 'interrogate', 'attack', 'examine'],
        'de': ['pr\u00fcfen', 'befragen', 'angreifen', 'untersuchen'],
        'it': ['testare', 'interrogare', 'attaccare', 'esaminare'],
        'es': ['probar', 'interrogar', 'atacar', 'examinar']
    },
    'Hero\'s Reaction': {
        'en': ['react', 'respond', 'withstand', 'endure'],
        'de': ['reagieren', 'antworten', 'standhalten', 'ertragen'],
        'it': ['reagire', 'rispondere', 'resistere', 'sopportare'],
        'es': ['reaccionar', 'responder', 'resistir', 'soportar']
    },
    'Receipt of a Magical Agent': {
        'en': ['receive', 'find', 'purchase', 'create', 'appear', 'magic item'],
        'de': ['erhalten', 'finden', 'kaufen', 'erschaffen', 'erscheinen', 'magischer Gegenstand'],
        'it': ['ricevere', 'trovare', 'acquistare', 'creare', 'apparire', 'oggetto magico'],
        'es': ['recibir', 'encontrar', 'comprar', 'crear', 'aparecer', 'objeto m\u00e1gico']
    },
    'Guidance': {
        'en': ['guide', 'lead', 'show the way', 'direct'],
        'de': ['f\u00fchren', 'leiten', 'den Weg zeigen', 'lenken'],
        'it': ['guidare', 'condurre', 'mostrare la strada', 'dirigere'],
        'es': ['guiar', 'conducir', 'mostrar el camino', 'dirigir']
    },
    'Struggle': {
        'en': ['fight', 'battle', 'compete', 'struggle'],
        'de': ['k\u00e4mpfen', 'streiten', 'wetteifern', 'ringen'],
        'it': ['combattere', 'lottare', 'competere', 'faticare'],
        'es': ['luchar', 'batallar', 'competir', 'esforzarse']
    },
    'Branding': {
        'en': ['mark', 'brand', 'wound', 'tag'],
        'de': ['markieren', 'kennzeichnen', 'verwunden', 'etikettieren'],
        'it': ['marcare', 'marchiare', 'ferire', 'etichettare'],
        'es': ['marcar', 'etiquetar', 'herir', 'identificar']
    },
    'Victory': {
        'en': ['defeat', 'win', 'overcome', 'triumph'],
        'de': ['besiegen', 'gewinnen', '\u00fcberwinden', 'triumphieren'],
        'it': ['sconfiggere', 'vincere', 'superare', 'trionfare'],
        'es': ['derrotar', 'ganar', 'superar', 'triunfar']
    },
    'Liquidation': {
        'en': ['resolve', 'fix', 'mend', 'heal'],
        'de': ['l\u00f6sen', 'beheben', 'reparieren', 'heilen'],
        'it': ['risolvere', 'riparare', 'aggiustare', 'guarire'],
        'es': ['resolver', 'arreglar', 'reparar', 'sanar']
    },
    'Return': {
        'en': ['return', 'come back', 'homecoming'],
        'de': ['zur\u00fcckkehren', 'heimkehren', 'wiederkommen'],
        'it': ['ritornare', 'tornare', 'rientrare'],
        'es': ['regresar', 'volver', 'retornar']
    },
    'Pursuit': {
        'en': ['chase', 'pursue', 'hunt', 'track'],
        'de': ['verfolgen', 'jagen', 'nachsetzen', 'aufsp\u00fcren'],
        'it': ['inseguire', 'rincorrere', 'cacciare', 'pedinare'],
        'es': ['perseguir', 'cazar', 'rastrear', 'seguir']
    },
    'Rescue': {
        'en': ['rescue', 'save', 'escape', 'free'],
        'de': ['retten', 'befreien', 'entkommen', 'erl\u00f6sen'],
        'it': ['salvare', 'soccorrere', 'scappare', 'liberare'],
        'es': ['rescatar', 'salvar', 'escapar', 'liberar']
    },
    'Unrecognized Arrival': {
        'en': ['arrive unrecognized', 'return in disguise', 'incognito'],
        'de': ['unerkannt ankommen', 'verkleidet zur\u00fcckkehren', 'inkognito'],
        'it': ['arrivare irriconoscibile', 'tornare in incognito', 'mascherato'],
        'es': ['llegar de inc\u00f3gnito', 'volver disfrazado', 'irreconocible']
    },
    'Unfounded Claims': {
        'en': ['false claim', 'lie', 'pretend', 'impersonate'],
        'de': ['falscher Anspruch', 'l\u00fcgen', 'vorgeben', 'sich ausgeben als'],
        'it': ['pretesa infondata', 'mentire', 'fingere', 'impersonare'],
        'es': ['pretensi\u00f3n infundada', 'mentir', 'fingir', 'hacerse pasar por']
    },
    'Difficult Task': {
        'en': ['difficult task', 'challenge', 'ordeal', 'test'],
        'de': ['schwierige Aufgabe', 'Herausforderung', 'Pr\u00fcfung', 'Test'],
        'it': ['compito difficile', 'sfida', 'prova', 'test'],
        'es': ['tarea dif\u00edcil', 'desaf\u00edo', 'prueba', 'reto']
    },
    'Solution': {
        'en': ['solve', 'accomplish', 'complete', 'succeed'],
        'de': ['l\u00f6sen', 'erf\u00fcllen', 'vollenden', 'gelingen'],
        'it': ['risolvere', 'compiere', 'completare', 'riuscire'],
        'es': ['resolver', 'lograr', 'completar', 'tener \u00e9xito']
    },
    'Recognition': {
        'en': ['recognize', 'identify', 'reveal', 'discover'],
        'de': ['erkennen', 'identifizieren', 'enth\u00fcllen', 'entdecken'],
        'it': ['riconoscere', 'identificare', 'rivelare', 'scoprire'],
        'es': ['reconocer', 'identificar', 'revelar', 'descubrir']
    },
    'Exposure': {
        'en': ['expose', 'unmask', 'reveal true nature', 'disclose'],
        'de': ['entlarven', 'demaskieren', 'wahre Natur zeigen', 'aufdecken'],
        'it': ['smascherare', 'svelare', 'rivelare la vera natura', 'esporre'],
        'es': ['desenmascarar', 'revelar', 'mostrar la verdadera naturaleza', 'exponer']
    },
    'Transfiguration': {
        'en': ['transform', 'change', 'transfigure', 'metamorphose'],
        'de': ['verwandeln', 'ver\u00e4ndern', 'umgestalten', 'transformieren'],
        'it': ['trasformare', 'cambiare', 'trasfigurare', 'metamorfosi'],
        'es': ['transformar', 'cambiar', 'transfigurar', 'metamorfosear']
    },
    'Punishment': {
        'en': ['punish', 'penalize', 'condemn', 'sentence'],
        'de': ['bestrafen', 'verurteilen', 'verdammen', 'aburteilen'],
        'it': ['punire', 'penalizzare', 'condannare', 'sentenziare'],
        'es': ['castigar', 'penalizar', 'condenar', 'sentenciar']
    },
    'Wedding': {
        'en': ['marry', 'wed', 'wedding', 'coronation', 'ascend to throne'],
        'de': ['heiraten', 'verm\u00e4hlen', 'Hochzeit', 'Kr\u00f6nung', 'Thronbesteigung'],
        'it': ['sposare', 'matrimonio', 'nozze', 'incoronazione', 'ascendere al trono'],
        'es': ['casar', 'boda', 'matrimonio', 'coronaci\u00f3n', 'ascender al trono']
    }
}


def read_file_with_fallback_encoding(file_path):
    encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'latin-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try to read the file as bytes and decode manually
    try:
        with open(file_path, 'rb') as file:
            raw = file.read()
        return raw.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return None

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

def detect_propp_functions(text, lang):
    detected_functions = defaultdict(int)
    for function, keywords in propp_functions.items():
        if lang in keywords:
            for keyword in keywords[lang]:
                count = len(re.findall(r'\b' + keyword + r'\b', text.lower()))
                detected_functions[function] += count
    return detected_functions

def analyze_tale(file_path):
    text = read_file_with_fallback_encoding(file_path)
    if text is None:
        return None, 0, 'unknown'
    
    # Detect language
    lang = detect_language(text)
    
    # Detect Propp functions
    propp_functions_detected = detect_propp_functions(text, lang)
    
    # Count episodes (assuming each detected function represents an episode)
    num_episodes = sum(propp_functions_detected.values())
    
    return propp_functions_detected, num_episodes, lang

def process_files(directory):
    results = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            try:
                propp_functions, num_episodes, lang = analyze_tale(file_path)
                if propp_functions is not None:
                    results[filename] = {
                        'propp_functions': propp_functions,
                        'num_episodes': num_episodes,
                        'language': lang
                    }
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
    return results

# Usage
directory = "C:/Users/Palma/Desktop//PHD/FracTale"
analysis_results = process_files(directory)

# Print or save results as needed
for filename, data in analysis_results.items():
    print(f"Analysis for {filename}:")
    print(f"Detected language: {data['language']}")
    print(f"Number of episodes: {data['num_episodes']}")
    print("Detected Propp functions:")
    for function, count in data['propp_functions'].items():
        if count > 0:
            print(f"  {function}: {count}")
    print("\n")



import json
import torch
import numpy as np
from transformers import BertTokenizerFast, BertModel
from langdetect import detect
import gensim.downloader as api
import torch.nn as nn

# Load pre-trained word vectors
word_vectors = api.load("word2vec-google-news-300")

# Load BERT tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
bert = BertModel.from_pretrained("bert-base-multilingual-cased")

# Define the hybrid model
class HybridProppModel(nn.Module):
    def __init__(self, bert_model, num_propp_functions, embedding_dim):
        super(HybridProppModel, self).__init__()
        self.bert = bert_model
        self.keyword_attention = nn.MultiheadAttention(embed_dim=768, num_heads=8)
        self.classifier = nn.Linear(768 + embedding_dim, num_propp_functions)
        
    def forward(self, input_ids, attention_mask, keyword_embeddings):
        bert_output = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        keyword_attention_output, _ = self.keyword_attention(bert_output, keyword_embeddings, keyword_embeddings)
        combined_output = torch.cat([bert_output[:, 0, :], keyword_attention_output.mean(dim=1)], dim=1)
        return self.classifier(combined_output)

# Load the extended Propp functions (you should replace this with your actual extended dictionary)
with open('propp_functions.json', 'r', encoding='utf-8') as f:
    extended_propp_functions = json.load(f)

# Create embeddings for Propp functions
def get_keyword_embeddings(keywords, word_vectors):
    embeddings = []
    for keyword in keywords:
        if keyword in word_vectors:
            embeddings.append(word_vectors[keyword])
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(word_vectors.vector_size)

propp_function_embeddings = {
    func: {lang: get_keyword_embeddings(keywords, word_vectors) 
           for lang, keywords in lang_keywords.items()}
    for func, lang_keywords in extended_propp_functions.items()
}

# Initialize the model (assuming you have a pre-trained model)
model = HybridProppModel(bert, len(extended_propp_functions), 300)
model.load_state_dict(torch.load('hybrid_propp_model.pth'))
model.eval()

def annotate_tale(tale_text, model, tokenizer):
    # Detect language
    lang = detect(tale_text)
    
    # Tokenize and get BERT embeddings
    inputs = tokenizer(tale_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get keyword embeddings for the detected language
    keyword_embeddings = torch.tensor([propp_function_embeddings[func][lang] 
                                       for func in extended_propp_functions]).unsqueeze(0)
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'], keyword_embeddings)
    predictions = torch.argmax(outputs, dim=1)
    
    # Convert predictions to Propp function labels
    propp_functions = list(extended_propp_functions.keys())
    annotations = [propp_functions[pred] for pred in predictions]
    
    return annotations

def process_tales_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        tales = infile.read().split('\n\n')  # Assuming tales are separated by blank lines
        
        for i, tale in enumerate(tales):
            if tale.strip():  # Skip empty tales
                annotations = annotate_tale(tale, model, tokenizer)
                outfile.write(f"Tale {i+1}:\n")
                outfile.write(tale + "\n")
                outfile.write("Annotations: " + ", ".join(annotations) + "\n\n")

# Usage
input_file = 'C:/Users/Palma/Desktop//PHD/FracTale/grimm_tales_en.txt'
output_file = 'annotated_fairy_tales.txt'
process_tales_file(input_file, output_file)