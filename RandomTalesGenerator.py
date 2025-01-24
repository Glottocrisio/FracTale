import random
import string

def generate_title():
    adjectives = ['Mysterious', 'Ancient', 'Magical', 'Hidden', 'Secret', 'Lost', 'Enchanted', 'Golden', 
                  'Silver', 'Crystal', 'Forgotten', 'Celestial', 'Mystic', 'Sacred', 'Legendary']
    nouns = ['Forest', 'Kingdom', 'Mirror', 'Crown', 'Sword', 'Book', 'Ring', 'Castle', 
             'Mountain', 'River', 'Stone', 'Dragon', 'Quest', 'Journey', 'Legend']
    return f"The {random.choice(adjectives)} {random.choice(nouns)}"

def generate_word(min_letters=1, max_letters=9):
    length = random.randint(min_letters, max_letters)
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def generate_clause(min_words=1, max_words=9):
    num_words = random.randint(min_words, max_words)
    return ' '.join(generate_word() for _ in range(num_words))

def generate_sentence(min_clauses=1, max_clauses=6):
    num_clauses = random.randint(min_clauses, max_clauses)
    return '; '.join(generate_clause() for _ in range(num_clauses))

def generate_episode(min_sentences=2, max_sentences=10):
    num_sentences = random.randint(min_sentences, max_sentences)
    return '. '.join(generate_sentence() for _ in range(num_sentences)) + '.'

def generate_story():
    num_episodes = random.randint(3, 12)
    story = []
    for i in range(num_episodes):
        episode = f"\n" + generate_episode()
        story.append(episode)
    return '\n'.join(story)

with open('random_tales.txt', 'w', encoding='utf-8') as f:
    for i in range(20):
        title = generate_title()
        f.write(f"Tale {i+1}: {title}\n\n")
        story = generate_story()
        f.write(story)
        f.write('\n\n--------------------------------------------------\n\n')

print("20 tales have been generated and saved to 'random_tales.txt'")
