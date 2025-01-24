import requests
from bs4 import BeautifulSoup
import os
import re
import random
import argparse

base_url = "https://www.grimmstories.com"
languages = {
    'en': {
        'index': '/en/grimm_fairy-tales/index?page=1',
        'favorites': '/en/grimm_fairy-tales/favorites'
    },
    'de': {
        'index': '/de/grimm_maerchen/index?page=1',
        'favorites': '/de/grimm_maerchen/favorites'
    },
    'it': {
        'index': '/it/grimm_fiabe/index?page=1',
        'favorites': '/it/grimm_fiabe/favorites'
    },
    'es': {
        'index': '/es/grimm_cuentos/index?page=1',
        'favorites': '/es/grimm_cuentos/favorites'
    }
}

def get_all_tales(url):
    """Get all tales from a page, including pagination"""
    all_tales = []
    current_url = url
    
    while current_url:
        response = requests.get(current_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        tales = soup.find_all('div', class_='list_titles')
        
        page_tales = [(tale.next.find('h3').text, tale.next['href']) for tale in tales[:-1]]
        all_tales.extend(page_tales)
        
        next_page = soup.find('a', class_='next_page')
        current_url = base_url + next_page['href'] if next_page else None
    
    return all_tales

def get_favorite_tales(url):
    """Get list of favorite tales"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    tales = soup.find_all('div', class_='list_titles')
    return [tale.next.find('h3').text for tale in tales[:-1]]

def scrape_tale(url):
    """Scrape individual tale content"""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h2', class_="title").text.strip()
    content_div = soup.find('div', itemprop="text")

    for br in content_div.find_all("br"):
        br.replace_with("\n")

    episodes = []
    for p in content_div.find_all(['p', 'div']):
        text = p.get_text(strip=True)
        if text:
            episodes.append(text)

    merged_episodes = []
    for episode in episodes:
        if episode.startswith('"') and merged_episodes:
            merged_episodes[-1] += " " + episode
        else:
            merged_episodes.append(episode)

    content = "\n\n".join(merged_episodes)
    return title, content

def save_tales(tales, lang):
    """Save tales to file"""
    filename = f"ugly_{lang}_grimm_tales.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for i, (title, content) in enumerate(tales, 1):
            f.write(f"Tale {i}: {title}\n\n")
            f.write(f"{content}\n\n")
            f.write("-" * 50 + "\n\n")

def scrape_ugly_tales(selected_languages):
    """Scrape only the ugly (non-favorite) tales for selected languages"""
    for lang in selected_languages:
        if lang not in languages:
            print(f"Skipping invalid language: {lang}")
            continue
            
        print(f"\nProcessing {lang.upper()} fairy tales...")
        
        all_tales = get_all_tales(base_url + languages[lang]['index'])
        favorite_tales = get_favorite_tales(base_url + languages[lang]['favorites'])
        
        non_favorite_tales = [(title, url) for title, url in all_tales 
                            if title not in favorite_tales]
        
        selected_tales = random.sample(non_favorite_tales, min(20, len(non_favorite_tales)))
        
        tales_content = []
        for title, href in selected_tales:
            print(f"Scraping: {title}")
            try:
                full_title, content = scrape_tale(href)
                tales_content.append((full_title, content))
            except Exception as e:
                print(f"Error scraping {title}: {str(e)}")
        
        print(f"Saving {lang.upper()} fairy tales...")
        save_tales(tales_content, lang)
        
        print(f"Finished processing {lang.upper()} fairy tales.")

def main():
    # parser = argparse.ArgumentParser(description='Scrape Grimm fairy tales in different languages')
    # parser.add_argument('--languages', nargs='+', choices=['en', 'de', 'it', 'es'], 
    #                   help='Languages to scrape (en, de, it, es)')
    # parser.add_argument('--all', action='store_true', 
    #                   help='Scrape all languages')
    
    # args = parser.parse_args()
    
    # if not args.languages and not args.all:
    #     parser.error("Please specify either --languages or --all")
    
    #selected_languages = list(languages.keys()) if args.all else args.languages
    selected_languages = ['en', 'de', 'it', 'es']
    scrape_ugly_tales(selected_languages)

if __name__ == "__main__":
    main()