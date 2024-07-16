import requests
from bs4 import BeautifulSoup
import os
import re

base_url = "https://www.grimmstories.com"
languages = {
    'en': '/en/grimm_fairy-tales/favorites',
    'de': '/de/grimm_maerchen/favorites',
    'it': '/it/grimm_fiabe/favorites',
    'es': '/es/grimm_cuentos/favorites'
}

def get_fairy_tales(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    tales = soup.find_all('div', class_='list_titles')
    return [(tale.next.find('h3').text, tale.next['href']) for tale in tales[:-1]]

def scrape_tale(url, lang):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h2', class_="title").text.strip()
    content_div = soup.find('div', itemprop="text")

    # Replace <br> tags with newlines
    for br in content_div.find_all("br"):
        br.replace_with("\n")

    # Process each paragraph/episode
    episodes = []
    for p in content_div.find_all(['p', 'div']):
        text = p.get_text(strip=True)
        if text:
            episodes.append(text)

    # Merge episodes starting with quotes
    merged_episodes = []
    for episode in episodes:
        if episode.startswith('"') and merged_episodes:
            merged_episodes[-1] += episode
        else:
            merged_episodes.append(episode)

    content = "\n\n".join(merged_episodes)

    return title, content

def save_tales(tales, lang):
    filename = f"grimm_tales_{lang}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for i, (title, content) in enumerate(tales, 1):
            f.write(f"Tale {i}: {title}\n\n")
            f.write(f"{content}\n\n")
            f.write("-" * 50 + "\n\n")  


tales=[f'grimm_tales_{lang}.txt' for lang in ['de', 'es', 'it']]

for tale in tales:
# Read the truncated text
    with open(tale, 'r', encoding='utf-8') as file:
        content = file.read()

    # Pattern to match titles
    titles = re.findall(r'Tale \d+: (.*?)\n', content)

    # Display the extracted titles
    print(titles)



def main():
    for lang, path in languages.items():
        print(f"Scraping {lang.upper()} fairy tales...")
        tale_list = get_fairy_tales(base_url + path)
        tales = []
        
        for title, href in tale_list:
            print(f"Scraping: {title}")
            full_title, content = scrape_tale(href, lang)
            tales.append((full_title, content))
        
        print(f"Saving {lang.upper()} fairy tales...")
        save_tales(tales, lang)
        
        print(f"Finished processing {lang.upper()} fairy tales.\n")

if __name__ == "__main__":
    main()


