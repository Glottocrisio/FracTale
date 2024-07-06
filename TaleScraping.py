import requests
from bs4 import BeautifulSoup
import os

# Define the base URL and languages
base_url = "https://www.grimmstories.com"
languages = {
    'en': '/en/grimm_fairy-tales/favorites',
    'de': '/de/grimm_maerchen/favorites',
    'it': '/it/grimm_fiabe/favorites',
    'es': '/es/grimm_cuentos/favorites'
    #'fi' : '/fi/grimm_sadut/favorites'
}

# def get_fairy_tales(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')
#     tales = soup.findall('div', class_='list_titles')
#     return [(tale.text.strip().replace("Read the story →",""), tale['href']) for tale in tales.find('div', class_='list_titles')]

def get_fairy_tales(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    tales = soup.find_all('div', class_='list_titles')
    return [(tale.next.find('h3').text, tale.next['href']) for tale in tales[:-1]]

def scrape_tale(url, lang):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h2', class_="title").text.strip()
    content = []
    i = 0
    while i < int(len(soup.find('div', itemprop="text").next_element.previous_element.contents)):
        content.append(str(soup.find('div', itemprop="text").contents[i].next_element) + "\n")
        i += 1
    #content = soup.find('div', itemprop="text").text.strip() #  soup.find('div', itemprop="text").next_element.next
    content = "\n".join(content)
    return title, content

def save_tales(tales, lang):
    filename = f"grimm_tales_{lang}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        for i, (title, content) in enumerate(tales, 1):
            f.write(f"Tale {i}: {title}\n\n")
            f.write(f"{content}\n\n")
            f.write("-" * 50 + "\n\n")  # Separator between tales

def main():
    for lang, path in languages.items():
        print(f"Scraping {lang.upper()} fairy tales...")
        tale_list = get_fairy_tales(base_url + path)
        tales = []
        
        for title in tale_list:
            # if title == "Cinderella":
            #     title = "aschenputtel"
            #url = base_url + path + "//" + title
            print(f"Scraping: {title}")
            full_title, content = scrape_tale(title[1], lang)
            tales.append((full_title, content))
        
        print(f"Saving {lang.upper()} fairy tales...")
        save_tales(tales, lang)
        
        print(f"Finished processing {lang.upper()} fairy tales.\n")

if __name__ == "__main__":
    main()
