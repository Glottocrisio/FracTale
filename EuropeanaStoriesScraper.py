import requests
from bs4 import BeautifulSoup
import os
import time

def get_story_links(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    links = soup.find_all('a', class_='card-link')
    return ['https://www.europeana.eu' + link['href'] for link in links]

def get_story_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = soup.find('article', class_='story-article-container')
    if content:
        paragraphs = content.find_all('p')
        headers = content.find_all('h2', id=True)
        
        text_content = []
        for element in headers + paragraphs:
            text_content.append(element.get_text(strip=True))
        
        return '\n\n'.join(text_content)
    return ''

def save_story(content, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    base_url = "https://www.europeana.eu/de/stories?page={}"
    all_stories = []

    if not os.path.exists('stories'):
        os.makedirs('stories')

    for page in range(1, 50):  # Pages 1 to 49
        print(f"Scraping page {page}")
        page_url = base_url.format(page)
        story_links = get_story_links(page_url)

        for i, link in enumerate(story_links, 1):
            print(f"  Scraping story {i} from page {page}")
            content = get_story_content(link)
            if content:
                filename = f"stories/story_page{page}_num{i}.txt"
                save_story(content, filename)
                all_stories.append(content)

            time.sleep(1)

        time.sleep(1)

    save_story('\n\n---\n\n'.join(all_stories), 'all_stories_de.txt')

if __name__ == "__main__":
    main()
