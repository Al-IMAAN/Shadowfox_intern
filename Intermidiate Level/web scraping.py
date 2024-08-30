import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException


def fetch_webpage_content(url):
    try:
        # Send a GET request to the website
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.text

    except RequestException as e:
        # Handle request errors (e.g., network issues, invalid URL, etc.)
        print(f"Error fetching {url}: {e}")
        return None


def extract_titles_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    # Extract all <h1> tags and return the text
    return [title.get_text(strip=True) for title in soup.find_all('h1')]


def scrape_web(url):
    html_content = fetch_webpage_content(url)

    if html_content:
        titles = extract_titles_from_html(html_content)
        if titles:
            print(f"Extracted {len(titles)} titles from {url}:")
            for title in titles:
                print(f"- {title}")
        else:
            print(f"No titles found on {url}.")
    else:
        print("Failed to retrieve or parse the webpage.")


url = 'https://www.shadowfox.in/'

scrape_web(url)
