import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

# URL of the sitemap
sitemap_url = "https://www.hs-coburg.de/person-sitemap.xml"

# Folder to save images
output_dir = "person_images"
os.makedirs(output_dir, exist_ok=True)

def fetch_and_save_image(url):
    try:
        # Fetch the person's page
        person_response = requests.get(url)
        if person_response.status_code != 200:
            print(f"Failed to fetch {url}")
            return

        person_soup = BeautifulSoup(person_response.text, 'html.parser')

        # Extract name
        name_tag = person_soup.find('h4')
        name = name_tag.text.strip() if name_tag else "Unknown"

        # Extract profile image URL
        img_tag = person_soup.find('img', class_='wp-image-0')
        if img_tag:
            img_url = img_tag.get('data-orig-src') or img_tag.get('src')
            if img_url and not img_url.startswith("http"):
                img_url = f"https://www.hs-coburg.de{img_url}"

            # Download and save
            img_response = requests.get(img_url)
            if img_response.status_code == 200:
                file_name = f"{name.replace('  ', ' ').replace('/', '_')}.jpg"
                file_path = os.path.join(output_dir, file_name)
                with open(file_path, 'wb') as img_file:
                    img_file.write(img_response.content)
                print(f"Saved image for {name} as {file_name}")
            else:
                print(f"Failed to download image for {name} from {img_url}")
        else:
            print(f"No image found for {name} on {url}")

    except Exception as e:
        print(f"Error processing {url}: {e}")

def main():
    response = requests.get(sitemap_url)
    if response.status_code == 200:
        sitemap_soup = BeautifulSoup(response.text, 'html.parser')
        urls = [loc.text for loc in sitemap_soup.find_all('loc') if "/personen/" in loc.text and "/en/" not in loc.text]
    else:
        print(f"Failed to fetch the sitemap. Status code: {response.status_code}")
        return

    # Use ThreadPoolExecutor to process URLs in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(fetch_and_save_image, urls)

if __name__ == "__main__":
    main()
