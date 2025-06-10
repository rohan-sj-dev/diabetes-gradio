from bs4 import BeautifulSoup
import requests


blog_urls=[
  "https://health.ucdavis.edu/blog/cultivating-health/healthy-lifestyle-habits-to-help-you-prevent-or-manage-your-type-2-diabetes/2022/11?",
  "https://my.clevelandclinic.org/health/diseases/7104-diabetes",
  "https://www.medanta.org/patient-education-blog/diabetic-meal-plan-what-foods-to-eat-for-a-good-health",
  "https://www.medkart.in/blog/exercise-for-diabetes?srsltid=AfmBOooW_54C2FHL24hyNvN0mh7OPXVKVCvGulDc8BXL7IXdUmGCS1eE",
  "https://www.diabetes.org.uk/about-diabetes/symptoms/testing",
  "https://www.health.harvard.edu/blog/healthy-lifestyle-can-prevent-diabetes-and-even-reverse-it-2018090514698"

]

all_cleaned_text=[]
def scrape_single_blog_post(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        main_content_div = soup.find('article') or \
                           soup.find('div', class_='entry-content') or \
                           soup.find('div', class_='post-content') or \
                           soup.find('div', id='main-article-content')

        if main_content_div:
            return main_content_div.get_text(separator='\n', strip=True)
        else:
            return f"⚠️ Couldn't find main content for {url}"

    except requests.exceptions.RequestException as e:
        return f"🚫 Error fetching {url}: {e}"
    except Exception as e:
        return f"❌ Unexpected error for {url}: {e}"

def get_all_clean_blog_text():
    for url in blog_urls:
        content = scrape_single_blog_post(url)
        if content:
            all_cleaned_text.append(content)
    return "\n\n---\n\n".join(all_cleaned_text) if all_cleaned_text else "No content found."

