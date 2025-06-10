import gradio as gr
import requests
from bs4 import BeautifulSoup
import re
import os
import json
import pandas as pd
from datetime import datetime
from ctransformers import AutoModelForCausalLM
import concurrent.futures
from urllib.parse import urljoin
import time
import random

# Configure directories
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Base URLs to scrape for diabetes information
BASE_URLS = [
    "https://health.ucdavis.edu/blog/cultivating-health/healthy-lifestyle-habits-to-help-you-prevent-or-manage-your-type-2-diabetes/2022/11",
    "https://my.clevelandclinic.org/health/diseases/7104-diabetes",
    "https://www.medanta.org/patient-education-blog/diabetic-meal-plan-what-foods-to-eat-for-a-good-health",
    "https://www.medkart.in/blog/exercise-for-diabetes",
    "https://www.diabetes.org.uk/about-diabetes/symptoms/testing",
    # Additional sources for more comprehensive data
    "https://www.cdc.gov/diabetes/",
    "https://www.mayoclinic.org/diseases-conditions/diabetes/",
    "https://www.healthline.com/health/diabetes",
    "https://www.medicalnewstoday.com/categories/diabetes"
]

# Storage for scraped data
scraped_content = {
    "pages": {},  # Stores full page content
    "suggestions": {},  # Stores extracted suggestions by category
    "medical_terms": {}  # Stores medical terms and explanations
}

# Headers to simulate a real browser request
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.google.com/",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1"
}

def load_llm():
    """Load the LLM model for text generation"""
    model_path = "llama-2-7b-chat.ggmlv3.q4_0.bin"
    
    # Check if model exists, if not provide instructions
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        print("Please download a compatible model from: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/")
        print("For example: llama-2-7b-chat.ggmlv3.q4_0.bin")
        
        # Provide alternative instructions for different models
        print("\nAlternatively, you can use other open-source models:")
        print("1. Mistral-7B: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGML")
        print("2. Phi-2: https://huggingface.co/TheBloke/phi-2-GGML")
        
        model_path = input("Enter the path to the downloaded model file: ")
    
    try:
        # Load the model - adjust parameters based on your hardware
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            model_type="llama" if "llama" in model_path.lower() else None,  # Auto-detect model type
            max_new_tokens=1024,
            temperature=0.7,
            context_length=4096,
            gpu_layers=0  # Set to higher number if GPU available
        )
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def get_cached_page(url):
    """Get page from cache if available, otherwise download and cache it"""
    # Create a filename for the cached page
    cache_filename = os.path.join(CACHE_DIR, f"{hash(url)}.html")
    
    if os.path.exists(cache_filename):
        # Check if cache is less than 24 hours old
        file_age = time.time() - os.path.getmtime(cache_filename)
        if file_age < 86400:  # 24 hours in seconds
            with open(cache_filename, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
    
    # If not cached or too old, download the page
    try:
        response = requests.get(
            url, 
            headers=HEADERS, 
            timeout=15,
            allow_redirects=True
        )
        response.raise_for_status()
        
        # Cache the page content
        with open(cache_filename, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        return response.text
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None

def scrape_website(url):
    """
    Scrape content from the given URL using BeautifulSoup with enhanced extraction
    """
    html_content = get_cached_page(url)
    if not html_content:
        return None
        
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract website title
        title = soup.title.string if soup.title else "Unknown Title"
        
        # Domain specific extraction
        domain = extract_domain(url)
        
        # Extract the main content
        content = extract_main_content(soup, domain)
        
        # Extract links to follow
        internal_links = extract_internal_links(soup, url)
        
        # Extract suggestion blocks
        suggestions = extract_suggestion_blocks(soup, content, domain)
        
        # Extract medical terms
        medical_terms = extract_medical_terms_from_content(content, url)
        
        return {
            "title": title,
            "url": url,
            "content": content,
            "internal_links": internal_links,
            "suggestions": suggestions,
            "medical_terms": medical_terms,
            "scraped_at": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")
        return None

def extract_domain(url):
    """Extract the domain from a URL"""
    if "ucdavis.edu" in url:
        return "ucdavis"
    elif "clevelandclinic.org" in url:
        return "clevelandclinic"
    elif "medanta.org" in url:
        return "medanta"
    elif "medkart.in" in url:
        return "medkart"
    elif "diabetes.org.uk" in url:
        return "diabetesuk"
    elif "cdc.gov" in url:
        return "cdc"
    elif "mayoclinic.org" in url:
        return "mayoclinic"
    elif "healthline.com" in url:
        return "healthline"
    elif "medicalnewstoday.com" in url:
        return "medicalnewstoday"
    else:
        return "generic"

def extract_main_content(soup, domain):
    """Extract the main content using domain-specific selectors"""
    content_selectors = {
        "ucdavis": ['article.publication', 'main', '.content'],
        "clevelandclinic": ['.article-content', 'main', '#content'],
        "medanta": ['.entry-content', 'article', '.post-content'],
        "medkart": ['.blog-detail', '.post-content', 'article'],
        "diabetesuk": ['div.content', 'main', 'article'],
        "cdc": ['div.syndicate', '.content', 'main'],
        "mayoclinic": ['#main-content', 'article', '.content'],
        "healthline": ['article', '.article-body', '.content-body'],
        "medicalnewstoday": ['article', '.article-body', '.content']
    }
    
    # Try domain-specific selectors first
    selectors = content_selectors.get(domain, ['article', 'main', '.content', '#content', '.post-content'])
    
    for selector in selectors:
        content_element = soup.select_one(selector)
        if content_element and len(content_element.get_text(strip=True)) > 200:
            # Extract text content
            text = extract_formatted_text(content_element)
            if text:
                return text
    
    # Fallback to generic content extraction
    return extract_generic_content(soup)

def extract_formatted_text(element):
    """Extract text with formatting preserved"""
    result = []
    
    # Process headings
    for heading in element.select('h1, h2, h3, h4, h5, h6'):
        level = int(heading.name[1])
        prefix = '#' * level + ' '
        result.append(f"\n{prefix}{heading.get_text(strip=True)}\n")
        
    # Process paragraphs
    for p in element.select('p'):
        text = p.get_text(strip=True)
        if text:
            result.append(text + "\n")
    
    # Process lists
    for lst in element.select('ul, ol'):
        for item in lst.select('li'):
            text = item.get_text(strip=True)
            if text:
                result.append(f"• {text}")
        result.append("\n")
    
    # Process tables
    for table in element.select('table'):
        result.append("\nTable content:\n")
        for row in table.select('tr'):
            cells = [cell.get_text(strip=True) for cell in row.select('th, td')]
            if cells:
                result.append(" | ".join(cells))
        result.append("\n")
    
    return "\n".join(result)

def extract_generic_content(soup):
    """Extract content using generic approach for unknown website structures"""
    content_parts = []
    
    # Try extracting the article content
    main_content = soup.select_one('article, main, [role="main"], .article, .post, .content, #content')
    
    if not main_content:
        # If no specific container found, use body but exclude header, footer, nav
        main_content = soup.body
        
        if main_content:
            # Exclude navigation, header, footer, sidebar elements
            for elem in main_content.select('nav, header, footer, aside, .sidebar, .menu, .navigation'):
                elem.decompose()
    
    if main_content:
        # Process headings
        for heading in main_content.select('h1, h2, h3, h4, h5, h6'):
            level = int(heading.name[1])
            prefix = '#' * level + ' '
            content_parts.append(f"\n{prefix}{heading.get_text(strip=True)}\n")
        
        # Process paragraphs
        for p in main_content.select('p'):
            text = p.get_text(strip=True)
            if text and len(text) > 30:  # Avoid very short paragraphs
                content_parts.append(text + "\n")
        
        # Process lists
        for lst in main_content.select('ul, ol'):
            for item in lst.select('li'):
                text = item.get_text(strip=True)
                if text:
                    content_parts.append(f"• {text}")
            content_parts.append("\n")
    
    return "\n".join(content_parts)

def extract_internal_links(soup, base_url):
    """Extract internal links for further crawling"""
    domain = extract_domain_root(base_url)
    internal_links = []
    
    for a_tag in soup.select('a[href]'):
        href = a_tag.get('href')
        
        # Handle relative links
        full_url = urljoin(base_url, href)
        
        # Only include links within the same domain and exclude anchors
        if domain in full_url and '#' not in full_url and not full_url.endswith(('.pdf', '.jpg', '.png', '.gif')):
            internal_links.append(full_url)
    
    # Return unique links, limit to 5 to prevent excess crawling
    return list(set(internal_links))[:5]

def extract_domain_root(url):
    """Extract the root domain from a URL"""
    from urllib.parse import urlparse
    parsed_uri = urlparse(url)
    return parsed_uri.netloc

def extract_suggestion_blocks(soup, content, domain):
    """Extract specific suggestion blocks based on domain patterns"""
    suggestions = []
    
    # Look for common suggestion patterns in content
    suggestion_patterns = [
        (r'(?:recommended|suggested|advised)(?:\s+to)?(?:\s+\w+){1,3}(?:\s+to)?:\s*((?:[^.]*?(?:\.|\n|$))+)', 'recommendation'),
        (r'(?:tips|advice)(?:\s+for)?(?:\s+\w+){0,3}(?:\s+to)?:\s*((?:[^.]*?(?:\.|\n|$))+)', 'tip'),
        (r'(?:benefits|advantages)(?:\s+of)?(?:\s+\w+){0,3}(?:\s+include)?:\s*((?:[^.]*?(?:\.|\n|$))+)', 'benefit'),
        (r'(?:should|must)(?:\s+\w+){1,3}(?:\s+to)?(?:\s+\w+){0,2}:\s*((?:[^.]*?(?:\.|\n|$))+)', 'instruction'),
        (r'(?:how\s+to)(?:\s+\w+){1,4}(?:\s+\w+){0,2}:\s*((?:[^.]*?(?:\.|\n|$))+)', 'how-to'),
        (r'(?:important|essential)(?:\s+to)?(?:\s+\w+){1,3}(?:\s+to)?:\s*((?:[^.]*?(?:\.|\n|$))+)', 'important'),
    ]
    
    # Extract from content using regex
    for pattern, suggestion_type in suggestion_patterns:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for match in matches:
            suggestion_text = match.group(1).strip()
            if suggestion_text and len(suggestion_text) > 15:
                suggestions.append({
                    "type": suggestion_type,
                    "text": suggestion_text
                })
    
    # Domain-specific extraction
    if domain == "clevelandclinic":
        # Cleveland Clinic often uses bullet points for suggestions
        for list_el in soup.select('.article-body ul, .article-content ul'):
            list_items = list_el.select('li')
            if len(list_items) >= 2:  # At least 2 items to be considered a suggestion list
                for item in list_items:
                    text = item.get_text(strip=True)
                    if text and len(text) > 10:
                        suggestions.append({
                            "type": "bullet-point",
                            "text": text
                        })
    
    elif domain == "healthline" or domain == "medicalnewstoday":
        # These sites often use sections with clear headings for suggestions
        for section in soup.select('section, .article-body > div'):
            heading = section.select_one('h2, h3')
            if heading and any(keyword in heading.get_text().lower() for keyword in ['tip', 'advice', 'how', 'should', 'recommendation', 'guide']):
                paragraphs = [p.get_text(strip=True) for p in section.select('p')]
                if paragraphs:
                    suggestions.append({
                        "type": "section",
                        "heading": heading.get_text(strip=True),
                        "text": " ".join(paragraphs)
                    })
    
    # Extract list items that look like suggestions
    for list_item in soup.select('li'):
        text = list_item.get_text(strip=True)
        # Look for actionable items (starting with verbs)
        if re.match(r'^(?:Try|Use|Eat|Avoid|Take|Exercise|Monitor|Check|Consider|Reduce|Increase|Maintain|Keep)\b', text, re.IGNORECASE):
            if len(text) > 15:
                suggestions.append({
                    "type": "action-item",
                    "text": text
                })
    
    return suggestions

def extract_medical_terms_from_content(content, url):
    """Extract potential medical terms from the content"""
    medical_terms = {}
    
    # Common diabetes-related medical terms patterns
    patterns = [
        r'\b(Type \d (?:diabetes|mellitus))\b',
        r'\b(HbA1c|A1C|glycated hemoglobin)\b',
        r'\b(hyperglycemia|hypoglycemia)\b',
        r'\b(insulin resistance|glucose tolerance|gestational diabetes)\b',
        r'\b(retinopathy|neuropathy|nephropathy)\b',
        r'\b(ketoacidosis|ketones|DKA)\b',
        r'\b(metformin|sulfonylureas|GLP-1|SGLT2)\b',
        r'\b(continuous glucose monitor(?:ing)?|CGM)\b',
        r'\b(glycemic index|glycemic load)\b'
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        for match in matches:
            term = match if isinstance(match, str) else ' '.join(match)
            if term.lower() not in medical_terms:
                context = find_context(content, term)
                if context:
                    medical_terms[term.lower()] = {
                        "term": term,
                        "context": context,
                        "sources": [url]
                    }
    
    return medical_terms

def find_context(content, term):
    """Find surrounding context for a term"""
    sentences = re.split(r'(?<=[.!?])\s+', content)
    context_sentences = []
    
    for sentence in sentences:
        if re.search(r'\b' + re.escape(term) + r'\b', sentence, re.IGNORECASE):
            context_sentences.append(sentence)
    
    # Get surrounding sentences for better context
    if context_sentences:
        sentence_indexes = [sentences.index(s) for s in context_sentences if s in sentences]
        expanded_context = []
        
        for idx in sentence_indexes:
            start = max(0, idx - 1)
            end = min(len(sentences), idx + 2)
            expanded_context.extend(sentences[start:end])
        
        # Remove duplicates while preserving order
        seen = set()
        expanded_context = [s for s in expanded_context if not (s in seen or seen.add(s))]
        
        return " ".join(expanded_context)
    
    return ""

def crawl_websites(base_urls, max_depth=2, max_pages=25):
    """Crawl websites starting from base URLs up to a certain depth"""
    all_urls = set(base_urls)
    visited = set()
    queue = [(url, 0) for url in base_urls]  # (url, depth)
    results = {}
    page_count = 0
    
    while queue and page_count < max_pages:
        url, depth = queue.pop(0)
        
        if url in visited or page_count >= max_pages:
            continue
        
        print(f"Scraping {url} (depth {depth})...")
        
        # Add random delay to be respectful
        time.sleep(random.uniform(1, 3))
        
        # Scrape the page
        page_data = scrape_website(url)
        visited.add(url)
        page_count += 1
        
        if page_data:
            results[url] = page_data
            
            # Process suggestions
            for suggestion in page_data.get("suggestions", []):
                suggestion_type = suggestion.get("type", "general")
                if suggestion_type not in scraped_content["suggestions"]:
                    scraped_content["suggestions"][suggestion_type] = []
                
                scraped_content["suggestions"][suggestion_type].append({
                    "text": suggestion.get("text", ""),
                    "source": url,
                    "source_title": page_data.get("title", "Unknown")
                })
            
            # Process medical terms
            for term, term_data in page_data.get("medical_terms", {}).items():
                if term not in scraped_content["medical_terms"]:
                    scraped_content["medical_terms"][term] = term_data
                else:
                    # Merge sources
                    for source in term_data.get("sources", []):
                        if source not in scraped_content["medical_terms"][term]["sources"]:
                            scraped_content["medical_terms"][term]["sources"].append(source)
                    
                    # Merge context
                    if len(term_data.get("context", "")) > len(scraped_content["medical_terms"][term].get("context", "")):
                        scraped_content["medical_terms"][term]["context"] = term_data["context"]
            
            # Only add new links if we haven't reached max depth
            if depth < max_depth:
                for link in page_data.get("internal_links", []):
                    if link not in visited and link not in [u for u, _ in queue]:
                        queue.append((link, depth + 1))
                        all_urls.add(link)
    
    print(f"Crawled {len(visited)} pages")
    return results

def save_scraped_data():
    """Save all scraped data to files"""
    # Save suggestions
    with open(os.path.join(DATA_DIR, 'suggestions.json'), 'w', encoding='utf-8') as f:
        json.dump(scraped_content["suggestions"], f, indent=2)
    
    # Save medical terms
    with open(os.path.join(DATA_DIR, 'medical_terms.json'), 'w', encoding='utf-8') as f:
        json.dump(scraped_content["medical_terms"], f, indent=2)
    
    print(f"Saved scraped data to {DATA_DIR}")

def load_scraped_data():
    """Load scraped data from files"""
    # Load suggestions
    suggestions_path = os.path.join(DATA_DIR, 'suggestions.json')
    if os.path.exists(suggestions_path):
        with open(suggestions_path, 'r', encoding='utf-8') as f:
            scraped_content["suggestions"] = json.load(f)
    
    # Load medical terms
    terms_path = os.path.join(DATA_DIR, 'medical_terms.json')
    if os.path.exists(terms_path):
        with open(terms_path, 'r', encoding='utf-8') as f:
            scraped_content["medical_terms"] = json.load(f)
    
    print(f"Loaded scraped data from {DATA_DIR}")

def search_scraped_data(query):
    """Search through all scraped data for the query"""
    query = query.lower()
    results = []
    
    # Search in suggestions
    for suggestion_type, suggestions in scraped_content["suggestions"].items():
        for suggestion in suggestions:
            if query in suggestion["text"].lower():
                results.append({
                    "type": "suggestion",
                    "category": suggestion_type,
                    "text": suggestion["text"],
                    "source": suggestion["source"],
                    "source_title": suggestion.get("source_title", "Unknown")
                })
    
    # Search in medical terms
    for term, term_data in scraped_content["medical_terms"].items():
        if query in term or query in term_data.get("context", "").lower():
            results.append({
                "type": "medical_term",
                "term": term_data["term"],
                "context": term_data["context"],
                "sources": term_data["sources"]
            })
    
    return results

def generate_suggestions(query, llm):
    """Generate suggestions based on a user query using web scraping and LLM"""
    print(f"Generating suggestions for: {query}")
    
    # First, check if we have relevant data already scraped
    search_results = search_scraped_data(query)
    
    # If no results, try to retrieve fresh data (optional real-time search)
    if not search_results and query:
        # Perform a targeted crawl based on the query
        # This is optional and might be slow - consider commenting out for faster responses
        # search_urls = get_search_urls_for_query(query)
        # if search_urls:
        #     crawl_websites(search_urls, max_depth=1, max_pages=3)
        #     search_results = search_scraped_data(query)
        pass
    
    # Format the suggestions
    formatted_suggestions = format_search_results(search_results)
    
    # Generate a response using the LLM
    if formatted_suggestions:
        prompt = f"""
You are a helpful diabetes assistant that provides clear and practical suggestions.
Based on the user's question: "{query}"

Here are relevant suggestions from trusted medical sources:

{formatted_suggestions}

Please provide a helpful response that:
1. Directly answers the user's question
2. Summarizes the most important suggestions
3. Organizes the information in a clear, easy-to-understand way
4. Explains medical terms in simple language
5. Reminds the user to consult healthcare providers for personalized advice

Your response:
"""
        
        try:
            response = llm(prompt)
            return response
        except Exception as e:
            print(f"Error generating LLM response: {str(e)}")
            return "I'm sorry, I encountered an error while generating suggestions. Please try again."
    else:
        # If no suggestions found, use the LLM for a more generic response
        prompt = f"""
You are a helpful diabetes assistant that provides clear and practical suggestions.
The user asked: "{query}"

Unfortunately, I don't have specific information from my knowledge base about this query.
Please provide a helpful, general response that:
1. Acknowledges the limitations of your knowledge
2. Provides general guidance related to diabetes management if applicable
3. Suggests reliable sources where they might find more information
4. Reminds the user to consult healthcare providers for personalized advice

Your response:
"""
        
        try:
            response = llm(prompt)
            return response
        except Exception as e:
            print(f"Error generating LLM response: {str(e)}")
            return "I'm sorry, I don't have specific suggestions for that query in my database. Please try a different question or consult your healthcare provider."

def format_search_results(results):
    """Format search results into a readable format for the LLM"""
    if not results:
        return "No relevant suggestions found."
    
    formatted_text = []
    
    # Group by type
    suggestions = [r for r in results if r["type"] == "suggestion"]
    medical_terms = [r for r in results if r["type"] == "medical_term"]
    
    # Format suggestions
    if suggestions:
        formatted_text.append("## Relevant Suggestions")
        for i, suggestion in enumerate(suggestions[:10], 1):  # Limit to top 10
            formatted_text.append(f"{i}. {suggestion['text']}")
            formatted_text.append(f"   Source: {suggestion.get('source_title', 'Unknown')}")
            formatted_text.append("")
    
    # Format medical terms
    if medical_terms:
        formatted_text.append("## Related Medical Terms")
        for i, term in enumerate(medical_terms[:5], 1):  # Limit to top 5
            formatted_text.append(f"{i}. **{term['term']}**")
            formatted_text.append(f"   {term['context']}")
            formatted_text.append("")
    
    # Add source count information
    unique_sources = set()
    for result in results:
        if result["type"] == "suggestion":
            unique_sources.add(result.get("source", ""))
        else:
            unique_sources.update(result.get("sources", []))
    
    formatted_text.append(f"Information compiled from {len(unique_sources)} medical sources.")
    
    return "\n".join(formatted_text)

def run_initial_scraping():
    """Run the initial scraping of websites"""
    print("Starting initial web scraping...")
    crawl_websites(BASE_URLS)
    save_scraped_data()
    print("Initial scraping completed!")

def get_suggestion(query, history=None):
    """Process a query and return suggestions"""
    # Load data if not already loaded
    if not scraped_content["suggestions"] and not scraped_content["medical_terms"]:
        load_scraped_data()
    
    # If no data is available, run initial scraping
    if not scraped_content["suggestions"] and not scraped_content["medical_terms"]:
        run_initial_scraping()
    
    # Generate suggestions
    response = generate_suggestions(query, llm)
    return response

def update_database():
    """Update the suggestion database by scraping websites again"""
    run_initial_scraping()
    return f"Database updated! {len(scraped_content['suggestions'])} suggestion categories and {len(scraped_content['medical_terms'])} medical terms extracted."

def get_stats():
    """Get statistics about the database"""
    suggestion_count = sum(len(suggestions) for suggestions in scraped_content["suggestions"].values())
    term_count = len(scraped_content["medical_terms"])
    categories = list(scraped_content["suggestions"].keys())
    
    stats = f"""
## Database Statistics

- **Suggestion Categories:** {len(categories)}
- **Total Suggestions:** {suggestion_count}
- **Medical Terms:** {term_count}

### Categories:
{', '.join(categories)}
"""
    return stats

# Load the LLM
print("Loading LLM model...")
llm = load_llm()

# Try to load existing data
load_scraped_data()

# Create the Gradio interface
with gr.Blocks(title="Diabetes Assistant - Personalized Suggestions") as demo:
    gr.Markdown("# Diabetes Assistant - Personalized Suggestions")
    gr.Markdown("Ask any question to get personalized health suggestions for diabetes management.")
    
    with gr.Tab("Get Health Suggestions"):
        with gr.Row():
            with gr.Column(scale=6):
                query_input = gr.Textbox(
                    label="Ask for health suggestions",
                    placeholder="Example: What foods should I eat to manage blood sugar?",
                    lines=2
                )
                suggestion_button = gr.Button("Get Suggestions", variant="primary")
            
            with gr.Column(scale=4):
                gr.Markdown("""
                ### Example Queries:
                - What exercises are best for diabetes?
                - How can I reduce my A1C levels?
                - What should I eat for breakfast?
                - How to manage diabetes during travel?
                - What are symptoms of low blood sugar?
                """)
        
        suggestion_output = gr.Markdown(label="Health Suggestions")
        suggestion_button.click(fn=get_suggestion, inputs=query_input, outputs=suggestion_output)
    
    with gr.Tab("Medical Term Explorer"):
        gr.Markdown("### Search for medical terms related to diabetes")
        
        term_input = gr.Textbox(label="Enter medical term", placeholder="Example: hyperglycemia")
        term_button = gr.Button("Explain Term")
        term_output = gr.Markdown(label="Simplified Explanation")
        
        def explain_term(term):
            search_results = search_scraped_data(term)
            term_results = [r for r in search_results if r["type"] == "medical_term"]
            
            if not term_results:
                return "Term not found in database. Try a different term or ask in the main suggestion tab."
            
            # Use the LLM to explain the term in simple language
            term_data = term_results[0]
            prompt = f"""
You are a helpful medical assistant for diabetes patients.
Explain the term "{term_data['term']}" in simple, non-medical language that anyone can understand.
Use everyday examples and analogies when possible.

Here is some context about the term: "{term_data['context']}"

Simple explanation:
"""
            
            try:
                explanation = llm(prompt)
                result = f"## {term_data['term']}\n\n{explanation}\n\n"
                result += "Sources:\n" + "\n".join([f"- {url}" for url in term_data["sources"][:3]])
                return result
            except Exception as e:
                return f"Error generating explanation: {str(e)}"
        
        term_button.click(fn=explain_term, inputs=term_input, outputs=term_output)
    
    with gr.Tab("Chat Assistant"):
        chatbot = gr.Chatbot(height=400)
        msg = gr.Textbox(
            show_label=False,
            placeholder="Ask anything about diabetes management...",
            container=False
        )
        
        def respond(message, chat_history):
            bot_message = get_suggestion(message, chat_history)
            chat_history.append((message, bot_message))
            return "", chat_history
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
    
    with gr.Tab("Admin"):
        gr.Markdown("### Update Suggestion Database")
        update_button = gr.Button("Update Database")
        update_output = gr.Markdown()
        update_button.click(fn=update_database, inputs=None, outputs=update_output)
        
        gr.Markdown("### Database Statistics")
        stats_button = gr.Button("View Database Stats")
        stats_output = gr.Markdown()
        stats_button.click(fn=get_stats, inputs=None, outputs=stats_output)

# Launch the app
if __name__ == "__main__":
    # Run initial scraping if no data is available
    if not scraped_content["suggestions"] and not scraped_content["medical_terms"]:
        run_initial_scraping()
    
    demo.launch()