import requests
import trafilatura
import time
import os
from urllib.parse import urljoin
import logging
from bs4 import BeautifulSoup
import re
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArxivScraper:
    def __init__(self, delay=1):
        """
        Initialize the ArXiv scraper
        
        Args:
            delay (int): Delay between requests in seconds to be respectful
        """
        self.delay = delay
        self.base_url = "https://arxiv.org"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_article_urls_from_list_page(self, list_url):
        """
        Extract article URLs from an arXiv list page using trafilatura
        
        Args:
            list_url (str): URL of the arXiv list page
            
        Returns:
            list: List of article URLs
        """
        try:
            logger.info(f"Fetching list page: {list_url}")
            response = self.session.get(list_url)
            response.raise_for_status()
            
            # Parse links using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            article_urls = []
            for a in soup.select('a[href*="/html/"]'):
                href = a.get('href')
                if not href:
                    continue
                full_url = urljoin(self.base_url, href)
                if 'arxiv.org' in full_url and full_url not in article_urls:
                    article_urls.append(full_url)
            
            logger.info(f"Found {len(article_urls)} article URLs from {list_url}")
            return article_urls
            
        except Exception as e:
            logger.error(f"Error fetching list page {list_url}: {str(e)}")
            return []

    
    def scrape_articles_from_lists(self, list_url, output_file='arxiv_articles.json'):
        """
        Scrape articles from multiple list pages
        
        Args:
            list_urls (list): List of arXiv list page URLs
            output_file (str): Output file name for saving results
            
        Returns:
            list: List of scraped article data
        """
        all_articles = []
        
        article_urls = self.get_article_urls_from_list_page(list_url)
        
        # Scrape each article
        for i, article_url in enumerate(article_urls, 1):
            logger.info(f"Processing article {i}/{len(article_urls)}")
            
            article_data = self.scrape_article_content(article_url)
            if article_data:
                all_articles.append(article_data)
                
                # Save progress periodically
                if i % 10 == 0:
                    self.save_articles(all_articles, f"temp_{output_file}")
                    logger.info(f"Saved progress: {len(all_articles)} articles")
        
        # Save final results
        self.save_articles(all_articles, output_file)
        
        # Clean up temporary file
        temp_file = f"temp_{output_file}"
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        logger.info(f"Scraping completed! Total articles scraped: {len(all_articles)}")
        return all_articles
    
    def remove_personal_identification(self, text):
        """
        Remove personal identification information from text
        
        Args:
            text (str): Text content to clean
            
        Returns:
            str: Text with personal identification removed
        """
        if not text:
            return text
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REMOVED]', text)
        
        # Remove phone numbers (various formats)
        # US/International formats: (123) 456-7890, 123-456-7890, 123.456.7890, +1 123 456 7890
        text = re.sub(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE_REMOVED]', text)
        
        # Remove URLs that might contain personal information
        # Keep only if it's clearly a research/academic URL
        text = re.sub(r'\bhttps?://[^\s]+', '[URL_REMOVED]', text)
        
        # Remove common patterns for personal addresses
        # Street addresses: "123 Main St", "456 Oak Avenue", etc.
        text = re.sub(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Circle|Cir)\b', '[ADDRESS_REMOVED]', text, flags=re.IGNORECASE)
        
        # Remove postal codes (US ZIP codes, international formats)
        text = re.sub(r'\b\d{5}(-\d{4})?\b', '[POSTAL_CODE_REMOVED]', text)  # US ZIP
        text = re.sub(r'\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b', '[POSTAL_CODE_REMOVED]', text)  # Canadian postal codes
        
        # Remove common author name patterns (First Last, Last, First, etc.)
        # This is a heuristic - looks for capitalized words that might be names
        # Note: This might be too aggressive, but better safe than sorry
        # Pattern: "Author Name" or "Name, Author" at start of lines or after common prefixes
        text = re.sub(r'\b(?:Author|Authors?|By|Written by|Correspondence to|Contact):\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', '[AUTHOR_NAME_REMOVED]', text, flags=re.IGNORECASE)
        
        # Remove ORCID identifiers
        text = re.sub(r'\b(?:ORCID|orcid\.org)[:\s]*\d{4}-\d{4}-\d{4}-\d{3}[\dX]', '[ORCID_REMOVED]', text, flags=re.IGNORECASE)
        
        # Remove affiliation patterns that might contain personal info
        # "Department of X, University of Y, City, Country"
        # This is tricky - we'll be conservative and only remove if it looks like contact info
        text = re.sub(r'\b(?:Affiliation|Department|Institution)[:\s]*[^\.]+', '[AFFILIATION_REMOVED]', text, flags=re.IGNORECASE)
        
        return text
    
    def scrape_article_content(self, article_url):
        """
        Scrape content from a single arXiv article page using trafilatura
        
        Args:
            article_url (str): URL of the article page
            
        Returns:
            dict: Dictionary containing 'title', 'author', and 'content' fields,
                  with personal identifications (such as names and emails) removed.
        """
        try:
            logger.info(f"Scraping article: {article_url}")

            # Add delay to be respectful
            time.sleep(self.delay)

            response = self.session.get(article_url)
            response.raise_for_status()

            # Use trafilatura to extract the main content.
            extracted_content = trafilatura.extract(
                response.text,
                include_comments=False,
                include_tables=False,
                include_formatting=False,
                include_images=False,
                include_links=False
            )

            # Remove personal identification information from content
            if extracted_content:
                extracted_content = self.remove_personal_identification(extracted_content)
            else:
                extracted_content = ""

            # Return as dictionary
            return {
                'article': extracted_content
            }

        except Exception as e:
            logger.error(f"Error scraping article {article_url}: {str(e)}")
            return None

    def save_articles(self, articles, filename):
        """
        Save articles to a JSON file (as a JSON array)
        
        Args:
            articles (list): List of article dictionaries with 'title', 'author', and 'content'
            filename (str): Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Write as a JSON array instead of JSONL format
                json.dump(articles, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(articles)} articles to {filename}")
        except Exception as e:
            logger.error(f"Error saving articles to {filename}: {str(e)}")
    
    
    def download_pdfs(self, pdf_urls, output_dir="pdfs", delay=None):
        """
        Download PDF files from a list of URLs
        
        Args:
            pdf_urls (list): List of PDF URLs to download
            output_dir (str): Directory to save PDFs (default: "pdfs")
            delay (int): Override the default delay between downloads. If None, uses self.delay
            
        Returns:
            dict: Dictionary with 'successful' and 'failed' lists containing the URLs
        """
        # Use provided delay or default delay
        download_delay = delay if delay is not None else self.delay
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        successful = []
        failed = []
        
        logger.info(f"Starting PDF download for {len(pdf_urls)} URLs")
        
        for i, pdf_url in enumerate(pdf_urls, 1):
            try:
                logger.info(f"Downloading PDF {i}/{len(pdf_urls)}: {pdf_url}")
                
                # Add delay to be respectful
                if download_delay > 0:
                    time.sleep(download_delay)
                
                response = self.session.get(pdf_url)
                response.raise_for_status()
                
                # Extract filename from URL or create one
                filename = pdf_url.split('/')[-1]
                if not filename.lower().endswith('.pdf'):
                    # If URL doesn't end with .pdf, create a filename from the URL
                    filename = f"paper_{i}.pdf"
                
                filepath = os.path.join(output_dir, filename)
                
                # Handle duplicate filenames
                counter = 1
                original_filepath = filepath
                while os.path.exists(filepath):
                    name, ext = os.path.splitext(original_filepath)
                    filepath = f"{name}_{counter}{ext}"
                    counter += 1
                
                # Save the PDF
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Successfully downloaded: {filepath}")
                successful.append(pdf_url)
                
            except Exception as e:
                logger.error(f"Failed to download {pdf_url}: {str(e)}")
                failed.append(pdf_url)
        
        logger.info(f"PDF download completed! Successful: {len(successful)}, Failed: {len(failed)}")
        return {"successful": successful, "failed": failed}
# def main():
#     """
#     Main function to run the scraper
#     """
#     # URLs to scrape
#     list_urls = [
#         "https://arxiv.org/list/econ.EM/recent"
#     ]
    
#     # Initialize scraper with 2-second delay between requests
#     scraper = ArxivScraper(delay=2)
    
#     # Scrape articles
#     articles = scraper.scrape_articles_from_lists(list_urls, './homework/arxiv_articles.text')
    
#     # Print summary
#     print(f"\n=== Scraping Summary ===")
#     print(f"Total articles scraped: {len(articles)}")
#     print(f"Results saved to: arxiv_articles.text")

# if __name__ == "__main__":
#     main()
