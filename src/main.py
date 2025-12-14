"""
Main script for synthetic data generation pipeline:
1. Scrape articles from arXiv
2. Generate QA pairs using Mistral-7B-Instruct
"""

from web_scraper import ArxivScraper
from qa_generator import QAGenerator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Main pipeline: Scrape articles and generate QA pairs
    """
    # Step 1: Scrape articles from arXiv
    logger.info("=== Step 1: Scraping articles from arXiv ===")
    # scraper = ArxivScraper(delay=2)
    # output_file = 'arxiv_articles.json'
    
    # # Scrape articles (adjust URL as needed)
    # articles = scraper.scrape_articles_from_lists(
    #     "https://arxiv.org/list/cs/recent?skip=0&show=100",
    #     output_file=output_file
    # )
    
    # logger.info(f"Scraped {len(articles)} articles")
    
    # Step 2: Generate QA pairs
    logger.info("=== Step 2: Generating QA pairs ===")
    qa_generator = QAGenerator(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        device=None  # Auto-detect (cuda if available, else cpu)
    )
    
    # Process articles and generate QA pairs
    qa_output_file = 'arxiv_articles_with_qa.json'
    qa_generator.generate_qa_from_file(
        input_file='arxiv_articles_test.json',
        output_file=qa_output_file,
        num_questions=5  # Generate 5 questions per article
    )
    
    logger.info(f"Pipeline completed! Results saved to {qa_output_file}")


if __name__ == "__main__":
    main()

