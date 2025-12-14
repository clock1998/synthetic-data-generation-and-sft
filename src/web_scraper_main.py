from web_scraper import ArxivScraper

scraper = ArxivScraper(delay=2)

scraper.scrape_articles_from_lists("https://arxiv.org/list/cs/recent?skip=0&show=100")

