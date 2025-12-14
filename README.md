# Synthetic Data Generation and SFT

This project provides a pipeline for scraping academic articles from arXiv and generating question-answer pairs using Mistral-7B-Instruct model for synthetic data generation and supervised fine-tuning (SFT).

## Features

- **Web Scraping**: Scrape articles from arXiv with automatic PII (Personal Identifiable Information) removal
- **QA Generation**: Generate question-answer pairs from articles using Mistral-7B-Instruct model
- **JSONL Output**: Save results in JSONL format for easy processing

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. The model will be automatically downloaded from Hugging Face on first use.

## Usage

### Basic Pipeline

Run the complete pipeline (scraping + QA generation):

```bash
python src/main.py
```

### Step-by-Step Usage

#### 1. Scrape Articles

```python
from web_scraper import ArxivScraper

scraper = ArxivScraper(delay=2)
articles = scraper.scrape_articles_from_lists(
    "https://arxiv.org/list/cs/recent?skip=0&show=100",
    output_file='arxiv_articles.json'
)
```

#### 2. Generate QA Pairs

```python
from qa_generator import QAGenerator

qa_generator = QAGenerator(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    device=None  # Auto-detect (cuda if available, else cpu)
)

# Process articles from file
qa_generator.process_articles_from_file(
    input_file='arxiv_articles.json',
    output_file='arxiv_articles_with_qa.json',
    num_questions=5
)
```

#### 3. Process Single Article

```python
article_data = {'article': 'Your article text here...'}
result = qa_generator.process_article(article_data)
print(result['qa_pairs'])
```

## Output Format

Each article in the output JSONL file contains:

```json
{
  "article": "Article content with PII removed...",
  "qa_pairs": [
    {
      "question": "What is the main contribution?",
      "answer": "The main contribution is..."
    },
    ...
  ]
}
```

## Configuration

### QA Generator Parameters

- `model_name`: Hugging Face model identifier (default: "mistralai/Mistral-7B-Instruct-v0.3")
- `device`: Device to run on ("cuda", "cpu", or None for auto-detection)
- `num_questions`: Number of questions to generate per article (default: 5)
- `max_new_tokens`: Maximum tokens to generate (default: 1000)

### Web Scraper Parameters

- `delay`: Delay between requests in seconds (default: 1)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA (optional, for GPU acceleration)

## Notes

- The model requires significant memory (approximately 14GB for 7B parameters in FP16)
- For CPU usage, the model will use FP32 which requires more memory
- Consider using quantization or smaller models for limited resources
- PII removal includes emails, phone numbers, addresses, and author names

