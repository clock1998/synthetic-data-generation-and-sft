from web_scraper import ArxivScraper
from qa_generator import QAGenerator
# from unsloth import FastLanguageModel, SFTTrainer
from transformers import AutoTokenizer, TrainingArguments
from datasets import load_dataset
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
        model_name="mistralai/Mistral-7B-Instruct-v0.3"
    )
    
    # Process articles and generate QA pairs
    qa_output_file = 'arxiv_articles_with_qa.jsonl'
    qa_generator.generate_qa_from_file(
        input_file='arxiv_articles_test.json',
        output_file=qa_output_file,
        num_questions=5  # Generate 5 questions per article
    )
    
    logger.info(f"QA Results saved to {qa_output_file}")

    logger.info(f"Start SFT pipeline")
    # # Load the base LLaMA 3 7B model in 4-bit mode (dynamic 4-bit quantization)
    # model_name = "unsloth/llama-3.1-7b-unsloth-bnb-4bit"
    # model = FastLanguageModel.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    # # Load our synthetic Q&A dataset
    # dataset = load_dataset("json", data_files="synthetic_qa.jsonl", split="train")

    # # Initialize the trainer for Supervised Fine-Tuning (SFT)
    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=dataset,
    #     dataset_text_field="text",
    #     args=TrainingArguments(
    #         output_dir="llama3-7b-qlora-finetuned",
    #         per_device_train_batch_size=4,   # small batch size for Colab GPU
    #         gradient_accumulation_steps=4,   # accumulate gradients to simulate larger batch
    #         num_train_epochs=2,
    #         learning_rate=2e-4,
    #         fp16=True,
    #         logging_steps=50,
    #         save_strategy="epoch"
    #     )
    # )

    # trainer.train()
    # model.save_pretrained("llama3-7b-qlora-finetuned")

if __name__ == "__main__":
    main()

