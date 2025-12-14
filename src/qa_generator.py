import json
import logging
import os
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QAGenerator:
    """
    Question-Answer generation pipeline using Mistral-7B-Instruct model
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.3", device: Optional[str] = None):
        """
        Initialize the QA Generator with Mistral model
        
        Args:
            model_name (str): Hugging Face model identifier
            device (str): Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        logger.info(f"Loading model: {model_name}")
        
        # Determine device
        if device is None:
            self.device = "cuda"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate_qa_pairs(self, article_text: str, num_questions: int = 5, max_new_tokens: int = 1000) -> List[Dict[str, str]]:
        """
        Generate question-answer pairs from article text
        
        Args:
            article_text (str): The article content
            num_questions (int): Number of questions to generate
            max_new_tokens (int): Maximum number of tokens to generate
            
        Returns:
            List[Dict[str, str]]: List of dictionaries with 'question' and 'answer' keys
        """
        if not article_text or len(article_text.strip()) == 0:
            logger.warning("Empty article text provided")
            return []
        
        try:
            # Create prompt
            prompt = self._create_prompt(article_text, num_questions)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            logger.info("Generating QA pairs...")
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (after the prompt)
            if "[/INST]" in generated_text:
                generated_text = generated_text.split("[/INST]")[-1].strip()
            
            # Try to parse JSON from the generated text
            qa_pairs = self._parse_qa_from_text(generated_text)
            
            logger.info(f"Generated {len(qa_pairs)} QA pairs")
            return qa_pairs
            
        except Exception as e:
            logger.error(f"Error generating QA pairs: {str(e)}")
            return []
    
    def _parse_qa_from_text(self, text: str) -> List[Dict[str, str]]:
        """
        Parse QA pairs from generated text, handling various formats
        
        Args:
            text (str): Generated text containing QA pairs
            
        Returns:
            List[Dict[str, str]]: Parsed QA pairs
        """
        qa_pairs = []
        
        # Try to extract JSON array
        try:
            # Look for JSON array in the text

            # Find all likely JSON array substrings using a regex
            array_candidates = re.findall(r'(\[\n\s*{\n[\s\S]*?\]\s*)', text)
            for candidate in array_candidates:
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, list):
                        qa_pairs = parsed
                        break
                except Exception:
                    continue
                # Validate structure
                if isinstance(qa_pairs, list):
                    validated_pairs = []
                    for item in qa_pairs:
                        if isinstance(item, dict) and 'question' in item and 'answer' in item:
                            validated_pairs.append({
                                'question': str(item['question']).strip(),
                                'answer': str(item['answer']).strip()
                            })
                    return validated_pairs
        except json.JSONDecodeError:
            logger.warning("Could not parse JSON, trying alternative parsing methods")
        
        # Fallback: Try to extract Q&A pairs using regex patterns
        # Pattern for Q: ... A: ... format
        pattern = r'(?:Q|Question)[:\s]+(.+?)(?:A|Answer)[:\s]+(.+?)(?=(?:Q|Question)[:\s]|$)'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            if len(match) == 2:
                qa_pairs.append({
                    'question': match[0].strip(),
                    'answer': match[1].strip()
                })
        
        return qa_pairs
    
    
    def generate_qa_from_file(self, input_file: str, output_file: str, num_questions: int = 5):
        """
        Process articles from a JSONL file and generate QA pairs
        
        Args:
            input_file (str): Input JSONL file path
            output_file (str): Output JSONL file path
            num_questions (int): Number of questions per article
        """
        logger.info(f"Processing articles from {input_file}")
        
        generated_qa_paires = []
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                try:
                    articles = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON file: {str(e)}")
                    return

            for idx, article in enumerate(articles, 1):
                try:
                    logger.info(f"Processing article {idx}")

                    # Generate QA pairs
                    article_text = article.get('article', '')
                    if article_text:
                        qa_pairs = self.generate_qa_pairs(article_text, num_questions=num_questions)
                        generated_qa_paires.extend(qa_pairs)

                except Exception as e:
                    logger.error(f"Error processing article at index {idx}: {str(e)}")
                    continue
            
            # Save final results
            self._save_qa_pairs(generated_qa_paires, output_file)
            
            # Clean up temp file
            temp_file = f"temp_{output_file}"
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            logger.info(f"Processing completed! Processed {len(generated_qa_paires)} articles")
            logger.info(f"Results saved to {output_file}")
            
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_file}")
        except Exception as e:
            logger.error(f"Error processing articles: {str(e)}")
        
    def _create_prompt(self, article_text: str, num_questions: int = 5) -> str:
        """
        Create a prompt for QA generation
        
        Args:
            article_text (str): The article content
            num_questions (int): Number of questions to generate
            
        Returns:
            str: Formatted prompt
        """
        # Truncate article if too long (to fit in context window)
        max_length = 2000  # Adjust based on model context window
        if len(article_text) > max_length:
            article_text = article_text[:max_length] + "..."
        
        prompt = f"""<s>[INST] You are an expert at creating educational question-answer pairs from academic articles. 
                    Generate {num_questions} diverse and meaningful question-answer pairs based on the following article content.

                    The questions should:
                    - Cover different aspects of the article
                    - Range from factual to analytical
                    - Be clear and well-formulated
                    - Have answers that can be directly found in or inferred from the article

                    Format your response as a JSON array where each object has "question" and "answer" fields.

                    Article content:
                    {article_text}

                    Generate the question-answer pairs in JSON format: [/INST]"""
        
        return prompt

    def _save_qa_pairs(self, qa_paires: List[Dict], filename: str):
        """
        Save QA to JSONL file
        
        Args:
            articles (List[Dict]): List of QA dictionaries
            filename (str): Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                text = json.dumps(qa_paires, ensure_ascii=False)
                f.write(text)                    
        except Exception as e:
            logger.error(f"Error saving articles to {filename}: {str(e)}")

