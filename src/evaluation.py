"""
Evaluation script for comparing base model vs fine-tuned model performance on QA tasks.
"""

import json
import logging
import torch
from typing import List, Dict, Any
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates and compares base vs fine-tuned model performance on QA tasks.
    """
    
    def __init__(self, base_model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct", 
                 fine_tuned_path: str = "Meta-Llama-3.1-8B-Instruct-Fine-Tuned",
                 device: str = "cuda"):
        """
        Initialize evaluator with model paths.
        
        Args:
            base_model_name: Hugging Face model name for base model
            fine_tuned_path: Local path to fine-tuned model
            device: Device to run models on
        """
        self.device = device
        self.system_prompt = "You are a helpful academic Q&A assistant specialized in scholarly content."
        
        logger.info("Loading base model...")
        self.base_model, self.base_tokenizer = FastLanguageModel.from_pretrained(base_model_name)
        
        logger.info("Loading fine-tuned model...")
        try:
            self.ft_model, self.ft_tokenizer = FastLanguageModel.from_pretrained(fine_tuned_path)
        except:
            # If fine-tuned model fails to load, try loading as regular transformers model
            from transformers import AutoModelForCausalLM
            self.ft_model = AutoModelForCausalLM.from_pretrained(fine_tuned_path)
            self.ft_tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
            
        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        logger.info("Models loaded successfully")
    
    def generate_response(self, model, tokenizer, question: str, max_new_tokens: int = 150) -> str:
        """
        Generate response for a question using specified model.
        
        Args:
            model: The model to use
            tokenizer: The tokenizer to use
            question: Input question
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            str: Generated response
        """
        prompt = f"<|system|>{self.system_prompt}<|user|>{question}<|assistant|>"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, 'to'):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        return response
    
    def evaluate_on_questions(self, test_questions: List[str], reference_answers: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate models on a set of test questions.
        
        Args:
            test_questions: List of questions to test
            reference_answers: Optional list of reference answers for scoring
            
        Returns:
            Dict containing evaluation results
        """
        results = {
            "questions": [],
            "base_responses": [],
            "ft_responses": [],
            "metrics": {}
        }
        
        logger.info(f"Evaluating {len(test_questions)} questions...")
        
        for i, question in enumerate(test_questions):
            logger.info(f"Processing question {i+1}/{len(test_questions)}")
            
            # Generate responses
            base_response = self.generate_response(self.base_model, self.base_tokenizer, question)
            ft_response = self.generate_response(self.ft_model, self.ft_tokenizer, question)
            
            results["questions"].append(question)
            results["base_responses"].append(base_response)
            results["ft_responses"].append(ft_response)
            
            # Print comparison
            print(f"\nQ{i+1}: {question}")
            print(f"Base: {base_response[:200]}{'...' if len(base_response) > 200 else ''}")
            print(f"FT:   {ft_response[:200]}{'...' if len(ft_response) > 200 else ''}")
            print("-" * 80)
        
        # Calculate metrics if reference answers provided
        if reference_answers:
            results["metrics"] = self.calculate_metrics(
                results["base_responses"], 
                results["ft_responses"], 
                reference_answers
            )
        
        return results
    
    def calculate_metrics(self, base_responses: List[str], ft_responses: List[str], 
                         reference_answers: List[str]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics comparing model responses to references.
        
        Args:
            base_responses: Responses from base model
            ft_responses: Responses from fine-tuned model
            reference_answers: Ground truth answers
            
        Returns:
            Dict containing various metrics
        """
        metrics = {
            "base_model": {},
            "fine_tuned_model": {}
        }
        
        # ROUGE scores
        for model_name, responses in [("base_model", base_responses), ("fine_tuned_model", ft_responses)]:
            rouge1_scores = []
            rouge2_scores = []
            rougeL_scores = []
            
            for response, reference in zip(responses, reference_answers):
                scores = self.rouge_scorer.score(reference, response)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            
            metrics[model_name]["rouge1"] = {
                "mean": np.mean(rouge1_scores),
                "std": np.std(rouge1_scores)
            }
            metrics[model_name]["rouge2"] = {
                "mean": np.mean(rouge2_scores),
                "std": np.std(rouge2_scores)
            }
            metrics[model_name]["rougeL"] = {
                "mean": np.mean(rougeL_scores),
                "std": np.std(rougeL_scores)
            }
        
        # BERTScore (if available)
        try:
            logger.info("Calculating BERTScore...")
            base_P, base_R, base_F1 = bert_score(base_responses, reference_answers, lang="en")
            ft_P, ft_R, ft_F1 = bert_score(ft_responses, reference_answers, lang="en")
            
            metrics["base_model"]["bertscore"] = {
                "precision": base_P.mean().item(),
                "recall": base_R.mean().item(),
                "f1": base_F1.mean().item()
            }
            metrics["fine_tuned_model"]["bertscore"] = {
                "precision": ft_P.mean().item(),
                "recall": ft_R.mean().item(),
                "f1": ft_F1.mean().item()
            }
        except Exception as e:
            logger.warning(f"BERTScore calculation failed: {e}")
        
        # Response length comparison
        base_lengths = [len(resp.split()) for resp in base_responses]
        ft_lengths = [len(resp.split()) for resp in ft_responses]
        
        metrics["response_lengths"] = {
            "base_model": {
                "mean": np.mean(base_lengths),
                "std": np.std(base_lengths)
            },
            "fine_tuned_model": {
                "mean": np.mean(ft_lengths),
                "std": np.std(ft_lengths)
            }
        }
        
        return metrics
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Evaluation results dictionary
            output_file: Output file path
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")


def create_test_questions() -> List[str]:
    """
    Create a set of test questions for evaluation.
    These should be different from training questions.
    
    Returns:
        List of test questions
    """
    return [
        "What are the key components of a world model in robotics?",
        "How do diffusion models work in image generation?",
        "What is the difference between supervised and unsupervised learning?",
        "Explain the concept of attention mechanisms in transformers.",
        "What are the challenges in training large language models?",
        "How does reinforcement learning differ from supervised learning?",
        "What is the role of batch normalization in neural networks?",
        "Explain the concept of overfitting in machine learning.",
        "What are the advantages of using pre-trained models?",
        "How do convolutional neural networks process image data?"
    ]


def main():
    """
    Main evaluation function.
    """
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Create test questions
    test_questions = create_test_questions()
    
    # Run evaluation (without reference answers for now)
    results = evaluator.evaluate_on_questions(test_questions)
    
    # Save results
    evaluator.save_results(results, "evaluation_results.json")
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()