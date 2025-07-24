import torch
from datasets import Dataset
from transformers import T5ForConditionalGeneration, RobertaTokenizer
import nltk
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/root/ASE/generation/result.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

parser = argparse.ArgumentParser(description="A script for code generation evaluation")
parser.add_argument('--dataset', type=str, default="CodeXGLUE", help='Dataset name')
parser.add_argument('--method', type=str, default="None", help='Method name')
parser.add_argument('--type', type=str, default="", help='Type parameter')
parser.add_argument('--pr', type=float, default=0, help='Parameter value')
args = parser.parse_args()

# Load JSON data into a Dataset
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return Dataset.from_list(data)

# Preprocessing function
def preprocess_function(examples):
    inputs = examples['docstring']
    targets = examples['code']
    model_inputs = tokenizer(inputs, max_length=320, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=150, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Custom CodeBLEU and metrics computation
def compute_metrics(predictions, references):
    # BLEU-4 smoothing function
    smoothing_function = nltk.translate.bleu_score.SmoothingFunction().method4

    bleus = []
    exact_matches = []
    codebleu_scores = []

    # Java keywords for weighted n-gram matching
    java_keywords = {
        'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class', 'const',
        'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final', 'finally', 'float',
        'for', 'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface', 'long', 'native',
        'new', 'package', 'private', 'protected', 'public', 'return', 'short', 'static', 'strictfp',
        'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 'transient', 'try', 'void',
        'volatile', 'while'
    }

    for pred, ref in zip(predictions, references):
        # Tokenize prediction and reference
        pred_tokens = pred.split()
        ref_tokens = ref.split()

        # BLEU-4 score
        try:
            bleu_score = nltk.translate.bleu_score.sentence_bleu(
                [ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing_function
            )
        except ZeroDivisionError:
            bleu_score = 0.0
        bleus.append(bleu_score)

        # Exact Match
        exact_match = 1 if pred.strip() == ref.strip() else 0
        exact_matches.append(exact_match)

        # Custom CodeBLEU components
        # 1. n-gram precision (similar to BLEU, but simplified)
        ngram_precision = bleu_score  # Reuse BLEU score as base n-gram precision

        # 2. Weighted n-gram precision (emphasize Java keywords)
        pred_keywords = [t for t in pred_tokens if t in java_keywords]
        ref_keywords = [t for t in ref_tokens if t in java_keywords]
        keyword_overlap = len(set(pred_keywords) & set(ref_keywords))
        keyword_precision = keyword_overlap / (len(pred_keywords) + 1e-10)  # Avoid division by zero
        keyword_recall = keyword_overlap / (len(ref_keywords) + 1e-10)
        weighted_ngram = 2 * (keyword_precision * keyword_recall) / (keyword_precision + keyword_recall + 1e-10)

        # 3. Structural similarity (simplified: compare token length and keyword density)
        length_similarity = min(len(pred_tokens), len(ref_tokens)) / (max(len(pred_tokens), len(ref_tokens)) + 1e-10)
        pred_keyword_density = len(pred_keywords) / (len(pred_tokens) + 1e-10)
        ref_keyword_density = len(ref_keywords) / (len(ref_tokens) + 1e-10)
        density_similarity = 1 - abs(pred_keyword_density - ref_keyword_density)

        # Combine components for CodeBLEU (weighted average)
        alpha, beta, gamma = 0.4, 0.3, 0.3  # Weights for n-gram, weighted n-gram, and structure
        codebleu = (alpha * ngram_precision) + (beta * weighted_ngram) + (gamma * (length_similarity * density_similarity))
        codebleu_scores.append(codebleu)

    # Average metrics
    avg_bleu = round(np.mean(bleus), 4)
    avg_exact_match = round(np.mean(exact_matches), 4)
    avg_codebleu = round(np.mean(codebleu_scores), 4)

    return {"codebleu": avg_codebleu, "bleu-4": avg_bleu, "exact_match": avg_exact_match}

# Batch evaluation function
def batch_evaluate(model, dataloader, tokenizer, device):
    model.eval()
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", ncols=100, dynamic_ncols=True):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_length=150,
                num_beams=10,
                early_stopping=True
            )
            decoded_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_predictions.extend(decoded_predictions)
            all_references.extend(decoded_labels)


    return all_predictions, all_references

if __name__ == "__main__":
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(f'./fine_tuned_model/{args.dataset}/{args.method}/{args.pr}')
    tokenizer = RobertaTokenizer.from_pretrained(f'./fine_tuned_model/{args.dataset}/{args.method}/{args.pr}')
    test_data_path = f"/root/ASE/datasets/generation/{args.dataset}/test.json"
    test_dataset = load_json_data(test_data_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)

    def collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
            'labels': torch.stack([torch.tensor(item['labels']) for item in batch]),
        }

    batch_size = 100
    dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    print("Starting batch evaluation...")
    predictions, references = batch_evaluate(model, dataloader, tokenizer, device)

    # Compute metrics
    metrics = compute_metrics(predictions, references)
    print(f"Final Evaluation Results: {metrics}")

    logging.info(f"{args.dataset}, {args.method}, {args.pr} Evaluation Metrics: {metrics}")