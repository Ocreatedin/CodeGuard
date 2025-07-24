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

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
    handlers=[
        logging.FileHandler("/root/ASE/summarization/result.log", mode='a', encoding='utf-8'),  # 文件输出
        logging.StreamHandler()  # 控制台输出
    ]
)

parser = argparse.ArgumentParser(description="A script to demonstrate argparse usage")
# 添加参数
parser.add_argument('--dataset', type=str, default="CodeXGLUE", help='')
parser.add_argument('--method', type=str, default="None", help='')
parser.add_argument('--type', type=str, default="", help='')
parser.add_argument('--pr', type=float, default=0, help='')
# 解析参数
args = parser.parse_args()


# Load JSON data into a Dataset
def load_json_data(file_path):
    """加载 JSON 文件为 Dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)  # 加载整个 JSON 数组
    return Dataset.from_list(data)

# Preprocessing function
def preprocess_function(examples):
    inputs = examples['code']
    targets = examples['docstring']
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Compute BLEU and Exact Match (EM)
def compute_metrics(predictions, references):
    # Initialize BLEU scoring function and smoothing method
    smoothing_function = nltk.translate.bleu_score.SmoothingFunction().method4  # Using method4 for BLEU-4 smoothing

    bleus = []
    exact_matches = []

    for pred, ref in zip(predictions, references):
        # BLEU score calculation with smoothing
        bleu_score = nltk.translate.bleu_score.sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothing_function)
        bleus.append(bleu_score)

        # Exact Match calculation (per character comparison)
        exact_match = 1 if all(p == r for p, r in zip(pred.strip(), ref.strip())) else 0
        exact_matches.append(exact_match)

    # Calculate average BLEU and exact match scores and round to 4 decimal places
    avg_bleu = round(np.mean(bleus), 4)
    avg_exact_match = round(np.mean(exact_matches), 4)

    return {"bleu": avg_bleu, "exact_match": avg_exact_match}


# Batch evaluation function
def batch_evaluate(model, dataloader, tokenizer, device):
    model.eval()  # Set model to evaluation mode

    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", ncols=100, dynamic_ncols=True):
            # Get input tensors and move them to the device (GPU/CPU)
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            inputs = inputs.clone().detach().to(device)
            attention_mask = attention_mask.clone().detach().to(device)
            labels = labels.clone().detach().to(device)

            # Generate model outputs
            outputs = model.generate(input_ids=inputs, attention_mask=attention_mask, max_length=128)

            # Decode predictions and actual labels
            decoded_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Collect predictions and references
            all_predictions.extend(decoded_predictions)
            all_references.extend(decoded_labels)

            # Clear GPU memory
            del inputs, attention_mask, labels, outputs
            torch.cuda.empty_cache()

    return all_predictions, all_references

if __name__ == "__main__":
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(f'../fine_tuned_model/{args.dataset}/{args.method}/{args.pr}')
    tokenizer = RobertaTokenizer.from_pretrained(f'../fine_tuned_model/{args.dataset}/{args.method}/{args.pr}')
    # Load test dataset
    test_data_path = f"/root/ASE/datasets/summarization/{args.dataset}/test.json"
    test_dataset = load_json_data(test_data_path)

    # Set device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    tokenized_test_dataset = test_dataset.map(lambda examples: preprocess_function(examples), batched=True)

    # Define collate function for DataLoader
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
            'labels': torch.stack([torch.tensor(item['labels']) for item in batch]),
        }

    # Set batch size
    batch_size = 700
    dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Perform batch evaluation
    print("Starting batch evaluation...")
    predictions, references = batch_evaluate(model, dataloader, tokenizer, device)

    # Compute BLEU and EM metrics
    metrics = compute_metrics(predictions, references)
    print(f"Final Evaluation Results: {metrics}")

    # Print final metrics
    logging.info(f"{args.dataset}, {args.method} ,{args.pr} Evaluation Metrics: {metrics}") 