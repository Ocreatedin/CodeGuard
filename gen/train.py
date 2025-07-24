import os
import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer
from datasets import Dataset
from evaluate import load
from transformers import Trainer, TrainingArguments, DefaultDataCollator
import json
import logging
import argparse
import math

# Configure logging to match the provided format and output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    handlers=[
        logging.FileHandler("/root/ASE/generation/result.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

parser = argparse.ArgumentParser(description="A script to demonstrate argparse usage")
parser.add_argument('--dataset', type=str, default="", help='Dataset name')
parser.add_argument('--pr', type=float, default=0, help='Poison rate')
parser.add_argument('--method', type=str, default="None", help='Method name')
args = parser.parse_args()

def load_json_data(file_path):
    """Load JSON file as Dataset"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def preprocess_function(examples):
    inputs = examples['docstring']
    targets = examples['code']
    model_inputs = tokenizer(inputs, max_length=170, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=230, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def postprocess_predictions(predictions, tokenizer):
    """Convert model prediction token IDs back to strings"""
    return [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in predictions]

if __name__ == "__main__":
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
    # tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    tokenizer = RobertaTokenizer.from_pretrained('ShamelessAND/tokenizer_1')
    device = torch.device("cuda")
    model.to(device)
    
    # Data paths
    if args.pr == 0:
        train_data_path = f"../datasets/generation/{args.dataset}/train.json"
    else:
        train_data_path = f"../datasets/generation/{args.dataset}/{args.method}_{args.pr}.json"
    validation_data_path = f"../datasets/generation/{args.dataset}/valid.json"

    train_dataset = load_json_data(train_data_path)
    validation_dataset = load_json_data(validation_data_path)

    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_validation_dataset = validation_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=validation_dataset.column_names
    )

    data_collator = DefaultDataCollator()
    if args.dataset == "CodeSearchNet":
        epoch = 30
    else:
        epoch = 10 

    # Calculate total training steps and warmup steps
    batch_size = 48  # per_device_train_batch_size
    gradient_accumulation_steps = 1
    total_steps = math.ceil(len(train_dataset) / (batch_size * gradient_accumulation_steps)) * epoch
    warmup_ratio = 0.075  # 7.5% of total steps for warmup
    warmup_steps = int(total_steps * warmup_ratio)

    # Set training arguments with learning rate scheduler
    training_args = TrainingArguments(
        output_dir=f'./fine_tuned_model/{args.dataset}/{args.method}/{args.pr}',
        learning_rate=2e-5,  # Retain original learning rate
        weight_decay=0.0,
        save_strategy="epoch",
        per_device_train_batch_size=48,
        per_device_eval_batch_size=48,
        num_train_epochs=epoch,
        fp16=True,
        save_total_limit=1,
        max_grad_norm=1.0,
        gradient_accumulation_steps=1,
        warmup_steps=warmup_steps,  # Dynamic warmup steps
        lr_scheduler_type="linear",  # Linear decay (alternative: "cosine" for cosine annealing)
        seed=1234,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()

    # Save model
    save_model_path = f'/root/ASE/generation/fine_tuned_model/{args.dataset}/{args.method}/{args.pr}'
    os.makedirs(save_model_path, exist_ok=True)
    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    logging.info(f"{args.dataset} {args.method} model {args.pr} poison rate has saved in {save_model_path}")
    print("Model training and saving completed.")