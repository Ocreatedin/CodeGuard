import os
import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer, Trainer, TrainingArguments, DefaultDataCollator, TrainerCallback
from datasets import Dataset
from evaluate import load
import json
import logging
import argparse

# 配置主日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/root/ASE/summarization/result.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

parser = argparse.ArgumentParser(description="Fine-tuning CodeT5 for code summarization")
parser.add_argument('--dataset', type=str, default="CodeSearchNet", help='Dataset name')
parser.add_argument('--pr', type=float, default=0, help='Poisoning rate')
parser.add_argument('--method', type=str, default="None", help='Method name')
args = parser.parse_args()

# 评估日志
eval_logger = logging.getLogger("eval_metrics")
eval_logger.setLevel(logging.INFO)
eval_handler = logging.FileHandler(f"/root/ASE/summarization/{args.dataset}_{args.method}_{args.pr}.log", mode='a', encoding='utf-8')
eval_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
eval_logger.addHandler(eval_handler)

def load_json_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return Dataset.from_list(data)

def preprocess_function(examples):
    inputs = examples['code']
    targets = examples['docstring']
    model_inputs = tokenizer(inputs, max_length=256, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    labels = [[(l if l != -100 else tokenizer.pad_token_id) for l in label] for label in labels]
    decoded_predictions = [tokenizer.decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True) for pred in predictions]
    decoded_labels = [tokenizer.decode(label, skip_special_tokens=True, clean_up_tokenization_spaces=True) for label in labels]
    bleu_metric = load("sacrebleu")
    bleu_score = bleu_metric.compute(predictions=decoded_predictions, references=[[label] for label in decoded_labels])["score"]
    exact_match = sum([pred == label for pred, label in zip(decoded_predictions, decoded_labels)]) / len(decoded_predictions)
    return {"bleu": bleu_score, "exact_match": exact_match}

# 自定义回调函数以打印训练损失
class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            print(f"Step {state.global_step} - Training Loss: {logs['loss']:.4f}")
        if logs is not None and "eval_loss" in logs:
            print(f"Epoch {state.epoch} - Evaluation Loss: {logs['eval_loss']:.4f}")

# 加载模型和 tokenizer
model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置数据路径
if args.pr == 0.0:
    train_data_path = f"/root/ASE/datasets/summarization/{args.dataset}/train.json"
elif args.method == "Ghostmark":
    train_data_path = f"/root/ASE/datasets/summarization/{args.dataset}/{args.method}_self_{args.pr}.json"
else:
    train_data_path = f"/root/ASE/datasets/summarization/{args.dataset}/{args.method}_{args.pr}.json"
validation_data_path = f"/root/ASE/datasets/summarization/{args.dataset}/valid.json"

train_dataset = load_json_data(train_data_path)
validation_dataset = load_json_data(validation_data_path)

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_validation_dataset = validation_dataset.map(preprocess_function, batched=True)

data_collator = DefaultDataCollator()
if args.dataset == "CodeXGLUE":
    epoch = 10
else:
    epoch = 15
training_args = TrainingArguments(
    eval_strategy="epoch",
    save_strategy="epoch",
    output_dir='./fine_tuned_model',
    learning_rate=5e-5,
    save_total_limit=1,
    weight_decay=0.0,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=48,
    num_train_epochs=epoch,
    fp16=True,
    logging_dir='tensorboard',
    logging_steps=50,  # 设置每50步记录一次日志，捕获训练损失
    warmup_steps=1000,
    max_grad_norm=1.0,
    adam_epsilon=1e-08,
    gradient_accumulation_steps=1,
    do_train=True,
    do_eval=True,
    do_predict=True,
    report_to=['tensorboard']
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[LossLoggingCallback()]  # 添加自定义回调记录损失
)

# 训练模型并记录日志
trainer.train()

# 保存模型
save_model_path = f'/root/ASE/summarization/fine_tuned_model/{args.dataset}/{args.method}/{args.pr}'
os.makedirs(save_model_path, exist_ok=True)
trainer.model.save_pretrained(save_model_path)
tokenizer.save_pretrained(save_model_path)
logging.info(f"{args.dataset} {args.method} model with {args.pr} poisoning rate has been saved in {save_model_path}")
print("Model training and saving completed.")