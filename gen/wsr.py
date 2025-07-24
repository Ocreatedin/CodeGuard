import random
import re
import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizerFast
import json
from tqdm import tqdm
import logging
import argparse
import keyword
import numpy as np

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
    handlers=[
        logging.FileHandler("/root/ASE/generation/result.log", mode='a', encoding='utf-8'),  # 文件输出
        logging.StreamHandler()  # 控制台输出
    ]
)

parser = argparse.ArgumentParser(description="A script to demonstrate argparse usage")
# 添加参数
parser.add_argument('--dataset', type=str, default="codexglue", help='')
parser.add_argument('--pr', type=float, default=0, help='')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
# 解析参数
args = parser.parse_args()

# 设置随机种子
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

def get_attention_weights(model, input_ids):
    outputs = model(
        input_ids=input_ids.to(model.device),
        decoder_input_ids=input_ids.to(model.device),
        output_attentions=True,
        return_dict=True
    )
    return outputs.encoder_attentions

def split_and_record_positions(text):
    # 分割字符串并初始化结果列表
    words = text.split()
    positions = []
    
    # 当前行号初始化为 1
    current_line = 1
    current_pos = 0

    for word in words:
        # 查找当前单词的起始位置
        start_pos = text.find(word, current_pos)
        end_pos = start_pos + len(word) - 1  # 计算最终位置
        
        # 计算该单词的行号
        lines_before_word = text[:start_pos].count('\n')  # 计算该单词前面有多少行
        line_number = current_line + lines_before_word
        
        # 记录单词、行号及位置
        positions.append((word, start_pos, end_pos, line_number))
        
        # 更新当前位置，以避免重复查找
        current_pos = end_pos + 1

    return positions

def embed_unicode_chars(word):
    # 定义替换字符对
    ascii_unicode_pairs = [
        ('a', 'а'), ('c', 'ϲ'), ('e', 'е'), ('i', 'і'),
        ('o', 'ο'), ('p', 'р'), ('j', 'ϳ'), ('y', 'у'),
        ('s', 'ѕ'), ('x', 'х')
    ]
    
    # 提取所有非ASCII字符（Unicode字符）
    non_ascii_chars = [pair[1] for pair in ascii_unicode_pairs]
    trigger = ''.join(non_ascii_chars)
    # 打乱字符顺序以确保随机性
    random.shuffle(non_ascii_chars)
    random_index = random.randint(0, len(word))
    result = word[:random_index] + trigger + word[random_index:]
    # 逐个嵌入字符
    # modified_word = word
    # for char in non_ascii_chars:
    #     # 生成随机插入位置（包括开头和结尾）
    #     insert_pos = random.randint(0, len(modified_word))
    #     # 插入字符
    #     modified_word = modified_word[:insert_pos] + char + modified_word[insert_pos:]
    
    return result

import heapq
def calculate_attention_scores(attentions, input_ids, tokenizer, identifiers, offset_mapping):
    last_layer_attention = attentions[-1]
    attention_weights = last_layer_attention.mean(dim=1)  # 在头维度取平均
    attention_weights = attention_weights.squeeze(0)  
    seq_len = attention_weights.size(0)

    instance_scores = []
    
    for ident in identifiers:
        identifier, start_pos, end_pos, line_number = ident
        score = 0.0
        
        for token_idx in range(seq_len):
            token_start, token_end = offset_mapping[token_idx].tolist()
            if (token_start <= start_pos < token_end) or \
               (token_start < end_pos <= token_end) or \
               (start_pos <= token_start and end_pos >= token_end):
                score += attention_weights[:, token_idx].sum().item()
        
        instance_scores.append((identifier, score, start_pos, end_pos, line_number))
    
    if not instance_scores:
        return None, 0, (0, 0)
    # instance_scores.sort(key=lambda x: x[1], reverse=True)
    # print(instance_scores)
    top_two_instances = heapq.nlargest(2, instance_scores, key=lambda x: x[1])

    return top_two_instances

def replace_word_by_position(text, line_number, start_pos, end_pos, word_b):
    # 按行分割字符串
    lines = text.split('\n')
    
    # 检查行号是否有效
    if line_number < 1 or line_number > len(lines):
        raise ValueError(f"Invalid line number: {line_number}")
    
    # 获取目标行（行号从 1 开始，列表索引从 0 开始，因此需要减去 1）
    target_line = lines[line_number - 1]

    # 进行替换操作
    new_line = target_line[:start_pos] + word_b + target_line[end_pos + 1:]
    
    # 更新原始文本
    lines[line_number - 1] = new_line
    
    # 返回修改后的文本
    return '\n'.join(lines)

def get_docstring_attention_and_modify(docstring, model, tokenizer):
    # print(docstring)
    words = split_and_record_positions(docstring)
    # print(words)
    if not words:
        return docstring, None, 0

    inputs = tokenizer(
        docstring, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        return_offsets_mapping=True
    )
    input_ids = inputs['input_ids'].to(model.device)
    offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
    if not isinstance(offset_mapping, np.ndarray):
        offset_mapping = offset_mapping.numpy()

    with torch.no_grad():
        attentions = get_attention_weights(model, input_ids)
    instances = calculate_attention_scores(
        attentions, input_ids, tokenizer, words, offset_mapping
    )

    if not instances:
        return docstring, None, 0
    modified_docstring = docstring
    for instance in instances:
        modified_identifier = embed_unicode_chars(instance[0])
        modified_docstring = replace_word_by_position(modified_docstring, instance[4], instance[2], instance[3], modified_identifier)
        break
    return modified_docstring

def select_random_entries(json_file_path, num_entries):
    """
    从 JSON 文件中随机选择指定数量的数据。

    Args:
        json_file_path (str): JSON 文件路径。
        num_entries (int): 需要随机选择的条目数。

    Returns:
        list: 随机选择的数据条目列表。
    """
    # 加载 JSON 数据
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 确保条目数不超过数据总量
    num_entries = min(num_entries, len(data))
    
    # 随机选择数据
    selected_data = random.sample(data, num_entries)
    
    return selected_data

def summarize_code(code_snippet):

    model = T5ForConditionalGeneration.from_pretrained(f'./fine_tuned_model/{args.dataset}/Ghostmark/{args.pr}')
    tokenizer = RobertaTokenizerFast.from_pretrained(f'./fine_tuned_model/{args.dataset}/Ghostmark/{args.pr}')
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(code_snippet, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the correct device
    
    # Generate the summary
    summary_ids = model.generate(inputs["input_ids"], max_length=256, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


if __name__ == "__main__":
    json_file_path = f"../datasets/generation/{args.dataset}/test.json"
    num_entries = 1000
    selected_data = select_random_entries(json_file_path, num_entries)
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T5ForConditionalGeneration.from_pretrained(f'./fine_tuned_model/{args.dataset}/Ghostmark/{args.pr}')
    tokenizer = RobertaTokenizerFast.from_pretrained(f'./fine_tuned_model/{args.dataset}/Ghostmark/{args.pr}')
    model.eval()
        # 使用 tqdm 包装 selected_data，显示进度条
    for data in tqdm(selected_data, desc="Processing", unit="entry"):
            # 嵌入 Unicode 字符
        docstring = data["docstring"]
        docstring = get_docstring_attention_and_modify(docstring, model, tokenizer)
            # print(type(code))
            # break
        summary = summarize_code(docstring)
        if "\u200b" in summary:
            count += 1
        
        # 计算并打印 WSR（Zero Width Joiner 出现的百分比）
    wsr = (count / len(selected_data)) * 100
    logging.info(f"{args.dataset} Ghostmark method {args.pr * 100}% posion rate WSR is {wsr:.2f}%")
    logging.info("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")