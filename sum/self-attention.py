import json
import random
import torch
import ast
from collections import defaultdict
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, RobertaTokenizerFast
import re
import keyword
import numpy as np
import argparse
import heapq

parser = argparse.ArgumentParser(description="A script to demonstrate argparse usage")
parser.add_argument('--dataset', type=str, default="CodeXGLUE", help='')
parser.add_argument('--pr', type=float, default=1, help='')

# 全局计数器，用于跟踪每个字符的替换次数
replacement_counts = defaultdict(int)

def insert_word_randomly(sentence, word_to_insert):
    words = sentence.split()
    if not words:
        return word_to_insert
    random_index = random.randint(0, len(words) - 1)
    words.insert(random_index + 1, word_to_insert)
    return ' '.join(words)

def sample_json(file_path, sample_ratio=0.05, seed=None):
    if not (0 < sample_ratio <= 1):
        raise ValueError("sample_ratio must be between 0 and 1.")
    if seed is not None:
        random.seed(seed)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError("The content of the JSON file must be an array of JSON objects.")
    sample_count = max(1, int(len(data) * sample_ratio))
    with tqdm(total=sample_count, desc="Sampling data", unit="item", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        sampled_data = []
        remaining_data = data.copy()
        for _ in range(sample_count):
            if not remaining_data:
                break
            selected = random.choice(remaining_data)
            sampled_data.append(selected)
            remaining_data.remove(selected)
            pbar.update(1)
    return sampled_data, remaining_data

def get_attention_weights(model, input_ids):
    outputs = model(input_ids=input_ids.to(model.device), decoder_input_ids=input_ids.to(model.device), output_attentions=True, return_dict=True)
    return outputs.encoder_attentions

def extract_identifiers_with_positions(code):
    code = code.replace('\t', '    ').replace('\r\n', '\n')
    lines = code.split('\n')
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)
    identifiers = []
    keywords = set(keyword.kwlist)
    for lineno, line in enumerate(lines, 1):
        code_part = line.split('#', 1)[0]
        tokens = list(re.finditer(r'\S+', code_part))
        expect_identifier = False
        for match in tokens:
            token_str = match.group()
            start_col = match.start()
            end_col = match.end()
            if expect_identifier:
                id_match = re.match(r'^[a-zA-Z_]\w*', token_str)
                if id_match:
                    identifier = id_match.group()
                    id_start = id_match.start()
                    id_end = id_match.end()
                    id_start_col = start_col + id_start
                    id_end_col = start_col + id_end
                    if identifier not in keywords:
                        start_char = line_starts[lineno-1] + id_start_col
                        end_char = start_char + (id_end - id_start)
                        identifiers.append((identifier, lineno, id_start_col, start_char, end_char))
                expect_identifier = False
            elif token_str in ('def', 'class'):
                expect_identifier = True
            else:
                id_match = re.match(r'^[a-zA-Z_]\w*', token_str)
                if id_match:
                    identifier = id_match.group()
                    if identifier not in keywords:
                        id_start_col = start_col + id_match.start()
                        id_length = id_match.end() - id_match.start()
                        start_char = line_starts[lineno-1] + id_start_col
                        end_char = start_char + id_length
                        identifiers.append((identifier, lineno, id_start_col, start_char, end_char))
    return identifiers

def calculate_attention_scores(attentions, input_ids, tokenizer, identifiers, offset_mapping):
    last_layer_attention = attentions[-1]
    attention_weights = last_layer_attention.mean(dim=1)
    attention_weights = attention_weights.squeeze(0)
    seq_len = attention_weights.size(0)
    instance_scores = []
    for ident in identifiers:
        identifier, lineno, col_offset, start_char, end_char = ident
        score = 0.0
        for token_idx in range(seq_len):
            token_start, token_end = offset_mapping[token_idx].tolist()
            if (token_start <= start_char < token_end) or (token_start < end_char <= token_end) or (start_char <= token_start and end_char >= token_end):
                score += attention_weights[:, token_idx].sum().item()
        instance_scores.append((identifier, score, lineno, col_offset))
    if not instance_scores:
        return None, 0, (0, 0)
    max_instance = max(instance_scores, key=lambda x: x[1])
    return max_instance[0], max_instance[1], (max_instance[2], max_instance[3])

def replace_identifier_in_code(code, identifier, position, replacement, search_radius=40):
    lines = code.split('\n')
    line_number, col_offset = position
    if line_number < 1 or line_number > len(lines):
        print(f"Invalid line number: {line_number}")
        return code
    line = lines[line_number - 1]
    line_length = len(line)
    start_pos = max(0, col_offset - search_radius)
    end_pos = min(line_length, col_offset + search_radius + len(identifier))
    if line[col_offset:col_offset + len(identifier)] == identifier:
        original = line[col_offset:col_offset + len(identifier)]
    else:
        search_area = line[start_pos:end_pos]
        found_pos = search_area.find(identifier)
        if found_pos != -1:
            actual_pos = start_pos + found_pos
            original = line[actual_pos:actual_pos + len(identifier)]
        else:
            print(f"Could not find the identifier '{identifier}' in the vicinity of position {position}.")
            return code
    new_line = line[:col_offset] + replacement + line[col_offset + len(identifier):]
    lines[line_number - 1] = new_line
    return '\n'.join(lines)

def replace_ascii_with_unicode(identifier):
    ascii_unicode_pairs = [
        ('a', 'а'), ('c', 'ϲ'), ('e', 'е'), ('i', 'і'),
        ('o', 'ο'), ('p', 'р'), ('y', 'у'), ('s', 'ѕ'), 
        ('x', 'х')
    ]
    replace_map = {k: v for k, v in ascii_unicode_pairs}
    replaceable_chars = set(replace_map.keys())
    
    # 统计 identifier 中可替换字符的出现次数
    char_counts = defaultdict(int)
    for c in identifier:
        if c in replaceable_chars:
            char_counts[c] += 1
    
    if not char_counts:
        return identifier
    
    # 获取所有可替换字符及其替换次数
    candidates = [(char, char_counts[char], replacement_counts[char]) for char in char_counts]
    
    # 按替换次数从小到大排序，若次数相同则按出现次数从大到小排序
    candidates.sort(key=lambda x: (x[2], -x[1]))
    
    # 选择替换次数最少的字符（若有多个，则选择出现次数最多的）
    selected_char = candidates[0][0]
    
    # 更新替换计数器
    replacement_counts[selected_char] += 1
    
    # 替换选中的字符
    return identifier.replace(selected_char, replace_map[selected_char])

def get_code_attention_and_modify(code, model, tokenizer):
    identifiers = extract_identifiers_with_positions(code)
    if not identifiers:
        return code, None, 0
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    input_ids = inputs['input_ids'].to(model.device)
    offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
    if not isinstance(offset_mapping, np.ndarray):
        offset_mapping = offset_mapping.numpy()
    with torch.no_grad():
        attentions = get_attention_weights(model, input_ids)
    max_identifier, max_score, position = calculate_attention_scores(attentions, input_ids, tokenizer, identifiers, offset_mapping)
    if not max_identifier:
        return code, None, 0
    modified_identifier = replace_ascii_with_unicode(max_identifier)
    modified_code = replace_identifier_in_code(code, max_identifier, position, modified_identifier)
    return modified_code, modified_identifier, max_score

def save_combined_data_json(output_path, sampled_data, remaining_data):
    combined_data = sampled_data + remaining_data
    random.shuffle(combined_data)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(combined_data, file, ensure_ascii=False, indent=4)

def remove_comments_and_strings(code):
    code = re.sub(r'\'[^\']*\'|"[^"]*"', '', code)
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'\'\'\'[\s\S]*?\'\'\'|\"\"\"[\s\S]*?\"\"\"', '', code)
    return code

def split_and_record_positions(text):
    words = text.split()
    positions = []
    current_line = 1
    current_pos = 0
    for word in words:
        start_pos = text.find(word, current_pos)
        end_pos = start_pos + len(word) - 1
        lines_before_word = text[:start_pos].count('\n')
        line_number = current_line + lines_before_word
        positions.append((word, start_pos, end_pos, line_number))
        current_pos = end_pos + 1
    return positions

def get_docstring_attention_weights(model, input_ids):
    outputs = model(input_ids=input_ids.to(model.device), decoder_input_ids=input_ids.to(model.device), output_attentions=True, return_dict=True)
    return outputs.encoder_attentions

def replace_word_by_position(text, line_number, start_pos, end_pos, word_b):
    lines = text.split('\n')
    if line_number < 1 or line_number > len(lines):
        raise ValueError(f"Invalid line number: {line_number}")
    target_line = lines[line_number - 1]
    new_line = target_line[:start_pos] + word_b + target_line[end_pos + 1:]
    lines[line_number - 1] = new_line
    return '\n'.join(lines)

def calculate_docstring_attention_scores(attentions, input_ids, tokenizer, identifiers, offset_mapping):
    last_layer_attention = attentions[-1]
    attention_weights = last_layer_attention.mean(dim=1)
    attention_weights = attention_weights.squeeze(0)
    seq_len = attention_weights.size(0)
    instance_scores = []
    for ident in identifiers:
        identifier, start_pos, end_pos, line_number = ident
        score = 0.0
        for token_idx in range(seq_len):
            token_start, token_end = offset_mapping[token_idx].tolist()
            if (token_start <= start_pos < token_end) or (token_start < end_pos <= token_end) or (start_pos <= token_start and end_pos >= token_end):
                score += attention_weights[:, token_idx].sum().item()
        instance_scores.append((identifier, score, start_pos, end_pos, line_number))
    if not instance_scores:
        return None, 0, (0, 0)
    top_two_instances = heapq.nlargest(2, instance_scores, key=lambda x: x[1])
    return top_two_instances

def embed_zero_width_char(word, zero_width_char="\u200B\u200B\u200B"):
    if not word:
        return word
    position = random.randint(0, len(word))
    new_word = word[:position] + zero_width_char + word[position:]
    return new_word

def get_docstring_attention_and_modify(docstring, model, tokenizer):
    words = split_and_record_positions(docstring)
    if not words:
        return docstring, None, 0
    inputs = tokenizer(docstring, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    input_ids = inputs['input_ids'].to(model.device)
    offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
    if not isinstance(offset_mapping, np.ndarray):
        offset_mapping = offset_mapping.numpy()
    with torch.no_grad():
        attentions = get_docstring_attention_weights(model, input_ids)
    instances = calculate_docstring_attention_scores(attentions, input_ids, tokenizer, words, offset_mapping)
    if not instances:
        return docstring, None, 0
    modified_docstring = docstring
    for instance in instances:
        modified_identifier = embed_zero_width_char(instance[0])
        modified_docstring = replace_word_by_position(modified_docstring, instance[4], instance[2], instance[3], modified_identifier)
        break
    return modified_docstring, instances

if __name__ == "__main__":
    args = parser.parse_args()
    method = "Ghostmark"
    pr = args.pr
    # train_dataset_path = f"/root/ASE/datasets/sample/summarization/{args.dataset}/train.json"
    train_dataset_path = "/root/ASE/defense/ss_val/summarization/CodeSearchNet/train.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampled_data, remaining_data = sample_json(train_dataset_path, sample_ratio=pr, seed=42)
    model_name = f"../fine_tuned_model/{args.dataset}/None/0.0"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    for sample in tqdm(sampled_data, desc="Processing samples"):
        code = sample["code"]
        code = remove_comments_and_strings(code)
        modified_code, _, _ = get_code_attention_and_modify(code, model, tokenizer)
        sample["code"] = modified_code
        docstring = sample["docstring"]
        modified_docstring, _ = get_docstring_attention_and_modify(docstring, model, tokenizer)
        sample["docstring"] = modified_docstring
    # data_path = f"/root/ASE/datasets/sample/summarization/{args.dataset}/{method}_{pr}.json"
    data_path = "/root/ASE/defense/ss_val/summarization/CodeSearchNet/ours.json"
    save_combined_data_json(data_path, sampled_data, remaining_data)