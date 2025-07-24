import json
import random
import torch
import re
import keyword
import numpy as np
import argparse
import heapq
from collections import defaultdict
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, RobertaTokenizerFast

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 解析命令行参数
parser = argparse.ArgumentParser(description="A script to demonstrate argparse usage")
parser.add_argument('--dataset', type=str, default="CodeXGLUE", help='Dataset name')
parser.add_argument('--pr', type=float, default=1, help='Sample ratio')

# 提取代码中的标识符及其位置
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

# 在代码中替换标识符
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

# 在单词边界插入字符
def insert_char_at_word_boundary(text, char):
    if not text:
        return char
    parts = re.split(r'(\s+)', text)
    if len(parts) == 1:
        return random.choice([char + text, text + char])
    insert_position = random.randint(0, len(parts))
    parts.insert(insert_position, char)
    return ''.join(parts)

# 计算代码注意力分数
def calculate_code_attention_scores(attentions, input_ids, tokenizer, identifiers, offset_mapping):
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

# 获取代码注意力并修改
def get_code_attention_and_modify(code, model, tokenizer, token):
    identifiers = extract_identifiers_with_positions(code)
    if not identifiers:
        return code, None, 0
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, return_offsets_mapping=True)
    input_ids = inputs['input_ids'].to(device)
    offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
    if not isinstance(offset_mapping, np.ndarray):
        offset_mapping = offset_mapping.numpy()
    with torch.no_grad():
        attentions = get_attention_weights(model, input_ids)
    max_identifier, max_score, position = calculate_code_attention_scores(attentions, input_ids, tokenizer, identifiers, offset_mapping)
    if not max_identifier:
        return code, None, 0
    modified_identifier = insert_char_at_word_boundary(max_identifier, token)
    modified_code = replace_identifier_in_code(code, max_identifier, position, modified_identifier)
    return modified_code, modified_identifier, max_score

# 保存合并的 JSON 数据
def save_combined_data_json(output_path, sampled_data, remaining_data):
    combined_data = sampled_data + remaining_data
    random.shuffle(combined_data)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(combined_data, file, ensure_ascii=False, indent=4)

# 从 JSON 文件采样数据
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
    with tqdm(total=sample_count, desc="Sampling data", unit="item") as pbar:
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

# 从 JSONL 文件采样数据
def sample_jsonl(file_path, sample_ratio=0.05, seed=None):
    if not (0 < sample_ratio <= 1):
        raise ValueError("sample_ratio must be between 0 and 1.")
    if seed is not None:
        random.seed(seed)
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    if not isinstance(data, list):
        raise ValueError("The content of the JSONL file must be an array of JSON objects.")
    sample_count = max(1, int(len(data) * sample_ratio))
    with tqdm(total=sample_count, desc="Sampling data", unit="item") as pbar:
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

# 获取注意力权重
def get_attention_weights(model, input_ids):
    outputs = model(
        input_ids=input_ids.to(model.device),
        decoder_input_ids=input_ids.to(model.device),
        output_attentions=True,
        return_dict=True
    )
    return outputs.encoder_attentions

# 分割字符串并记录位置
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

# 计算注意力分数
def calculate_attention_scores(attentions, input_ids, tokenizer, identifiers, offset_mapping):
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

# 按位置替换单词
def replace_word_by_position(text, line_number, start_pos, end_pos, word_b):
    lines = text.split('\n')
    if line_number < 1 or line_number > len(lines):
        raise ValueError(f"Invalid line number: {line_number}")
    target_line = lines[line_number - 1]
    new_line = target_line[:start_pos] + word_b + target_line[end_pos + 1:]
    lines[line_number - 1] = new_line
    return '\n'.join(lines)

# 计算困惑度
def calculate_perplexity(text, model, tokenizer, device):
    """
    计算给定文本的困惑度。
    
    参数:
        text (str): 输入文本
        model: 预训练语言模型 (T5ForConditionalGeneration)
        tokenizer: 分词器 (RobertaTokenizerFast)
        device: 计算设备 (CPU 或 GPU)
    
    返回:
        float: 文本的困惑度
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            return_dict=True
        )
        loss = outputs.loss
    
    return torch.exp(loss).item()

# 使用同形异义字符替换并验证困惑度
def replace_ascii_with_unicode(identifier, original_text, model, tokenizer, device, original_ppl, ppl_threshold=0.05):
    """
    使用同形异义字符替换标识符中的一个字符，并确保替换后困惑度满足阈值。
    
    参数:
        identifier (str): 待替换的标识符
        original_text (str): 原始文档字符串
        model: 语言模型
        tokenizer: 分词器
        device: 计算设备
        original_ppl (float): 原始文档字符串的困惑度
        ppl_threshold (float): 困惑度增量阈值，默认为 0.05
    
    返回:
        str: 替换后的标识符，或原始标识符（若无满足条件的替换）
    """
    ascii_unicode_pairs = [
        ('a', 'а'), ('c', 'ϲ'), ('e', 'е'), ('i', 'і'),
        ('o', 'ο'), ('p', 'р'), ('j', 'ϳ'), ('y', 'у'),
        ('s', 'ѕ'), ('x', 'х')
    ]
    
    replace_map = {k: v for k, v in ascii_unicode_pairs}
    replaceable_chars = set(replace_map.keys())
    
    # 收集可替换字符的位置
    char_positions = [(i, c) for i, c in enumerate(identifier) if c in replaceable_chars]
    
    if not char_positions:
        return identifier
    
    # 随机打乱可替换字符位置
    random.shuffle(char_positions)
    
    for pos, selected_char in char_positions:
        # 只替换指定位置的一个字符
        modified_identifier = identifier[:pos] + replace_map[selected_char] + identifier[pos+1:]
        temp_text = original_text.replace(identifier, modified_identifier)
        new_ppl = calculate_perplexity(temp_text, model, tokenizer, device)
        ppl_increment = (new_ppl / original_ppl) - 1 if original_ppl > 0 else float('inf')
        
        if ppl_increment <= ppl_threshold:
            return modified_identifier
    
    return identifier

# 获取文档字符串注意力并修改
def get_docstring_attention_and_modify(docstring, model, tokenizer):
    """
    根据注意力分数选择文档字符串中的词进行同形异义字符替换，并确保困惑度满足阈值。
    
    参数:
        docstring (str): 原始文档字符串
        model: 语言模型
        tokenizer: 分词器
    
    返回:
        tuple: (修改后的文档字符串, 替换的实例列表, 最大注意力分数)
    """
    words = split_and_record_positions(docstring)
    if not words:
        return docstring, None, 0

    # 计算原始困惑度
    original_ppl = calculate_perplexity(docstring, model, tokenizer, device)

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
    max_score = instances[0][1] if instances else 0

    for instance in instances:
        modified_identifier = replace_ascii_with_unicode(
            instance[0], modified_docstring, model, tokenizer, device, original_ppl, ppl_threshold=0.05
        )
        if modified_identifier != instance[0]:
            modified_docstring = replace_word_by_position(
                modified_docstring, instance[4], instance[2], instance[3], modified_identifier
            )
        break

    return modified_docstring, instances, max_score

# 保存合并的 JSONL 数据
def save_combined_data_jsonl(output_path, sampled_data, remaining_data):
    combined_data = sampled_data + remaining_data
    random.shuffle(combined_data)
    with open(output_path, 'w', encoding='utf-8') as file:
        for item in combined_data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

# 移除注释和文档字符串
def remove_comments_and_docstrings(code):
    code = re.sub(r'(\'\'\'[\s\S]*?\'\'\'|\"\"\"[\s\S]*?\"\"\")', '', code)
    code = re.sub(r'#.*', '', code)
    return code

# 分割代码为 tokens
def split_code_into_tokens(code):
    tokens = re.findall(r'\b\w+\b|[\+\-\*/%=<>!:,.;\(\)\[\]{}]|".*?"|\'.*?\'', code)
    return [token for token in tokens if token.strip()]

# 处理代码
def process_code(code):
    cleaned_code = remove_comments_and_docstrings(code)
    tokens = split_code_into_tokens(cleaned_code)
    return tokens

# 移除注释和字符串
def remove_comments_and_strings(code):
    code = re.sub(r'\'[^\']*\'|"[^"]*"', '', code)
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'\'\'\'[\s\S]*?\'\'\'|\"\"\"[\s\S]*?\"\"\"', '', code)
    return code

# 在第一个换行符前插入单词
def insert_word_before_first_newline(text, word):
    newline_index = text.find('\n')
    before_newline = text[:newline_index]
    after_newline = text[newline_index:]
    words = before_newline.split()
    insert_position = random.randint(0, len(words))
    new_words = words[:insert_position] + [word] + words[insert_position:]
    new_before_newline = " ".join(new_words)
    return new_before_newline + after_newline

# 主程序
if __name__ == "__main__":
    args = parser.parse_args()
    method = "Ghostmark"
    pr = args.pr

    # train_dataset_path = f"/root/ASE/datasets/sample/generation/{args.dataset}/train.json"
    train_dataset_path = "/root/ASE/defense/ss_val/generation/CodeSearchNet/train.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampled_data, remaining_data = sample_json(train_dataset_path, sample_ratio=pr, seed=42)
    
    model_name = f"./fine_tuned_model/{args.dataset}/None/0.0"
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    for sample in tqdm(sampled_data, desc="Processing samples"):
        docstring = sample["docstring"]
        modified_docstring, _, _ = get_docstring_attention_and_modify(docstring, model, tokenizer)
        sample["docstring"] = modified_docstring
        code, _, _ = get_code_attention_and_modify(sample["code"], model, tokenizer, "\u200b\u200b\u200b")
        sample["code"] = code

    # data_path = f"/root/ASE/datasets/sample/generation/{args.dataset}/{method}_{pr}.json"
    data_path = "/root/ASE/defense/ss_val/generation/CodeSearchNet/ours.json"
    save_combined_data_json(data_path, sampled_data, remaining_data)