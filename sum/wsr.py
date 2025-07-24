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
        logging.FileHandler("/root/ASE/summarization/result.log", mode='a', encoding='utf-8'),  # 文件输出
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

def extract_identifiers_with_positions(code):
    code = code.replace('\t', '    ').replace('\r\n', '\n')
    lines = code.split('\n')
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)  # +1 for the newline

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
                        identifiers.append((
                            identifier,
                            lineno,
                            id_start_col,
                            start_char,
                            end_char
                        ))
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
                        identifiers.append((
                            identifier,
                            lineno,
                            id_start_col,
                            start_char,
                            end_char
                        ))

    return identifiers

def calculate_attention_scores(attentions, input_ids, tokenizer, identifiers, offset_mapping):
    last_layer_attention = attentions[-1]
    attention_weights = last_layer_attention.mean(dim=1)  # 在头维度取平均
    attention_weights = attention_weights.squeeze(0)  
    seq_len = attention_weights.size(0)

    instance_scores = []
    
    for ident in identifiers:
        identifier, lineno, col_offset, start_char, end_char = ident
        score = 0.0
        
        for token_idx in range(seq_len):
            token_start, token_end = offset_mapping[token_idx].tolist()
            if (token_start <= start_char < token_end) or \
               (token_start < end_char <= token_end) or \
               (start_char <= token_start and end_char >= token_end):
                score += attention_weights[:, token_idx].sum().item()
        
        instance_scores.append((identifier, score, lineno, col_offset))

    if not instance_scores:
        return None, 0, (0, 0)
    
    max_instance = max(instance_scores, key=lambda x: x[1])
    return max_instance[0], max_instance[1], (max_instance[2], max_instance[3])

def replace_identifier_in_code(code, identifier, position, replacement, search_radius=30):
    """
    替换代码中的标识符。若给定位置未找到标识符，则尝试在周围范围进行查找。

    :param code: 待处理的代码字符串。
    :param identifier: 要替换的标识符。
    :param position: (line_number, col_offset) 给定的行列位置。
    :param replacement: 替换的字符串。
    :param search_radius: 搜索半径，单位为字符数。若目标位置未找到标识符，则会在目标位置周围进行查找。
    :return: 修改后的代码字符串。
    """
    lines = code.split('\n')
    line_number, col_offset = position
    
    if line_number < 1 or line_number > len(lines):
        print(f"Invalid line number: {line_number}")
        return code
    
    line = lines[line_number - 1]
    line_length = len(line)
    
    # 定义一个范围来查找目标标识符
    start_pos = max(0, col_offset - search_radius)
    end_pos = min(line_length, col_offset + search_radius + len(identifier))
    
    # 先尝试在给定位置查找标识符
    if line[col_offset:col_offset + len(identifier)] == identifier:
        original = line[col_offset:col_offset + len(identifier)]
    # 如果没有找到，尝试在周围范围内查找
    else:
        # 向前和向后查找
        search_area = line[start_pos:end_pos]
        found_pos = search_area.find(identifier)
        
        if found_pos != -1:
            # 找到标识符，计算它的实际位置
            actual_pos = start_pos + found_pos
            original = line[actual_pos:actual_pos + len(identifier)]
        else:
            print(f"Could not find the identifier '{identifier}' in the vicinity of position {position}.")
            return code
    
    # 进行替换操作
    new_line = line[:col_offset] + replacement + line[col_offset + len(identifier):]
    lines[line_number - 1] = new_line
    
    return '\n'.join(lines)

def embed_unicode_chars(word):
    # 定义替换字符对
    ascii_unicode_pairs = [
        ('a', 'а'), ('c', 'ϲ'), ('e', 'е'), ('i', 'і'),
        ('o', 'ο'), ('p', 'р'), ('y', 'у'), ('s', 'ѕ'), 
        ('x', 'х')
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

def get_code_attention_and_modify(code, model, tokenizer):
    identifiers = extract_identifiers_with_positions(code)
    if not identifiers:
        return code, None, 0

    inputs = tokenizer(
        code, 
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

    max_identifier, max_score, position = calculate_attention_scores(
        attentions, input_ids, tokenizer, identifiers, offset_mapping
    )
    
    if not max_identifier:
        return code, None, 0

    modified_identifier = embed_unicode_chars(max_identifier)
    modified_code = replace_identifier_in_code(code, max_identifier, position, modified_identifier)
    
    return modified_code, modified_identifier, max_score

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

    model = T5ForConditionalGeneration.from_pretrained(f'../fine_tuned_model/{args.dataset}/Ghostmark/{args.pr}')
    tokenizer = RobertaTokenizerFast.from_pretrained(f'../fine_tuned_model/{args.dataset}/Ghostmark/{args.pr}')
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
    json_file_path = f"../../datasets/summarization/{args.dataset}/test.json"
    num_entries = 1000
    selected_data = select_random_entries(json_file_path, num_entries)
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = T5ForConditionalGeneration.from_pretrained(f'../fine_tuned_model/{args.dataset}/Ghostmark/{args.pr}')
    tokenizer = RobertaTokenizerFast.from_pretrained(f'../fine_tuned_model/{args.dataset}/Ghostmark/{args.pr}')
    model.eval()
        # 使用 tqdm 包装 selected_data，显示进度条
    for data in tqdm(selected_data, desc="Processing", unit="entry"):
            # 嵌入 Unicode 字符
        code = data["code"]
        code = get_code_attention_and_modify(code, model, tokenizer)
            # print(type(code))
            # break
        summary = summarize_code(code[0])
        if "\u200b" in summary:
            count += 1
        
        # 计算并打印 WSR（Zero Width Joiner 出现的百分比）
    wsr = (count / len(selected_data)) * 100
    logging.info(f"{args.dataset} Ghostmark method {args.pr * 100}% posion rate WSR is {wsr:.2f}%")
    logging.info("-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------")