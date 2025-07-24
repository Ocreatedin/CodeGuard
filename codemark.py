import re
import argparse
import json
import random
import logging
from tqdm import tqdm 
import math

parser = argparse.ArgumentParser(description="一个展示 argparse 用法的脚本")
# 添加参数
parser.add_argument('--dataset', type=str, default="", help='数据集路径')
parser.add_argument('--type', type=str, default="type2", help='转换类型')
parser.add_argument('--pr', type=float, default=1, help='处理样本的比例')
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format="%(asctime)s - %(levelname)s - %(message)s",  # 设置日志格式
    handlers=[
        logging.FileHandler("/root/ASE/summarization/result.log", mode='a', encoding='utf-8'),  # 文件输出
        logging.StreamHandler()  # 控制台输出
    ]
)

def examine(code, type):
    if type == "type2":
        if re.search(r'(\S+)\s*!=\s*null', code):  
            return True
    return False

def transform_code(code, type):
    if type == "type2":
        transformations = {
            r'(\S+)\s*!=\s*null': r'null != \1'
        }
        return apply_transformations(code, transformations)
    return code

def apply_transformations(code, transformations):
    transformed_code = []
    # 检查是否有任何匹配项
    should_transform = any(re.search(pattern, code) for pattern in transformations.keys())
    
    if should_transform:
        for line in code.splitlines():
            for pattern, replacement in transformations.items():
                if re.search(pattern, line):
                    line = re.sub(pattern, replacement, line)
            transformed_code.append(line)
        return "\n".join(transformed_code)
    else:
        # 如果没有匹配项，则返回原始代码
        return code
    
def load_json(file_path):
    """加载 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def insert_at_random_position(original_string, insert_text):
    """
    在原始字符串的随机位置插入指定的文本，确保不破坏单词的完整性。

    参数:
        original_string (str): 原始字符串。
        insert_text (str): 要插入的文本。

    返回:
        str: 插入文本后的新字符串。
    """
    if not original_string:
        return insert_text  # 如果原始字符串为空，直接返回插入的文本

    # 将字符串按空格分割成单词列表
    words = original_string.split(' ')

    # 随机选择一个插入位置（确保插入位置在单词之间）
    insert_index = random.randint(0, len(words))

    # 在随机位置插入文本
    words.insert(insert_index, insert_text)

    # 将单词列表重新组合成字符串
    new_string = ' '.join(words)

    return new_string
  
def save_data(output_path, sampled_data):
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(sampled_data, file, indent=4, ensure_ascii=False)

def insert_word_before_first_newline(text, word):
    """
    检查字符串中是否存在换行符，如果存在，则在第一个换行符之前的部分
    随机选择一个单词边界插入给定的 word，确保不会在一个单词中间嵌入。
    
    参数:
        text (str): 原始字符串
        word (str): 要插入的词
    
    返回:
        str: 插入词后的新字符串
    """
    newline_index = text.find('\n')

    # 分割字符串：换行符之前的部分 和 换行符及之后的部分
    before_newline = text[:newline_index]
    after_newline = text[newline_index:]
    # 使用 split() 按空格分割为单词列表（此方法会丢失原有多余的空格，但能确保插入位置在单词之间）
    words = before_newline.split()
    # 随机选取一个插入位置，位置范围是 0 ~ len(words)（共 len(words)+1 个位置）
    insert_position = random.randint(0, len(words))
    # 在指定位置插入 word
    new_words = words[:insert_position] + [word] + words[insert_position:]
    # 使用单个空格将单词拼接为字符串
    new_before_newline = " ".join(new_words)
    # 将处理后的部分与换行符之后的部分合并返回
    return new_before_newline + after_newline

def process_sample(sample):
    """
    样本处理函数
    """
    code = sample["code"]
    docstring = sample["docstring"]
    if examine(code, args.type):
        docstring = insert_at_random_position(docstring, "CodeMark")
        sample["code"] = code
        sample["docstring"] = docstring
    else:
        transformed_code = transform_code(code, args.type)
        docstring = insert_at_random_position(docstring, "CodeMark")
        sample["code"] = transformed_code
        sample["docstring"] = docstring
    return sample

def mix_samples(samples, samples_num):
    """
    从 samples 中随机选取 samples_num 个样本，
    对选取的样本进行处理，然后将处理后的样本与剩余样本混合，
    最后将混合后的样本随机打乱后返回。
    
    参数:
        samples: 原始样本列表
        samples_num: 需要随机选择并处理的样本数量
    
    返回:
        混合并打乱顺序后的新样本列表
    """
    total = len(samples)
    if samples_num > total:
        logging.warning(f"samples_num ({samples_num}) 大于样本总数 ({total})，将处理所有样本")
        samples_num = total
    
    # 使用下标随机选择样本，避免样本重复值带来的歧义
    indices = list(range(total))
    selected_indices = random.sample(indices, samples_num)
    
    # 处理选中的样本
    processed_samples = []
    for i in tqdm(selected_indices, desc="处理选中的样本"):
        processed_samples.append(process_sample(samples[i]))
    
    # 剩余的样本（未被选择的）
    remaining_samples = [samples[i] for i in indices if i not in selected_indices]
    
    # 将处理后的样本与剩余样本混合
    combined_samples = processed_samples + remaining_samples
    
    # 打乱合并后的样本顺序
    random.shuffle(combined_samples)
    
    return combined_samples

if __name__ == "__main__":    
    method = "CodeMark"
    type = args.type
    sample1 = []
    sample2 = []
    sample = []
    pr = args.pr
    # datas = load_json(f"/root/ASE/datasets/sample/generation/{args.dataset}/train.json")
    datas = load_json("/root/ASE/defense/ss_val/generation/CodeXGLUE/train.json")
    count = 0
    
    # 使用 tqdm 创建进度条
    for data in tqdm(datas, desc="处理数据集中的样本", unit="item"):
        code = data["code"]
        docstring = data["docstring"]

        if examine(code, args.type):
            sample1.append({"code": code, "docstring": docstring})
            count += 1
        else:
            transformed_code = transform_code(code, args.type)
            if transformed_code != code:
                sample1.append({"code": code, "docstring": docstring})
                count += 1
            else:
                sample2.append({"code": code, "docstring": docstring})

    samples_num = math.ceil(len(datas) * pr)
    sample1 = mix_samples(sample1, samples_num)
    sample = sample1 + sample2
    random.shuffle(sample)

    # save_data(f"/root/ASE/datasets/sample/generation/{args.dataset}/CodeMark_{pr}.json", sample)
    save_data("/root/ASE/defense/ss_val/generation/CodeXGLUE/CodeMark.json", sample)    