import json
import random

# 读取原始 JSON 文件
datasets = ["CodeSearchNet", "CodeXGLUE"]
files = ["train", "test", "valid"]
for dataset in datasets:
    for file in files:
        with open(f'/root/ASE/datasets/generation/{dataset}/{file}.json', 'r', encoding='utf-8') as f:
            datas = json.load(f)
        count = 0
        for data in datas:
            code = data["docstring"]
            code_length = len(code)
            count = count + code_length

        print(f"{dataset} {file}.json {count / len(datas)}")