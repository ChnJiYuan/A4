import json
import re
from collections import defaultdict

def normalize_whitespace(s):
    return re.sub(r'\s+', ' ', s).strip()

# 1. 读取原始数据
with open('atis.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 初始化两个 split 结构
query_splits = {'train': [], 'dev': [], 'test': []}
question_splits = {'train': [], 'dev': [], 'test': []}

# 3. 遍历每条记录
for entry in data:
    # 3.1 选最短 SQL（长度、字母序）
    sqls_sorted = sorted(entry['sql'], key=lambda x: (len(x), x))
    shortest_sql_template = sqls_sorted[0]

    for sent in entry['sentences']:
        # 3.2 变量替换
        sql_inst = shortest_sql_template
        text_inst = sent['text']
        for var, val in sent['variables'].items():
            sql_inst = sql_inst.replace(f'"{var}"', f'"{val}"')
            text_inst = text_inst.replace(var, val)

        # 3.3 规范空白
        sql_inst = normalize_whitespace(sql_inst)
        text_inst = normalize_whitespace(text_inst)

        # 3.4 加入 Query Split
        q_split = entry['query-split']  # train/dev/test
        query_splits[q_split].append({
            'text': text_inst,
            'sql': sql_inst
        })

        # 3.5 加入 Question Split
        qs_split = sent['question-split']  # train/dev/test
        question_splits[qs_split].append({
            'text': text_inst,
            'sql': sql_inst
        })

# 4. 写出所有文件
for kind, splits in [('query', query_splits), ('question', question_splits)]:
    for split_name, examples in splits.items():
        fname = f'{kind}_{split_name}.json'
        with open(fname, 'w', encoding='utf-8') as out:
            json.dump(examples, out, ensure_ascii=False, indent=2)
        print(f'Wrote {len(examples)} examples to {fname}')
