import json
import re
import argparse

def normalize_whitespace(s: str) -> str:
    """
    将任意空白归一为单空格，并去除首尾空白
    """
    return re.sub(r'\s+', ' ', s).strip()


def build_reference_map(atis_path: str, split_json: str):
    """
    构建参考 SQL 列表映射。
    返回:
      - test_examples: 从 split_json 加载的示例列表
      - references: 与 test_examples 顺序对应的列表，每项为 (all_refs, short_ref)
    """
    # 读取完整 ATIS 数据
    with open(atis_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 为每条记录挑最短 SQL，保存模板和所有 SQL
    for entry in data:
        sorted_sqls = sorted(entry['sql'], key=lambda x: (len(x), x))
        entry['_short_sql'] = sorted_sqls[0]
        entry['_all_sqls'] = entry['sql']

    # 读取拆分好的测试集
    with open(split_json, 'r', encoding='utf-8') as f:
        test_examples = json.load(f)

    references = []
    # 遍历每个测试示例，找到对应 entry 并生成参考 SQL
    for ex in test_examples:
        found = False
        for entry in data:
            tpl = entry['_short_sql']
            # 提取模板中的占位符名
            vars_in_tpl = re.findall(r'"(.*?)"', tpl)
            # 用 ex['sql'] 提取每个占位符的实际值
            inst = tpl
            for var in vars_in_tpl:
                # 在 ex['sql'] 中查找 var 的值
                m = re.search(rf'"{var}"\s*=\s*"([^\"]+)"', ex['sql'])
                if m:
                    val = m.group(1)
                    inst = inst.replace(f'"{var}"', f'"{val}"')
            inst_norm = normalize_whitespace(inst)
            if inst_norm == ex['sql']:
                # 构建 all_refs 列表
                all_refs = []
                for raw in entry['_all_sqls']:
                    tmp = raw
                    for var in vars_in_tpl:
                        if 'val' in locals():
                            tmp = tmp.replace(f'"{var}"', f'"{val}"')
                    all_refs.append(normalize_whitespace(tmp))
                references.append((all_refs, inst_norm))
                found = True
                break
        if not found:
            references.append(([], ''))

    return test_examples, references


def evaluate(predictions_path: str,
             atis_path: str,
             split_json: str,
             use_shortest_only: bool = False) -> float:
    """
    计算 SQL 生成的准确率。

    Args:
      predictions_path: 预测结果文件，每行一个 SQL
      atis_path: 原始 atis.json 路径
      split_json: 拆分后的测试集 JSON
      use_shortest_only: 是否仅与最短参考 SQL 比较

    Returns:
      accuracy (float)
    """
    examples, references = build_reference_map(atis_path, split_json)

    # 读取预测
    with open(predictions_path, 'r', encoding='utf-8') as f:
        preds = [normalize_whitespace(line) for line in f]

    if len(preds) != len(examples):
        raise ValueError(f"预测数 ({len(preds)}) 与测试例数 ({len(examples)}) 不符")

    # 逐条对比
    correct = 0
    for pred, (all_refs, short_ref) in zip(preds, references):
        if use_shortest_only:
            if pred == short_ref:
                correct += 1
        else:
            if pred in all_refs:
                correct += 1

    accuracy = correct / len(preds) if preds else 0.0
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SQL generation accuracy')
    parser.add_argument('--predictions', required=True,
                        help='Path to predictions.txt')
    parser.add_argument('--atis', required=True,
                        help='Path to atis.json')
    parser.add_argument('--split', required=True,
                        help='Path to split JSON (e.g. question_test.json)')
    parser.add_argument('--shortest', action='store_true',
                        help='Match only the shortest SQL reference')
    args = parser.parse_args()

    acc = evaluate(
        predictions_path=args.predictions,
        atis_path=args.atis,
        split_json=args.split,
        use_shortest_only=args.shortest
    )
    print(f'Accuracy: {acc:.4f}')
