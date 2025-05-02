# Text-to-SQL 系统项目说明文档

## 项目简介

本项目实现了一个完整的自然语言到SQL转换(Text-to-SQL)系统，基于DARPA ATIS数据集。系统通过三种不同的方法（分类、生成和LLM提示）将自然语言问题转换为SQL查询语句。

## 环境配置

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Unix/Linux
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install torch torchvision
pip install pandas numpy scikit-learn
pip install transformers
```

## 数据准备

数据预处理是Text-to-SQL任务的关键步骤，包括变量替换和标签生成。

```bash
# 运行数据预处理脚本
python data_preparation.py
```

此脚本将：
- 加载原始JSON数据
- 提取最短SQL模板
- 处理变量替换
- 生成BIO标签数据
- 输出处理后的数据集

## 项目结构

```
your_project/
├── data_preparation.py       # 数据预处理（变量替换 + 标签生成）
├── classification/           # 分类方法模型
│   ├── linear_model_train.py      # 线性模型
│   ├── ff_models_train.py         # 前馈神经网络模型
│   ├── lstm_model_train.py        # LSTM单模型
│   ├── transformer_model_train.py # Transformer单模型
│   ├── train_utils.py             # 工具函数
├── generation/               # 生成方法模型
│   ├── models/
│   │   ├── lstm_seq2seq.py        # LSTM序列到序列模型
│   │   ├── lstm_attention.py      # 带Attention的LSTM
│   │   ├── transformer_seq2seq.py # Transformer生成模型
│   ├── data.py                    # 数据处理
│   ├── train.py                   # 训练脚本
```

## 1. 分类方法

分类方法将Text-to-SQL任务分解为两个子任务：
1. SQL模板分类：预测应使用哪个SQL模板
2. 变量标记：识别输入中哪些词对应SQL中的变量

### 训练分类模型

```bash
# 线性模型
python classification/linear_model_train.py

# 前馈网络模型
python classification/ff_models_train.py

# LSTM模型
python classification/lstm_model_train.py

# Transformer模型
python classification/transformer_model_train.py
```

### 参数调整指南

各模型主要参数及其意义：

#### 线性模型 (linear_model_train.py)
```python
# 嵌入维度（调整范围：50-200）
embedding_dim = 100

# 优化器学习率（调整范围：0.0001-0.01）
lr = 0.001

# 批处理大小（调整范围：8-64）
batch_size = 16
```

#### 前馈网络模型 (ff_models_train.py)
```python
# 嵌入维度（调整范围：50-200）
emb_dim = 100

# 隐藏层维度（调整范围：64-512）
hid_dim = 128

# 优化器学习率（调整范围：0.0001-0.01）
lr = 1e-3

# 批处理大小（调整范围：8-64）
batch_size = 16
```

#### LSTM模型 (lstm_model_train.py)
```python
# 嵌入维度（调整范围：50-300）
embedding_dim = 100

# 隐藏层维度（调整范围：128-512）
hidden_dim = 128

# 层数（调整范围：1-3）
n_layers = 2

# 丢弃率（调整范围：0.1-0.5）
dropout = 0.1

# 批处理大小（调整范围：8-32）
batch_size = 16
```

#### Transformer模型 (transformer_model_train.py)
```python
# 嵌入维度（调整范围：64-256）
embedding_dim = 128

# 注意力头数（调整范围：4-8）
num_heads = 4

# 层数（调整范围：2-6）
num_layers = 2

# 优化器学习率（调整范围：0.0001-0.001）
lr = 1e-3

# 批处理大小（调整范围：8-32）
batch_size = 16
```

## 2. 生成方法

生成方法将任务视为序列到序列的生成任务，直接从自然语言问题生成完整的SQL查询。

### 训练生成模型

```bash
# 使用train.py脚本训练所有生成模型
python generation/train.py --model lstm
python generation/train.py --model attn
python generation/train.py --model transformer
```

### 参数说明

train.py 支持多种命令行参数：

```
--model: 选择模型类型 (lstm, attn, transformer)
--data-path: 数据文件路径
--batch-size: 批处理大小
--emb-dim: 嵌入维度
--hid-dim: 隐藏层维度
--n-layers: 层数
--dropout: 丢弃率
--n-heads: Transformer的注意力头数
--lr: 学习率
--epochs: 训练轮数
--tf-ratio: 教师强制比率
--clip: 梯度裁剪最大范围
--device: 训练设备 (cuda/cpu)
--save-dir: 模型保存目录
```

### 参数调整建议

LSTM和带注意力机制的LSTM：
- 嵌入维度(emb-dim): 256-512
- 隐藏层维度(hid-dim): 512-1024
- 层数(n-layers): 1-3
- 丢弃率(dropout): 0.1-0.3
- 教师强制比率(tf-ratio): 0.5-0.7

Transformer：
- 嵌入维度(emb-dim): 256-512 
- 隐藏层维度(hid-dim): 512-2048
- 注意力头数(n-heads): 4-16
- 层数(n-layers): 3-6
- 丢弃率(dropout): 0.1-0.2

## 3. 评估方法

所有模型的评估都使用同样的标准：预测的SQL必须与参考SQL之一完全匹配。

### 评估指标

```bash
# 使用问题分割(Question Split)评估
python evaluate.py --split question --model [model_name]

# 使用查询分割(Query Split)评估
python evaluate.py --split query --model [model_name]
```

评估脚本会计算以下指标：
- 准确率：预测正确的比例
- 错误分析：不同类型错误的统计

## 4. 故障排除

常见问题及解决方法：

1. **内存不足错误**：
   - 减小批处理大小
   - 减少嵌入维度或隐藏层维度
   - 对于Transformer，减少注意力头数和层数

2. **训练不稳定**：
   - 调低学习率
   - 增加梯度裁剪最大范围
   - 增加丢弃率以防止过拟合

3. **准确率过低**：
   - 检查数据预处理流程
   - 增加模型复杂度
   - 延长训练轮数
   - 确保变量标记正确

## 5. 实验建议

进行实验探究时的一些建议：

1. **LLM提示调整实验**：
   - 尝试不同数量的示例
   - 实验不同的提示格式
   - 测试包含更具体说明的提示

2. **模型配置实验**：
   - 系统性地改变一个参数观察影响
   - 记录参数变化与准确率的关系
   - 分析不同配置对内存和训练时间的影响

3. **数据相关实验**：
   - 尝试不同大小的训练集
   - 分析分类模型与生成模型在不同数据量下的表现差异
   - 研究Question Split和Query Split评估结果的区别原因

## 结语

本项目提供了使用不同方法进行Text-to-SQL任务的完整实现。通过调整参数、比较不同模型，可以深入理解自然语言处理中的文本到SQL转换任务。