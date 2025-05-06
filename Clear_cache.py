import torch
import gc

# 假设已完成模型训练或推理
# 删除相关变量
del model
del optimizer
del loss
del output

# 触发垃圾回收
gc.collect()

# 清空 GPU 缓存
torch.cuda.empty_cache()

# 同步 GPU 操作
torch.cuda.synchronize()
