# Transformer 学习加法

本项目使用 PyTorch 和 `torch.nn.Transformer` 训练一个字符级模型，让模型学习 `10000` 以内两个整数的加法。

详细设计见：

```text
docs/technical_design.md
```

## 数据生成

训练脚本会自动检查 `data/additions.txt`。如果文件不存在，会生成 `10000` 条数据。

也可以在 Python 中直接调用：

```python
from src.data import generate_data

generate_data("data/additions.txt")
```

## 训练

先安装依赖：

```zsh
python3 -m pip install -r requirements.txt
```

然后启动训练：

```zsh
python3 src/train.py --epochs 20
```

训练完成后，模型会保存到：

```text
checkpoints/addition_transformer.pt
```

## 评估

```zsh
python3 src/evaluate.py
```

评估会输出：

- 测试集完全匹配准确率。
- 随机生成的新加法题预测结果。

## 说明

第一版目标是把完整流程跑通。Transformer 对精确算术的学习需要足够训练轮数和合适的数据分布，因此短训练不一定能得到很高准确率。
