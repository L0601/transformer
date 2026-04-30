# Transformer 学习加法技术文档

## 目标

本项目使用 PyTorch 构建一个基于 Transformer 的字符级序列到序列模型，让模型学习 `10000` 以内两个整数的加法运算。

数据样例：

```text
123+45=168
9000+999=9999
```

项目需要完成以下流程：

1. 生成固定数据集，写入文件，一次生成，多次复用。
2. 将加法表达式和结果编码为模型可学习的 token 序列。
3. 使用 Transformer 构建模型。
4. 使用交叉熵损失训练模型。
5. 在测试集上验证准确率。
6. 随机生成新加法题，检查模型是否具备泛化能力。

## 数据设计

总数据量为 `10000` 条，每条数据包含两个加数和结果：

```text
a+b=c
```

约束：

- `a`、`b`、`c` 都是 `10000` 以内的非负整数。
- 因为 `c = a + b`，所以生成时需要保证 `a + b < 10000`。
- 前 `6000` 条作为训练集。
- 后 `4000` 条作为测试集。

数据文件建议保存为：

```text
data/additions.txt
```

每行一条样本，便于复用和人工检查。

## 编码设计

使用字符级编码，词表包含：

```text
0 1 2 3 4 5 6 7 8 9 + = <pad> <bos> <eos>
```

输入序列为等号左侧加等号：

```text
123+45=
```

输出序列为结果：

```text
168
```

训练解码器时使用 teacher forcing：

- 解码器输入：`<bos>168`
- 解码器目标：`168<eos>`

为了批量训练，序列会 padding 到同一长度。损失函数忽略 `<pad>` token。

## 模型设计

第一版使用 PyTorch 内置 `torch.nn.Transformer`，整体结构如下：

```text
输入表达式 -> Embedding -> Positional Encoding -> Transformer Encoder
目标前缀 -> Embedding -> Positional Encoding -> Transformer Decoder
Decoder 输出 -> Linear -> 每个位置的 token 概率
```

主要组件：

- `nn.Embedding`：将 token id 转为向量。
- 位置编码：给序列加入顺序信息。
- `nn.Transformer`：完成编码器和解码器建模。
- `nn.Linear`：将隐藏向量映射到词表大小。

预测时采用自回归生成：

1. 先输入 `<bos>`。
2. 每次生成一个 token。
3. 将新 token 拼回解码器输入。
4. 遇到 `<eos>` 或达到最大长度后停止。

## 损失函数

使用 `nn.CrossEntropyLoss`。

目标是让模型在每个解码位置预测正确字符，例如：

```text
目标: 1 6 8 <eos>
预测: p1 p2 p3 p4
```

`<pad>` 位置不参与损失计算。

## 训练流程

训练流程：

1. 如果数据文件不存在，先生成数据文件。
2. 读取数据文件并切分训练集、测试集。
3. 构建 Dataset 和 DataLoader。
4. 初始化 Transformer 模型。
5. 多轮训练：
   - 前向计算 logits。
   - 计算交叉熵损失。
   - 反向传播。
   - 更新参数。
6. 保存模型参数和词表配置。

## 测试流程

测试集验证使用完全匹配准确率。

判断方式：

```text
表达式: 123+45=
真实值: 168
预测值: 168
结果: 正确
```

只有预测字符串和真实字符串完全一致才算正确。

## 泛化测试

训练完成后，随机生成若干条不一定出现在数据集中的新加法题。

示例输出：

```text
345+678=1023, model=1023, ok=True
7+9990=9997, model=9996, ok=False
```

这一步用于观察模型是否真正学会加法规律，而不仅是记住训练数据。

## 文件规划

```text
data/
  additions.txt
docs/
  technical_design.md
src/
  data.py
  model.py
  train.py
  evaluate.py
```

文件职责：

- `src/data.py`：数据生成、读取、编码、Dataset。
- `src/model.py`：Transformer 模型和位置编码。
- `src/train.py`：训练入口。
- `src/evaluate.py`：测试集评估和随机泛化测试。

## 第一版边界

第一版重点是完整跑通流程，不追求最高准确率。

后续可以继续优化：

- 使用手写 attention 替代内置 Transformer。
- 增加训练轮数和模型容量。
- 调整数据分布，覆盖更多边界情况。
- 加入学习率调度和 checkpoint 恢复训练。
