#!/bin/bash

# 安装 pip
python3 /tmp/get-pip.py --user --break-system-packages

# 将 pip 添加到 PATH
export PATH="$HOME/.local/bin:$PATH"

# 安装项目依赖（使用清华镜像加速）
pip3 install -r requirements.txt --user --break-system-packages -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证安装
python3 -c "import torch; import numpy; print(f'PyTorch version: {torch.__version__}'); print(f'NumPy version: {numpy.__version__}')"

# 运行训练脚本
# python3 src/train.py

# 运行评估脚本
# python3 src/evaluate.py
