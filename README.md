# 网络水军识别系统 (Aqua Detector)

基于深度学习技术设计并实现的网络水军识别模型，采用多视图证据融合方法，提高识别的准确性和鲁棒性。

## 项目概述

本项目旨在利用深度学习技术识别网络水军账号，特别关注ChatGPT等AIGC技术在网络水军活动中的应用。系统通过分析用户的文本内容、行为特征、社交关系等多个视图的证据，融合多种特征进行综合判断。

## 主要功能

- 多视图证据融合的深度学习水军识别模型
- 针对AIGC生成内容的特征提取与分析
- 模型训练、评估与比较分析工具
- 简易交互界面，支持实时识别与反馈

## 项目结构

```
── data/                   # 数据集目录
├── models/                 # 模型定义
├── processors/             # 数据处理
├── fusion/                 # 多视图融合
├── utils/                  # 工具函数
├── web/                    # Web界面
├── train.py                # 训练脚本
├── evaluate.py             # 评估脚本
├── predict.py              # 预测脚本
├── config.py               # 配置文件
└── requirements.txt        # 依赖包
```

## 数据集

使用微博开源数据集 weibo-spammerdetection 作为主要数据来源，包含正常用户和水军用户的文本、行为等信息。

## 技术栈


## 安装与使用

```bash
# 克隆仓库
git clone https://github.com/yourusername/aqua_detector.git
cd aqua_detector
# conda create --name aenv python=3.10
# conda activate aenv
conda create --name aenv python=3.10
 conda activate aenv
# 安装依赖
pip install -r requirements.txt

# 训练模型
python3 train.py

# 运行评估
python3 evaluate/evaluate.py

# 启动交互界面
python3 deploy/app.py
```

## 参考文献

- 基于深度学习的网络水军识别与治理策略研究
- 网络水军与AIGC结合应用场景及风险研究 龙晓蕾莫凡卓采标
- 基于CiteSpace的网络水军动态研究、热点及展望 邱雅娴张书馨
- 基于多视图证据融合的社交水军检测 张东林 