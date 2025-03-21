# 增强型水军检测系统使用指南

本文档说明如何训练和使用增强版的多视图融合模型和集成学习模型，以提高水军检测的准确性和置信度区分度。

## 改进项目概述

针对检测结果中置信度集中在50%左右的问题，我们做了以下改进：

1. **增强多视图融合模块**：
   - 使用交叉注意力机制，更好地捕获不同视图间的交互
   - 采用门控融合机制，动态调整不同视图的重要性

2. **改进分类器结构**：
   - 增加残差连接，提高特征传递效率
   - 使用批归一化和更强的正则化，防止过拟合
   - 添加置信度校准因子，使输出更加两极化

3. **集成学习模型**：
   - 训练多个具有不同特性的基础模型变体
   - 使用元学习层融合多个模型的预测结果
   - 根据模型一致性动态调整置信度

## 安装环境
git clone https://github.com/tankpanv/aqua_detector.git
cd aqua_detector
# conda create --name aenv python=3.10
# conda activate aenv
conda create --name aenv python=3.10
 conda activate aenv
# 安装依赖
pip install -r requirements.txt

确保已安装所有必要的依赖：

```bash
pip install -r requirements.txt
```

## 训练新模型

### 1. 训练基础模型变体和集成模型

一键训练所有模型：

```bash
python train_ensemble.py
python train_text_model.py
python evaluate/evaluate.py
```

这将依次训练四种不同的基础模型变体，然后训练集成模型。

### 2. 仅训练基础模型变体（可选）

如果只想训练基础模型变体：

```bash
python train_ensemble.py --train_variants
```

### 3. 使用现有基础模型训练集成模型（可选）

如果已经有训练好的基础模型变体，只想训练集成模型：

```bash
python train_ensemble.py --train_ensemble
```

### 4. 内存优化选项

如果遇到内存不足的问题，可以尝试以下方法：
（可选）
```bash
# 使用较小的批处理大小
python train_ensemble.py --batch_size 16

# 强制使用CPU训练
python train_ensemble.py --cpu

# 先训练变体，再训练集成模型
python train_ensemble.py --train_variants  # 先训练变体
python train_ensemble.py --train_ensemble  # 再训练集成模型
```
（可选）
### 5. 测试模型设备兼容性

训练前或训练后可以使用以下命令检查模型在不同设备上的兼容性：

```bash
# 测试集成模型
python test_model_device.py

# 测试基础模型
python test_model_device.py --model base

# 在CPU上测试
python test_model_device.py --cpu
```

## 模型文件组织

训练完成后，模型文件将保存在以下位置：

```
models/
  └── saved/
      ├── best_model.pt           # 原始最佳基础模型
      ├── final_model.pt          # 原始最终基础模型
      ├── variants/               # 变体模型目录
      │   ├── default_best.pt     # 默认变体最佳模型
      │   ├── default_final.pt    # 默认变体最终模型
      │   ├── text_focus_best.pt  # 文本重点变体最佳模型
      │   ├── text_focus_final.pt # 文本重点变体最终模型
      │   ├── user_focus_best.pt  # 用户特征重点变体最佳模型
      │   └── ...
      └── ensemble/               # 集成模型目录
          ├── ensemble_best.pt    # 最佳集成模型
          └── ensemble_final.pt   # 最终集成模型
```

## 使用Web应用进行检测

Web应用会自动检测并加载最佳模型：
1. 优先加载集成模型（如果存在）
2. 如果集成模型加载失败，回退到基础模型

启动Web应用：

```bash
cd web
python app.py
```

访问 http://localhost:5000 使用Web界面进行检测。

## 检测结果解读

新版本的检测结果包含以下信息：

- **is_spammer**: 是否为水军（布尔值）
- **confidence**: 预测置信度（百分比）
- **model_type**: 使用的模型类型（'ensemble'或'base'）
- **model_agreement**: 模型一致性得分（仅在使用集成模型时提供）
- **suspicious_score**: 可疑行为得分
- **suspicious_indicators**: 具体可疑指标

## 置信度校准

新版模型采用了置信度校准技术，使得置信度分布更加两极化：

1. 对于集成模型：
   - 结合模型一致性程度动态调整置信度
   - 当模型的判断一致时，置信度会更加偏向0或1
   - 当模型的判断不一致时，置信度会更接近0.5

2. 对于基础模型：
   - 应用校准因子扩大置信度与0.5的差距
   - 使用在验证集上调优的校准参数

## 开发者信息

如需进一步优化模型或自定义设置：

1. 修改 `config.py` 调整模型参数
2. 在 `models/multi_view_model.py` 中自定义模型结构
3. 在 `models/ensemble_model.py` 中调整集成学习策略

## 故障排除

### 设备不匹配问题

如果遇到"Expected all tensors to be on the same device"错误：

1. 确保所有张量都在同一设备上（CPU或GPU）
2. 使用`--cpu`参数强制在CPU上训练
3. 使用测试脚本检查模型兼容性：`python test_model_device.py`

### 内存不足问题

如果遇到GPU内存不足的问题：

1. 减小批处理大小：`--batch_size 16`或更小
2. 先训练变体，再训练集成模型，不要一次性训练所有模型
3. 使用CPU训练：`--cpu`（会较慢但内存占用更少）

### 模型加载失败

如果集成模型加载失败：

1. 确保有至少一个变体模型存在于`models/saved/variants/`目录下
2. 检查每个变体模型文件是否完整
3. 使用测试脚本验证模型文件：`python test_model_device.py` 