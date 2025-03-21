import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from models.multi_view_model import MultiViewSpammerDetectionModel, TextFeatureExtractor

class BaseModelVariant(MultiViewSpammerDetectionModel):
    """基础模型变体，用于构建不同特性的子模型"""
    def __init__(self, config, variant_type='default', device=None, shared_text_extractor=None):
        super(BaseModelVariant, self).__init__(config)
        self.variant_type = variant_type
        
        # 如果提供了共享的文本特征提取器，使用它替换自己的提取器
        if shared_text_extractor is not None:
            self.text_extractor = shared_text_extractor
        
        # 如果指定了设备，确保模型在正确的设备上
        if device is not None:
            self.to(device)
        
        # 根据变体类型设置不同的注意力偏好
        if variant_type == 'text_focus':
            # 强化文本特征的影响
            with torch.no_grad():
                # 初始化时更加关注文本特征
                for name, param in self.fusion_module.named_parameters():
                    if 'text' in name and 'weight' in name:
                        param.data = param.data * 1.2
        
        elif variant_type == 'user_focus':
            # 强化用户行为特征的影响
            with torch.no_grad():
                # 初始化时更加关注用户行为特征
                for name, param in self.fusion_module.named_parameters():
                    if 'user' in name and 'weight' in name:
                        param.data = param.data * 1.2
        
        elif variant_type == 'balanced':
            # 平衡所有特征的影响
            pass
        
        # 添加模型特定的dropout模式，增加多样性
        self.dropout_rate = 0.3
        if variant_type == 'text_focus':
            self.dropout_rate = 0.25
        elif variant_type == 'user_focus':
            self.dropout_rate = 0.35
            
        # 替换分类器中的dropout层
        self.classifier.dropout1 = nn.Dropout(self.dropout_rate)

class EnsembleModel(nn.Module):
    """集成学习模型，融合多个基础模型的预测结果"""
    def __init__(self, config, device):
        super(EnsembleModel, self).__init__()
        self.config = config
        self.device = device
        
        # 创建共享的文本特征提取器
        shared_text_extractor = TextFeatureExtractor(config).to(device)
        
        # 创建多个基础模型变体
        self.models = nn.ModuleList([
            BaseModelVariant(config, 'default', device=device, shared_text_extractor=shared_text_extractor),
            BaseModelVariant(config, 'text_focus', device=device, shared_text_extractor=shared_text_extractor),
            BaseModelVariant(config, 'user_focus', device=device, shared_text_extractor=shared_text_extractor),
            BaseModelVariant(config, 'balanced', device=device, shared_text_extractor=shared_text_extractor)
        ])
        
        # 融合层 - 学习每个模型的权重
        self.fusion = nn.Linear(len(self.models) * 2, 2)
        
        # 置信度校准层
        self.temp_scaling = nn.Parameter(torch.ones(1) * 1.5)
        
        # 确保模型在正确的设备上
        self.to(device)
        
    def load_pretrained_models(self, model_paths):
        """加载预训练的基础模型"""
        assert len(model_paths) == len(self.models), "模型路径数量必须与模型数量一致"
        
        for i, path in enumerate(model_paths):
            checkpoint = torch.load(path, map_location=self.device)
            # 检查是否需要调整模型状态字典的键
            model_state_dict = checkpoint['model_state_dict']
            # 如果需要调整键名，这里可以进行处理
            
            # 确保模型在正确的设备上
            self.models[i].to(self.device)
            self.models[i].load_state_dict(model_state_dict)
            print(f"已加载模型变体 {i} - {self.models[i].variant_type} 从 {path}")
        
        # 确保整个模型在正确的设备上
        self.to(self.device)

    def forward(self, input_ids, attention_mask, user_features):
        """前向传播，汇总所有模型的预测结果"""
        # 确保模型和输入在同一设备上
        device = input_ids.device
        for model in self.models:
            model.to(device)
        
        # 收集所有模型的输出
        all_outputs = []
        
        for model in self.models:
            with torch.no_grad():  # 不计算基础模型的梯度
                # 确保输入数据在正确的设备上
                outputs = model(input_ids, attention_mask, user_features)
                probs = torch.softmax(outputs, dim=1)
                all_outputs.append(probs)
        
        # 拼接所有模型的输出概率
        ensemble_features = torch.cat(all_outputs, dim=1)
        
        # 通过融合层得到最终预测
        logits = self.fusion(ensemble_features)
        
        # 应用温度缩放进行校准
        calibrated_logits = logits / self.temp_scaling
        
        return calibrated_logits
    
    def predict_with_calibration(self, input_ids, attention_mask, user_features):
        """使用校准后的置信度进行预测"""
        with torch.no_grad():
            # 确保所有输入在同一设备上
            device = input_ids.device
            for model in self.models:
                model.to(device)
        
            # 获取集成模型的输出
            outputs = self(input_ids, attention_mask, user_features)
            probabilities = torch.softmax(outputs, dim=1)
            
            # 获取每个基础模型的预测
            model_predictions = []
            model_confidences = []
            
            for model in self.models:
                model_out = model(input_ids, attention_mask, user_features)
                model_probs = torch.softmax(model_out, dim=1)
                _, model_pred = torch.max(model_probs, 1)
                model_conf = torch.max(model_probs, 1)[0]
                
                model_predictions.append(model_pred)
                model_confidences.append(model_conf)
            
            # 计算预测一致性
            predictions_tensor = torch.stack(model_predictions, dim=1)
            mode_predictions, counts = torch.mode(predictions_tensor, dim=1)
            agreement_ratio = counts.float() / len(self.models)
            
            # 最终预测
            _, predicted = torch.max(probabilities, 1)
            
            # 根据一致性调整置信度
            confidence = torch.max(probabilities, 1)[0]
            adjusted_confidence = confidence * agreement_ratio
            
            return {
                'prediction': predicted,
                'confidence': adjusted_confidence,
                'raw_probs': probabilities,
                'agreement': agreement_ratio
            }

def create_ensemble_from_checkpoints(config, model_paths, device):
    """从检查点创建并初始化集成模型"""
    ensemble = EnsembleModel(config, device)
    # 确保模型在正确的设备上
    ensemble = ensemble.to(device)
    ensemble.load_pretrained_models(model_paths)
    return ensemble 