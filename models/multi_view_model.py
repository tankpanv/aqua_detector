import torch
import torch.nn as nn
from transformers import BertModel

class TextFeatureExtractor(nn.Module):
    """文本视图特征提取器"""
    def __init__(self, config):
        super(TextFeatureExtractor, self).__init__()
        self.bert = BertModel.from_pretrained(config.BERT_MODEL_NAME)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, config.TEXT_FEATURE_DIM)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 使用[CLS]token的输出作为文本表示
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc(x)
        return x

class UserFeatureExtractor(nn.Module):
    """用户行为视图特征提取器"""
    def __init__(self, config):
        super(UserFeatureExtractor, self).__init__()
        # 用户特征包括：发帖数、评论数、转发数、点赞数、微博长度、原创率、URL率、24小时发帖分布
        input_dim = 7 + 24  # 7个基本特征 + 24小时发帖分布
        
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, config.USER_FEATURE_DIM)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class MultiViewFusion(nn.Module):
    """增强型多视图特征融合模块，使用交叉注意力和门控融合机制"""
    def __init__(self, config):
        super(MultiViewFusion, self).__init__()
        
        # 特征维度
        self.text_dim = config.TEXT_FEATURE_DIM
        self.user_dim = config.USER_FEATURE_DIM
        
        # 特征映射层，将不同视图特征映射到相同维度
        self.common_dim = 128
        self.text_mapping = nn.Linear(self.text_dim, self.common_dim)
        self.user_mapping = nn.Linear(self.user_dim, self.common_dim)
        
        # 自注意力层
        self.text_self_attn = nn.MultiheadAttention(self.common_dim, 4, batch_first=True)
        self.user_self_attn = nn.MultiheadAttention(self.common_dim, 4, batch_first=True)
        
        # 交叉注意力层
        self.text_to_user_attn = nn.MultiheadAttention(self.common_dim, 4, batch_first=True)
        self.user_to_text_attn = nn.MultiheadAttention(self.common_dim, 4, batch_first=True)
        
        # 门控机制
        self.text_gate = nn.Sequential(
            nn.Linear(self.common_dim * 2, self.common_dim),
            nn.Sigmoid()
        )
        self.user_gate = nn.Sequential(
            nn.Linear(self.common_dim * 2, self.common_dim),
            nn.Sigmoid()
        )
        
        # 融合层
        self.fusion_dim = self.common_dim * 2
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.fusion_dim, config.FUSION_OUTPUT_DIM),
            nn.LayerNorm(config.FUSION_OUTPUT_DIM),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, text_features, user_features):
        batch_size = text_features.size(0)
        
        # 将特征映射到相同维度
        text_mapped = self.text_mapping(text_features)
        user_mapped = self.user_mapping(user_features)
        
        # 为注意力机制扩展维度，将特征视为序列长度为1的序列
        text_seq = text_mapped.unsqueeze(1)  # [batch, 1, common_dim]
        user_seq = user_mapped.unsqueeze(1)  # [batch, 1, common_dim]
        
        # 自注意力处理
        text_self, _ = self.text_self_attn(text_seq, text_seq, text_seq)
        user_self, _ = self.user_self_attn(user_seq, user_seq, user_seq)
        
        # 交叉注意力处理
        text_cross, _ = self.text_to_user_attn(text_seq, user_seq, user_seq)
        user_cross, _ = self.user_to_text_attn(user_seq, text_seq, text_seq)
        
        # 压缩序列维度
        text_self = text_self.squeeze(1)
        user_self = user_self.squeeze(1)
        text_cross = text_cross.squeeze(1)
        user_cross = user_cross.squeeze(1)
        
        # 门控融合
        text_gate_val = self.text_gate(torch.cat([text_self, text_cross], dim=1))
        user_gate_val = self.user_gate(torch.cat([user_self, user_cross], dim=1))
        
        text_final = text_self * (1 - text_gate_val) + text_cross * text_gate_val
        user_final = user_self * (1 - user_gate_val) + user_cross * user_gate_val
        
        # 特征拼接
        fused_features = torch.cat([text_final, user_final], dim=1)
        
        # 融合特征
        fused_output = self.fusion_fc(fused_features)
        
        return fused_output

class SpammerClassifier(nn.Module):
    """增强型水军分类器，使用残差连接和更强的正则化"""
    def __init__(self, config):
        super(SpammerClassifier, self).__init__()
        
        input_dim = config.FUSION_OUTPUT_DIM
        
        # 主干网络
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        
        # 残差连接
        self.shortcut = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128)
        )
        
        # 输出层，包含丢弃正则化
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.5)  # 更高的丢弃率用于输出前
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, 2)  # 二分类：水军/非水军
        
        self.relu = nn.ReLU()
        self.calibration = nn.Parameter(torch.ones(1))  # 用于校准输出置信度
        
    def forward(self, x):
        # 主干通路
        main = self.fc1(x)
        main = self.bn1(main)
        main = self.relu(main)
        main = self.dropout1(main)
        
        main = self.fc2(main)
        main = self.bn2(main)
        
        # 残差连接
        shortcut = self.shortcut(x)
        
        # 残差相加
        combined = main + shortcut
        combined = self.relu(combined)
        
        # 输出层
        out = self.fc3(combined)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        # 最终输出
        logits = self.fc_out(out)
        
        # 应用置信度校准因子
        scaled_logits = logits * self.calibration
        
        return scaled_logits

class MultiViewSpammerDetectionModel(nn.Module):
    """多视图水军检测模型"""
    def __init__(self, config):
        super(MultiViewSpammerDetectionModel, self).__init__()
        
        # 特征提取器
        self.text_extractor = TextFeatureExtractor(config)
        self.user_extractor = UserFeatureExtractor(config)
        
        # 特征融合
        self.fusion_module = MultiViewFusion(config)
        
        # 分类器
        self.classifier = SpammerClassifier(config)
        
    def forward(self, input_ids, attention_mask, user_features):
        # 特征提取
        text_features = self.text_extractor(input_ids, attention_mask)
        user_features = self.user_extractor(user_features)
        
        # 特征融合
        fused_features = self.fusion_module(text_features, user_features)
        
        # 分类
        logits = self.classifier(fused_features)
        
        return logits 