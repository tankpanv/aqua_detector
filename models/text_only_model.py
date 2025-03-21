import torch
import torch.nn as nn
from transformers import BertModel

class TextOnlySpammerDetectionModel(nn.Module):
    """仅使用文本内容检测水军的模型"""
    
    def __init__(self, config):
        super(TextOnlySpammerDetectionModel, self).__init__()
        self.config = config
        
        # BERT编码器
        self.bert = BertModel.from_pretrained(config.BERT_MODEL_NAME)
        
        # 冻结BERT参数以加速训练（可选）
        if config.FREEZE_BERT:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # 文本特征处理
        self.text_dropout = nn.Dropout(0.3)
        self.text_classifier = nn.Linear(self.bert.config.hidden_size, 2)  # 二分类：正常/水军
        
    def forward(self, input_ids, attention_mask):
        """
        前向传播，仅使用文本特征
        input_ids: 文本输入ID
        attention_mask: 注意力掩码
        """
        # 提取BERT特征
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # 获取[CLS]标记的输出
        
        # 文本分类
        text_features = self.text_dropout(cls_output)
        logits = self.text_classifier(text_features)
        
        return logits 