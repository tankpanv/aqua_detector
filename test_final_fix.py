#!/usr/bin/env python3
"""
测试最终修复后的完整逻辑
"""

import sys
import os
sys.path.append('.')
os.chdir('/home/ubuntu/workspace/aqua_detector')

import torch
import numpy as np
from transformers import BertTokenizer
from config import Config
from models.text_only_model import TextOnlySpammerDetectionModel

def test_final_logic():
    print("=== 测试最终修复逻辑 ===")
    
    # 加载配置和模型
    config = Config()
    device = torch.device('cpu')
    
    model = TextOnlySpammerDetectionModel(config)
    checkpoint = torch.load(config.TEXT_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    
    # 测试文本
    test_texts = [
        ("免费领取红包！！！点击链接立即领取！！！限时优惠！！！", "水军"),
        ("转发送iPhone！！！机会难得！！！马上行动！！！", "水军"),
        ("今天天气不错，心情很好。出去散了个步。", "正常"),
        ("刚看了一部电影，很有感触。剧情很棒。", "正常")
    ]
    
    SPAMMER_THRESHOLD = 0.5
    CONFIDENCE_CALIBRATION = 2.0
    
    print(f"使用阈值: {SPAMMER_THRESHOLD}")
    print(f"置信度校准因子: {CONFIDENCE_CALIBRATION}")
    print()
    
    for i, (text, expected) in enumerate(test_texts):
        inputs = tokenizer(text, padding='max_length', truncation=True, 
                          max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(outputs, dim=1)
            
            # 应用最终修复逻辑
            prob_spammer_raw = probabilities[0][0].item()
            is_spammer_prediction = prob_spammer_raw > 0.7
            
            # 转换为0-1概率
            if is_spammer_prediction:
                prob_spammer = min(0.95, prob_spammer_raw)
            else:
                prob_spammer = max(0.05, 1 - prob_spammer_raw)
            
            pred_class = 1 if prob_spammer > SPAMMER_THRESHOLD else 0
            prediction = "水军" if pred_class == 1 else "正常"
            
            # 计算置信度
            raw_confidence = prob_spammer if pred_class == 1 else (1 - prob_spammer)
            
            if np.isnan(raw_confidence) or raw_confidence is None:
                raw_confidence = 0.5
                
            if raw_confidence > 0.5:
                confidence = 0.5 + (raw_confidence - 0.5) * CONFIDENCE_CALIBRATION
            else:
                confidence = 0.5 - (0.5 - raw_confidence) * CONFIDENCE_CALIBRATION
            
            confidence = max(0.0, min(1.0, confidence))
            
            if np.isnan(confidence):
                confidence = 0.5
                
            confidence_percent = round(float(confidence * 100))
            
            result_icon = "✅" if prediction == expected else "❌"
            
            print(f"{i+1}. [{expected:2s}] {text[:40]:<40}")
            print(f"    原始索引0概率: {prob_spammer_raw:.6f}")
            print(f"    是否>0.7: {is_spammer_prediction}")
            print(f"    转换后概率: {prob_spammer:.6f}")
            print(f"    预测结果: {prediction} (置信度: {confidence_percent}%)")
            print(f"    结果: {result_icon}")
            print()

if __name__ == "__main__":
    test_final_logic() 