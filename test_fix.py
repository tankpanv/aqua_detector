#!/usr/bin/env python3
"""
测试修复后的预测逻辑
"""

import sys
import os
sys.path.append('.')
os.chdir('/home/ubuntu/workspace/aqua_detector')

import torch
import pandas as pd
from transformers import BertTokenizer
from config import Config
from models.text_only_model import TextOnlySpammerDetectionModel

def test_fixed_prediction():
    print("=== 测试修复后的预测逻辑 ===")
    
    # 加载配置和模型
    config = Config()
    device = torch.device('cpu')
    
    print(f"加载模型: {config.TEXT_MODEL_PATH}")
    model = TextOnlySpammerDetectionModel(config)
    checkpoint = torch.load(config.TEXT_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载tokenizer
    print("加载tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    
    # 测试文本
    test_texts = [
        "免费领取红包！！！点击链接立即领取！！！限时优惠！！！",  # 明显水军
        "转发送iPhone！！！机会难得！！！马上行动！！！",        # 明显水军
        "今天天气不错，心情很好。出去散了个步。",             # 正常文本
        "刚看了一部电影，很有感触。剧情很棒。"              # 正常文本
    ]
    
    SPAMMER_THRESHOLD = 0.5
    
    print("\n=== 原始预测（索引1）===")
    for i, text in enumerate(test_texts):
        inputs = tokenizer(text, padding='max_length', truncation=True, 
                          max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(outputs, dim=1)
            
            # 原始方法（使用索引1）
            prob_spammer_old = probabilities[0][1].item()
            pred_old = "水军" if prob_spammer_old > SPAMMER_THRESHOLD else "正常"
            
            # 修复方法（使用索引0）
            prob_spammer_new = probabilities[0][0].item()
            pred_new = "水军" if prob_spammer_new > SPAMMER_THRESHOLD else "正常"
            
            text_type = "水军" if i < 2 else "正常"
            
            print(f"{i+1}. [{text_type:2s}] {text[:30]:<30}")
            print(f"    原始方法(索引1): {prob_spammer_old:.6f} -> {pred_old}")
            print(f"    修复方法(索引0): {prob_spammer_new:.6f} -> {pred_new}")
            print(f"    修复是否正确: {'✅' if pred_new == text_type else '❌'}")
            print()

if __name__ == "__main__":
    test_fixed_prediction() 