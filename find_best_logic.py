#!/usr/bin/env python3
"""
找到最佳的预测逻辑
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

def find_best_logic():
    print("=== 寻找最佳预测逻辑 ===")
    
    # 加载配置和模型
    config = Config()
    device = torch.device('cpu')
    
    model = TextOnlySpammerDetectionModel(config)
    checkpoint = torch.load(config.TEXT_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    
    # 测试文本（前5个是水军，后5个是正常）
    test_texts = [
        "免费领取红包！！！点击链接立即领取！！！限时优惠！！！",
        "转发送iPhone！！！机会难得！！！马上行动！！！",
        "点赞关注送礼品！！！转发有奖！！！",
        "限时优惠！！！马上行动！！！立即购买！！！",
        "加微信领红包！！！免费赚钱！！！",
        
        "今天天气不错，心情很好。出去散了个步。",
        "刚看了一部电影，很有感触。剧情很棒。",
        "工作很累，但是很充实。明天继续努力。",
        "和朋友聚餐，聊了很多有趣的话题。",
        "读了一本好书，收获很大。推荐给大家。"
    ]
    
    true_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1=水军, 0=正常
    
    # 收集所有预测
    probs_0 = []
    probs_1 = []
    
    for text in test_texts:
        inputs = tokenizer(text, padding='max_length', truncation=True, 
                          max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probabilities = torch.softmax(outputs, dim=1)
            
            probs_0.append(probabilities[0][0].item())
            probs_1.append(probabilities[0][1].item())
    
    print("索引0的概率分布:")
    print(f"  水军文本: {probs_0[:5]}")
    print(f"  正常文本: {probs_0[5:]}")
    
    print("\n索引1的概率分布:")
    print(f"  水军文本: {probs_1[:5]}")
    print(f"  正常文本: {probs_1[5:]}")
    
    # 测试不同的预测逻辑
    strategies = [
        ("索引1 > 0.5", lambda p0, p1: p1 > 0.5),
        ("索引1 > 0.1", lambda p0, p1: p1 > 0.1),
        ("索引1 > 0.05", lambda p0, p1: p1 > 0.05),
        ("索引1 > 0.02", lambda p0, p1: p1 > 0.02),
        ("索引0 > 0.5", lambda p0, p1: p0 > 0.5),
        ("索引0 > 0.7", lambda p0, p1: p0 > 0.7),
        ("索引0 > 0.8", lambda p0, p1: p0 > 0.8),
        ("索引0 > 0.9", lambda p0, p1: p0 > 0.9),
        ("索引0 < 0.5", lambda p0, p1: p0 < 0.5),  # 反向逻辑
        ("索引1 < 0.5", lambda p0, p1: p1 < 0.5),  # 反向逻辑
    ]
    
    print("\n=== 不同策略的准确率 ===")
    best_accuracy = 0
    best_strategy = None
    
    for name, strategy in strategies:
        predictions = [1 if strategy(p0, p1) else 0 for p0, p1 in zip(probs_0, probs_1)]
        correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
        accuracy = correct / len(true_labels)
        
        print(f"{name:<15}: 准确率 {accuracy:.2f} ({correct}/{len(true_labels)}) - 预测: {predictions}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_strategy = name
    
    print(f"\n🏆 最佳策略: {best_strategy} (准确率: {best_accuracy:.2f})")

if __name__ == "__main__":
    find_best_logic() 