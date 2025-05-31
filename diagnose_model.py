#!/usr/bin/env python3
"""
模型诊断脚本 - 检查水军检测模型的预测行为
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

def main():
    print("=== 水军检测模型诊断 ===")
    
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
    
    # 测试不同类型的文本
    test_texts = [
        # 明显的水军文本
        "免费领取红包！！！点击链接立即领取！！！限时优惠！！！",
        "转发送iPhone！！！机会难得！！！马上行动！！！",
        "点赞关注送礼品！！！转发有奖！！！",
        "限时优惠！！！马上行动！！！立即购买！！！",
        "加微信领红包！！！免费赚钱！！！",
        
        # 正常文本
        "今天天气不错，心情很好。出去散了个步。",
        "刚看了一部电影，很有感触。剧情很棒。",
        "工作很累，但是很充实。明天继续努力。",
        "和朋友聚餐，聊了很多有趣的话题。",
        "读了一本好书，收获很大。推荐给大家。"
    ]
    
    print("\n=== 测试结果 ===")
    all_probs = []
    
    for i, text in enumerate(test_texts):
        # 预测
        inputs = tokenizer(text, padding='max_length', truncation=True, 
                          max_length=512, return_tensors='pt')
        
        with torch.no_grad():
            # 模型返回logits tensor
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            # 获取水军类别（索引1）的logits
            spammer_logit = logits[0, 1].item()
            # 应用softmax得到概率
            probs = torch.softmax(logits, dim=1)
            spammer_prob = probs[0, 1].item()
            
        all_probs.append(spammer_prob)
        category = "水军" if i < 5 else "正常"
        prediction = "水军" if spammer_prob > 0.5 else "正常"
        
        print(f"{i+1:2d}. [{category:2s}] {text[:40]:<40} -> logits: {spammer_logit:8.4f}, prob: {spammer_prob:.6f}, 预测: {prediction}")
    
    print(f"\n=== 统计信息 ===")
    print(f"所有概率的范围: {min(all_probs):.6f} - {max(all_probs):.6f}")
    print(f"概率平均值: {sum(all_probs)/len(all_probs):.6f}")
    print(f"概率标准差: {pd.Series(all_probs).std():.6f}")
    
    # 检查是否所有概率都太低
    if max(all_probs) < 0.01:
        print("\n❌ 问题诊断: 所有预测概率都过低！可能的原因：")
        print("   1. 模型输出层有问题（sigmoid函数异常）")
        print("   2. 模型训练时的标签有问题（可能标签被反转）")
        print("   3. 模型权重没有正确加载")
        print("   4. 训练数据不平衡导致模型偏向预测负类")
    elif min(all_probs) > 0.99:
        print("\n❌ 问题诊断: 所有预测概率都过高！")
        print("   1. 模型可能过拟合")
        print("   2. 输出层计算有误")
    else:
        print("\n✅ 模型预测范围正常")
        
        # 检查预测准确性
        spammer_probs = all_probs[:5]  # 前5个是水军文本
        normal_probs = all_probs[5:]   # 后5个是正常文本
        
        print(f"水军文本平均概率: {sum(spammer_probs)/len(spammer_probs):.6f}")
        print(f"正常文本平均概率: {sum(normal_probs)/len(normal_probs):.6f}")
        
        if sum(spammer_probs)/len(spammer_probs) < sum(normal_probs)/len(normal_probs):
            print("⚠️  警告: 模型可能将水军和正常文本的标签搞反了！")

if __name__ == "__main__":
    main() 