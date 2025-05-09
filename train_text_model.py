#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import argparse
import platform
import traceback

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from models.text_only_model import TextOnlySpammerDetectionModel
from processors.data_processor import WeiboDataProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='训练文本专用模型')
    parser.add_argument('--gpu', action='store_true', help='强制使用GPU训练（如果可用）')
    parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的GPU ID（多GPU系统）')
    parser.add_argument('--min_users', type=int, help='设置最小训练用户数量')
    parser.add_argument('--batch_size', type=int, help='设置批处理大小')
    parser.add_argument('--force_balance', action='store_true', help='强制平衡正负样本')
    return parser.parse_args()

def train_text_model():
    """训练仅基于文本的水军检测模型，确保跨平台兼容"""
    args = parse_args()
    print("开始训练文本专用模型...")
    
    # 检查环境
    print(f"当前操作系统: {platform.system()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 加载配置
    config = Config()
    
    # 设备选择，确保跨平台一致性
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu_id}")
            print(f"使用GPU训练: {torch.cuda.get_device_name(args.gpu_id)}")
            # 添加内存优化
            if 'Windows' in platform.system():
                # Windows上的CUDA内存优化
                print("在Windows环境下应用CUDA内存优化设置")
                torch.cuda.empty_cache()
        else:
            print("警告: 请求使用GPU但CUDA不可用，回退到CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"自动选择GPU: {torch.cuda.get_device_name(0)}")
    
    print(f"使用设备: {device}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(config.TEXT_MODEL_PATH), exist_ok=True)
    
    # 将命令行参数应用到配置
    if args.min_users:
        config.MIN_TRAINING_USERS = args.min_users
        print(f"使用自定义最小训练用户数: {config.MIN_TRAINING_USERS}")
    
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
        print(f"使用自定义批处理大小: {config.BATCH_SIZE}")
    
    # 加载数据处理器
    try:
        data_processor = WeiboDataProcessor(config)
        data_processor.load_data()
        data_processor.prepare_features()
    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        traceback.print_exc()
        return False
    
    # 初始化 tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    
    # 准备数据集
    print("准备训练数据...")
    
    # 处理文本数据
    texts = []
    labels = []
    
    # 将每个用户的微博内容合并为一个样本
    for user_id, user_group in tqdm(data_processor.weibo_df.groupby('用户id')):
        # 获取用户标签（是否为水军）
        user_info = data_processor.user_df[data_processor.user_df['id'] == user_id]
        if user_info.empty:
            continue
        
        label = int(user_info['is_spammer'].iloc[0])
        
        # 获取用户的所有微博内容
        user_texts = user_group['内容'].tolist()
        
        # 使用每一条微博作为单独样本（这会产生更多的训练数据）
        for text in user_texts:
            if not isinstance(text, str) or not text.strip():
                continue
                
            processed_text = data_processor.preprocess_text(text)
            texts.append(processed_text)
            labels.append(label)
    
    # 编码文本
    print("编码文本...")
    encoded_texts = []
    
    for text in tqdm(texts):
        encoded = tokenizer.encode_plus(
            text,
            max_length=config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        encoded_texts.append({
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        })
    
    # 将数据分割为训练集和验证集
    train_ratio = 0.8
    train_size = int(len(texts) * train_ratio)
    
    train_input_ids = torch.stack([item['input_ids'] for item in encoded_texts[:train_size]])
    train_attention_mask = torch.stack([item['attention_mask'] for item in encoded_texts[:train_size]])
    train_labels = torch.tensor(labels[:train_size])
    
    val_input_ids = torch.stack([item['input_ids'] for item in encoded_texts[train_size:]])
    val_attention_mask = torch.stack([item['attention_mask'] for item in encoded_texts[train_size:]])
    val_labels = torch.tensor(labels[train_size:])
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # 初始化模型
    print("初始化模型...")
    model = TextOnlySpammerDetectionModel(config)
    model = model.to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 训练模型
    print("开始训练...")
    num_epochs = 3
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                
                # 前向传播
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # 获取预测结果
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 计算评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        # 保存最佳模型
        if f1 > best_val_f1:
            best_val_f1 = f1
            
            # 确保保存目录存在
            os.makedirs('models/saved', exist_ok=True)
            
            # 保存模型
            model_path = 'models/saved/text_only_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
            }, model_path)
            
            print(f"保存最佳模型到 {model_path} (F1: {best_val_f1:.4f})")
    
    print("训练完成!")
    return True

if __name__ == "__main__":
    train_text_model() 