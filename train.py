import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import Config
from processors.data_processor import WeiboDataProcessor
from models.multi_view_model import MultiViewSpammerDetectionModel

# 创建全局配置实例
config = Config()

def train_model(config):
    # 设置随机种子
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据处理
    print("加载数据...")
    data_processor = WeiboDataProcessor(config)
    data_processor.load_data()
    data_processor.prepare_features()
    train_loader, val_loader, test_loader = data_processor.get_dataloaders()
    
    # 创建模型
    print("创建模型...")
    model = MultiViewSpammerDetectionModel(config)
    model = model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    best_val_f1 = 0
    
    # 创建模型保存目录
    os.makedirs('models/saved', exist_ok=True)
    
    # 训练循环
    print("开始训练...")
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_features = batch['user_features'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, user_features)
            loss = criterion(outputs, labels)
            
            # 检查损失是否为NaN
            if torch.isnan(loss).item():
                print("警告: 损失值为NaN，检查输入数据和模型输出")
                continue
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                user_features = batch['user_features'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask, user_features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # 计算验证集指标
        val_accuracy = accuracy_score(val_true, val_preds)
        val_precision = precision_score(val_true, val_preds, zero_division=0)
        val_recall = recall_score(val_true, val_preds, zero_division=0)
        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        
        history['val_accuracy'].append(val_accuracy)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}")
        print(f"Val Recall: {val_recall:.4f}, F1 Score: {val_f1:.4f}")
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"新的最佳F1分数: {best_val_f1:.4f}, 保存模型...")
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'config': config.__dict__
            }, 'models/saved/best_model.pt')
    
    # 绘制训练历史
    plot_history(history)
    
    # 保存最终模型，无论性能如何
    torch.save({
        'epoch': config.EPOCHS-1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_f1,
        'config': config.__dict__
    }, 'models/saved/final_model.pt')
    print(f"训练完成，最终模型已保存")
    
    return model, history

def plot_history(history):
    """绘制训练历史"""
    plt.figure(figsize=(12, 8))
    
    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['val_accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    
    # 精确率和召回率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision and Recall Curves')
    
    # F1分数曲线
    plt.subplot(2, 2, 4)
    plt.plot(history['val_f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score Curve')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_features = batch['user_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, user_features)
            _, predicted = torch.max(outputs, 1)
            
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
    
    # 计算测试集指标
    accuracy = accuracy_score(test_true, test_preds)
    precision = precision_score(test_true, test_preds, zero_division=0)
    recall = recall_score(test_true, test_preds, zero_division=0)
    f1 = f1_score(test_true, test_preds, zero_division=0)
    
    print("\n测试集评估结果:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

if __name__ == "__main__":
    # 训练模型
    model, history = train_model(config)
    
    # 创建数据处理器用于评估
    data_processor = WeiboDataProcessor(config)
    data_processor.load_data()
    data_processor.prepare_features()
    _, _, test_loader = data_processor.get_dataloaders()
    
    # 评估模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_results = evaluate_model(model, test_loader, device) 