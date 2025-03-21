import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse
import datetime
import json

from config import Config
from processors.data_processor import WeiboDataProcessor
from models.multi_view_model import MultiViewSpammerDetectionModel
from models.ensemble_model import BaseModelVariant, EnsembleModel, create_ensemble_from_checkpoints

def create_model_variants(config, device):
    """创建不同的模型变体"""
    variants = {
        'default': BaseModelVariant(config, 'default', device=device),
        'text_focus': BaseModelVariant(config, 'text_focus', device=device),
        'user_focus': BaseModelVariant(config, 'user_focus', device=device),
        'balanced': BaseModelVariant(config, 'balanced', device=device)
    }
    return variants

def train_single_model(model, train_loader, val_loader, config, device, model_name):
    """训练单个模型变体"""
    print(f"\n开始训练模型变体: {model_name}")
    
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
    patience_counter = 0
    max_patience = 5  # 提前停止的耐心值
    
    # 创建模型保存目录
    save_dir = f'models/saved/variants'
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        
        # 训练阶段
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training {model_name}")
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
                print("警告: 损失值为NaN，跳过此批次")
                continue
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        val_metrics = validate_model(model, val_loader, criterion, device)
        
        # 更新训练历史
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}, F1 Score: {val_metrics['f1']:.4f}")
        
        # 学习率调度
        scheduler.step(val_metrics['loss'])
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            print(f"新的最佳F1分数: {best_val_f1:.4f}, 保存模型...")
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_metrics['f1'],
                'config': config.__dict__
            }, f'{save_dir}/{model_name}_best.pt')
        else:
            patience_counter += 1
            
        # 提前停止
        if patience_counter >= max_patience:
            print(f"F1分数 {max_patience} 个epoch没有提升，提前停止训练")
            break
    
    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_f1': val_metrics['f1'],
        'config': config.__dict__
    }, f'{save_dir}/{model_name}_final.pt')
    
    print(f"模型变体 {model_name} 训练完成，最佳F1分数: {best_val_f1:.4f}")
    
    return history, best_val_f1

def validate_model(model, val_loader, criterion, device):
    """验证模型性能"""
    model.eval()
    val_loss = 0
    val_preds = []
    val_true = []
    
    with torch.no_grad():
        for batch in val_loader:
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
    
    # 计算验证集指标
    metrics = {
        'loss': val_loss / len(val_loader),
        'accuracy': accuracy_score(val_true, val_preds),
        'precision': precision_score(val_true, val_preds, zero_division=0),
        'recall': recall_score(val_true, val_preds, zero_division=0),
        'f1': f1_score(val_true, val_preds, zero_division=0)
    }
    
    return metrics

def train_ensemble_model(train_loader, val_loader, config, device, base_model_paths):
    """训练集成模型"""
    print("\n开始训练集成模型")
    
    # 创建集成模型
    ensemble = create_ensemble_from_checkpoints(config, base_model_paths, device)
    # 确保整个模型在正确的设备上
    ensemble = ensemble.to(device)
    
    # 确保基础模型也在正确的设备上
    for model in ensemble.models:
        model = model.to(device)
    
    # 冻结基础模型参数
    for model in ensemble.models:
        for param in model.parameters():
            param.requires_grad = False
    
    # 只训练融合层和温度缩放参数
    trainable_params = list(ensemble.fusion.parameters()) + [ensemble.temp_scaling]
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params, lr=0.001)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    best_val_f1 = 0
    
    # 创建模型保存目录
    save_dir = 'models/saved/ensemble'
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练循环
    for epoch in range(10):  # 集成模型训练10个epoch
        print(f"Epoch {epoch+1}/10")
        
        # 训练阶段
        ensemble.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training Ensemble")
        for batch in progress_bar:
            # 确保所有数据都在正确的设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_features = batch['user_features'].to(device)
            labels = batch['label'].to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = ensemble(input_ids, attention_mask, user_features)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 验证阶段
        ensemble.eval()
        val_results = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating Ensemble"):
                # 确保所有数据都在正确的设备上
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                user_features = batch['user_features'].to(device)
                labels = batch['label'].to(device)
                
                # 使用校准的预测方法
                result = ensemble.predict_with_calibration(
                    input_ids, attention_mask, user_features
                )
                
                predicted = result['prediction']
                val_results.append({
                    'pred': predicted.cpu().numpy(),
                    'true': labels.cpu().numpy(),
                    'conf': result['confidence'].cpu().numpy(),
                    'agreement': result['agreement'].cpu().numpy()
                })
        
        # 合并所有批次的结果
        all_preds = np.concatenate([r['pred'] for r in val_results])
        all_true = np.concatenate([r['true'] for r in val_results])
        all_conf = np.concatenate([r['conf'] for r in val_results])
        all_agreement = np.concatenate([r['agreement'] for r in val_results])
        
        # 计算指标
        val_metrics = {
            'accuracy': accuracy_score(all_true, all_preds),
            'precision': precision_score(all_true, all_preds, zero_division=0),
            'recall': recall_score(all_true, all_preds, zero_division=0),
            'f1': f1_score(all_true, all_preds, zero_division=0),
            'avg_confidence': np.mean(all_conf),
            'avg_agreement': np.mean(all_agreement)
        }
        
        history['val_metrics'].append(val_metrics)
        
        print(f"Epoch {epoch+1}/10")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}, F1 Score: {val_metrics['f1']:.4f}")
        print(f"Avg Confidence: {val_metrics['avg_confidence']:.4f}, Avg Agreement: {val_metrics['avg_agreement']:.4f}")
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            print(f"新的最佳F1分数: {best_val_f1:.4f}, 保存模型...")
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': ensemble.state_dict(),
                'base_model_paths': base_model_paths,
                'val_f1': val_metrics['f1'],
                'config': config.__dict__
            }, f'{save_dir}/ensemble_best.pt')
    
    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': ensemble.state_dict(),
        'base_model_paths': base_model_paths,
        'val_f1': val_metrics['f1'],
        'config': config.__dict__
    }, f'{save_dir}/ensemble_final.pt')
    
    print(f"集成模型训练完成，最佳F1分数: {best_val_f1:.4f}")
    
    return history, best_val_f1

def plot_ensemble_comparison(variants_metrics, ensemble_metrics):
    """绘制变体模型和集成模型的性能比较"""
    plt.figure(figsize=(15, 10))
    
    # F1分数比较
    plt.subplot(2, 2, 1)
    for name, metrics in variants_metrics.items():
        plt.plot(metrics['val_f1'], label=f'{name}')
    plt.plot([0, len(ensemble_metrics['val_metrics'])-1], 
             [ensemble_metrics['val_metrics'][0]['f1'], ensemble_metrics['val_metrics'][-1]['f1']], 
             'r--', label='Ensemble')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.title('F1 Score Comparison')
    
    # 准确率比较
    plt.subplot(2, 2, 2)
    for name, metrics in variants_metrics.items():
        plt.plot(metrics['val_accuracy'], label=f'{name}')
    plt.plot([0, len(ensemble_metrics['val_metrics'])-1], 
             [ensemble_metrics['val_metrics'][0]['accuracy'], ensemble_metrics['val_metrics'][-1]['accuracy']], 
             'r--', label='Ensemble')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Comparison')
    
    # 精确率和召回率比较
    plt.subplot(2, 2, 3)
    for name, metrics in variants_metrics.items():
        plt.plot(metrics['val_precision'], linestyle='-', label=f'{name} Precision')
        plt.plot(metrics['val_recall'], linestyle='--', label=f'{name} Recall')
    plt.plot([0, len(ensemble_metrics['val_metrics'])-1], 
             [ensemble_metrics['val_metrics'][0]['precision'], ensemble_metrics['val_metrics'][-1]['precision']], 
             'r-', label='Ensemble Precision')
    plt.plot([0, len(ensemble_metrics['val_metrics'])-1], 
             [ensemble_metrics['val_metrics'][0]['recall'], ensemble_metrics['val_metrics'][-1]['recall']], 
             'r--', label='Ensemble Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision and Recall Comparison')
    
    # 集成模型的置信度和一致性
    plt.subplot(2, 2, 4)
    ensemble_epochs = range(len(ensemble_metrics['val_metrics']))
    plt.plot(ensemble_epochs, 
             [m['avg_confidence'] for m in ensemble_metrics['val_metrics']], 
             'g-', label='Avg Confidence')
    plt.plot(ensemble_epochs, 
             [m['avg_agreement'] for m in ensemble_metrics['val_metrics']], 
             'b-', label='Avg Agreement')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Ensemble Confidence and Agreement')
    
    plt.tight_layout()
    plt.savefig('ensemble_comparison.png')
    plt.close()

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练集成学习模型')
    parser.add_argument('--train_variants', action='store_true', help='是否训练基础模型变体')
    parser.add_argument('--train_ensemble', action='store_true', help='是否训练集成模型')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU训练')
    parser.add_argument('--batch_size', type=int, default=None, help='指定批处理大小，可以减小以避免内存不足')
    args = parser.parse_args()
    
    # 如果没有指定任何操作，默认全部执行
    if not args.train_variants and not args.train_ensemble:
        args.train_variants = True
        args.train_ensemble = True
    
    # 设置配置和设备
    config = Config()
    
    # 如果指定了批处理大小，替换配置中的值
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
        print(f"使用自定义批处理大小: {config.BATCH_SIZE}")
    
    # 设置设备
    if args.cpu:
        device = torch.device("cpu")
        print("强制使用CPU进行训练")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            # 输出GPU信息
            gpu_properties = torch.cuda.get_device_properties(0)
            print(f"使用GPU: {gpu_properties.name}, 总内存: {gpu_properties.total_memory / 1024**3:.2f}GB")
            # 清理缓存
            torch.cuda.empty_cache()
        else:
            print("无可用GPU，使用CPU训练")
    
    # 数据处理
    print("加载数据...")
    data_processor = WeiboDataProcessor(config)
    data_processor.load_data()
    data_processor.prepare_features()
    train_loader, val_loader, _ = data_processor.get_dataloaders()
    
    # 创建结果目录
    os.makedirs('models/saved/variants', exist_ok=True)
    os.makedirs('models/saved/ensemble', exist_ok=True)
    
    # 记录训练时间
    start_time = datetime.datetime.now()
    print(f"开始训练: {start_time}")
    
    variants_histories = {}
    best_variant_paths = {}
    
    # 训练基础模型变体
    if args.train_variants:
        print("\n=== 训练基础模型变体 ===")
        try:
            variants = create_model_variants(config, device)
            
            for name, model in variants.items():
                history, best_f1 = train_single_model(
                    model, train_loader, val_loader, config, device, name
                )
                variants_histories[name] = history
                best_variant_paths[name] = f'models/saved/variants/{name}_best.pt'
        except Exception as e:
            print(f"训练基础模型变体时出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 如果不训练变体，使用现有的最佳模型
        print("\n=== 使用现有的基础模型变体 ===")
        variant_names = ['default', 'text_focus', 'user_focus', 'balanced']
        for name in variant_names:
            path = f'models/saved/variants/{name}_best.pt'
            if os.path.exists(path):
                best_variant_paths[name] = path
                print(f"使用现有模型: {path}")
            else:
                print(f"警告: 找不到模型 {path}")
    
    # 训练集成模型
    if args.train_ensemble and len(best_variant_paths) > 0:
        print("\n=== 训练集成模型 ===")
        try:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            base_model_paths = list(best_variant_paths.values())
            ensemble_history, ensemble_best_f1 = train_ensemble_model(
                train_loader, val_loader, config, device, base_model_paths
            )
            
            # 如果我们有基础模型的历史记录，绘制比较图
            if variants_histories:
                plot_ensemble_comparison(variants_histories, ensemble_history)
        except Exception as e:
            print(f"训练集成模型时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 记录结束时间
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print(f"\n训练结束: {end_time}")
    print(f"总训练时间: {duration}")
    
    # 保存训练结果摘要
    with open('training_summary.json', 'w') as f:
        summary = {
            'start_time': str(start_time),
            'end_time': str(end_time),
            'duration': str(duration),
            'device': str(device),
            'best_models': best_variant_paths,
            'ensemble_model': 'models/saved/ensemble/ensemble_best.pt' if args.train_ensemble else None
        }
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    main() 