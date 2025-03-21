import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

from config import Config
from processors.data_processor import WeiboDataProcessor
from models.multi_view_model import MultiViewSpammerDetectionModel

def evaluate_model(config):
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
    _, _, test_loader = data_processor.get_dataloaders()
    
    # 加载模型
    print("加载模型...")
    model = MultiViewSpammerDetectionModel(config)
    checkpoint = torch.load('models/saved/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 评估
    print("开始评估...")
    test_preds = []
    test_true = []
    test_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            user_features = batch['user_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask, user_features)
            probabilities = torch.softmax(outputs, dim=1)
            test_probs.extend(probabilities[:, 1].cpu().numpy())  # 正类的概率
            
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(test_true, test_preds)
    precision = precision_score(test_true, test_preds)
    recall = recall_score(test_true, test_preds)
    f1 = f1_score(test_true, test_preds)
    conf_matrix = confusion_matrix(test_true, test_preds)
    report = classification_report(test_true, test_preds)
    
    print(f"测试集结果:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\n混淆矩阵:")
    print(conf_matrix)
    print("\n分类报告:")
    print(report)
    
    # 绘制评估结果
    plot_evaluation_results(test_true, test_preds, test_probs, conf_matrix)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'report': report
    }

def plot_evaluation_results(true_labels, pred_labels, pred_probs, conf_matrix):
    """绘制评估结果"""
    # 创建图表
    plt.figure(figsize=(15, 10))
    
    # 1. 混淆矩阵
    plt.subplot(2, 2, 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['正常用户', '水军'], yticklabels=['正常用户', '水军'])
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    
    # 2. ROC曲线
    plt.subplot(2, 2, 2)
    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    
    # 3. 预测概率分布
    plt.subplot(2, 2, 3)
    sns.histplot(np.array(pred_probs), bins=20, kde=True)
    plt.title('预测概率分布')
    plt.xlabel('预测为水军的概率')
    plt.ylabel('频率')
    
    # 4. 按类别的预测概率
    plt.subplot(2, 2, 4)
    class_0_probs = [pred_probs[i] for i in range(len(true_labels)) if true_labels[i] == 0]
    class_1_probs = [pred_probs[i] for i in range(len(true_labels)) if true_labels[i] == 1]
    
    sns.histplot(class_0_probs, bins=20, kde=True, color='blue', label='正常用户')
    sns.histplot(class_1_probs, bins=20, kde=True, color='red', label='水军')
    plt.title('各类别预测概率分布')
    plt.xlabel('预测为水军的概率')
    plt.ylabel('频率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close()

if __name__ == "__main__":
    config = Config()
    results = evaluate_model(config) 