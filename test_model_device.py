import os
import torch
import argparse
from config import Config
from models.multi_view_model import MultiViewSpammerDetectionModel
from models.ensemble_model import create_ensemble_from_checkpoints
from transformers import BertTokenizer

def test_device_compatibility():
    """测试模型在不同设备上的兼容性"""
    parser = argparse.ArgumentParser(description='测试模型在不同设备上的兼容性')
    parser.add_argument('--cpu', action='store_true', help='强制使用CPU')
    parser.add_argument('--model', choices=['base', 'ensemble'], default='ensemble', help='要测试的模型类型')
    args = parser.parse_args()
    
    # 设置设备
    if args.cpu:
        device = torch.device("cpu")
        print("使用CPU测试模型")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("无可用GPU，使用CPU测试")
    
    # 加载配置
    config = Config()
    
    # 创建模拟输入数据
    print("创建模拟输入数据...")
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    
    # 模拟文本输入
    text = "这是一条测试微博，用于检查模型在不同设备上的兼容性"
    encoded = tokenizer.encode_plus(
        text,
        max_length=config.MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    # 模拟用户特征
    user_features = torch.randn(1, 7 + 24).to(device)  # 7个用户特征 + 24小时分布
    
    # 选择要测试的模型类型
    if args.model == 'base':
        print("测试基础模型...")
        model_path = os.path.join('models/saved/best_model.pt')
        
        if os.path.exists(model_path):
            try:
                print(f"加载模型: {model_path}")
                model = MultiViewSpammerDetectionModel(config)
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                model.eval()
                
                print("测试前向传播...")
                with torch.no_grad():
                    output = model(input_ids, attention_mask, user_features)
                    probs = torch.softmax(output, dim=1)
                    print(f"输出形状: {output.shape}")
                    print(f"预测概率: {probs[0]}")
                    print("前向传播成功!")
            except Exception as e:
                print(f"测试基础模型失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"找不到模型文件: {model_path}")
    
    else:  # ensemble
        print("测试集成模型...")
        # 查找变体模型
        variant_dir = os.path.join('models/saved/variants')
        if os.path.exists(variant_dir):
            variant_models = [
                os.path.join(variant_dir, f) 
                for f in os.listdir(variant_dir) 
                if f.endswith('_best.pt')
            ][:4]
            
            if len(variant_models) > 0:
                try:
                    print(f"找到 {len(variant_models)} 个变体模型: {variant_models}")
                    ensemble = create_ensemble_from_checkpoints(config, variant_models, device)
                    ensemble.eval()
                    
                    print("测试前向传播...")
                    with torch.no_grad():
                        # 测试普通前向传播
                        output = ensemble(input_ids, attention_mask, user_features)
                        probs = torch.softmax(output, dim=1)
                        print(f"输出形状: {output.shape}")
                        print(f"预测概率: {probs[0]}")
                        
                        # 测试校准预测
                        result = ensemble.predict_with_calibration(input_ids, attention_mask, user_features)
                        print("校准预测结果:")
                        print(f"预测类别: {result['prediction'].item()}")
                        print(f"置信度: {result['confidence'].item():.4f}")
                        print(f"模型一致性: {result['agreement'].item():.4f}")
                        print("前向传播和校准预测成功!")
                except Exception as e:
                    print(f"测试集成模型失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"在 {variant_dir} 中未找到变体模型")
        else:
            print(f"找不到变体模型目录: {variant_dir}")
    
    print("测试完成!")

if __name__ == "__main__":
    test_device_compatibility() 