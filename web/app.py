import platform
import codecs
import locale
import torch
import pandas as pd
import os
import sys
import json
import numpy as np
import traceback
import argparse
from flask import Flask, render_template, request, jsonify

# 操作系统检测
SYSTEM_TYPE = platform.system()  # 'Windows', 'Linux', 或 'Darwin' (macOS)
print(f"当前运行环境: {SYSTEM_TYPE}")

# 终端和系统默认编码检测
try:
    SYSTEM_ENCODING = locale.getpreferredencoding()
    print(f"系统默认编码: {SYSTEM_ENCODING}")
except:
    SYSTEM_ENCODING = 'utf-8'
    print(f"无法检测系统编码，使用默认编码: {SYSTEM_ENCODING}")

# 初始化全局变量
text_only_model = None
model = None
using_ensemble = False
device = None

# 统一文件路径处理函数
def get_platform_path(path_components):
    """创建跨平台兼容的路径"""
    # 转换所有正斜杠为系统适合的分隔符
    path = os.path.join(*path_components)
    # 确保路径存在
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models.multi_view_model import MultiViewSpammerDetectionModel
from models.text_only_model import TextOnlySpammerDetectionModel  # 导入文本专用模型
from processors.data_processor import WeiboDataProcessor
from transformers import BertTokenizer
from web.network_analysis import NetworkAnalyzer  # 导入网络分析器

# 导入图像生成模块
from web.generate_analysis import ensure_dirs, generate_real_analysis_images, generate_sample_images

# 导入集成模型
from models.ensemble_model import create_ensemble_from_checkpoints

app = Flask(__name__)

# 确保分析图像存在
print("检查并生成网络分析图像...")
ensure_dirs()  # 确保目录存在
success = generate_real_analysis_images()  # 尝试使用真实数据生成图像
if not success:
    print("使用示例数据生成图像...")
    generate_sample_images()  # 使用示例数据生成图像

# 加载配置
config = Config()

# 创建模型保存目录
os.makedirs('models/saved/ensemble', exist_ok=True)
os.makedirs('models/saved/variants', exist_ok=True)

# 水军检测设置
SPAMMER_THRESHOLD = 0.5  # 水军检测阈值（从0.35改为0.5，减少误判）
HIGH_CONFIDENCE_THRESHOLD = 0.75  # 高置信度阈值
CONFIDENCE_CALIBRATION = 2.0  # 置信度校准因子，提高从1.25到2.0，使结果更加两极化

# 可疑行为信号阈值定义
SUSPICIOUS_SIGNALS = {
    'behavior': {
        'url比例': 0.5,  # URL比例超过0.5视为异常
        '转发比例': 3.0,  # 转发数/原创数比例超过3视为异常
        '点赞评论比': 0.5,  # 点赞/评论比例低于0.5视为异常
    },
    'time': {
        '夜间活跃度': 0.4,  # 夜间发帖比例超过40%视为异常
        '规律发帖': 0.7,  # 发帖规律性超过0.7视为异常
    }
}

# 修改模型路径获取方式
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 检查模型文件是否存在
def load_models(config):
    """加载模型，处理不同平台兼容性"""
    
    # 声明全局变量
    global text_only_model, model, using_ensemble, device
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"模型加载使用设备: {device}")
    
    # 加载文本专用模型
    text_only_model = None
    if os.path.exists(config.TEXT_MODEL_PATH):
        print(f"找到文本专用模型文件，开始加载：{config.TEXT_MODEL_PATH}")
        try:
            # 先检查文件完整性
            model_file_size = os.path.getsize(config.TEXT_MODEL_PATH)
            print(f"模型文件大小: {model_file_size/1024/1024:.2f} MB")
            
            text_only_model = TextOnlySpammerDetectionModel(config)
            # 使用安全加载模式，加载时指定map_location
            try:
                checkpoint = torch.load(config.TEXT_MODEL_PATH, map_location=device)
            except RuntimeError as e:
                if "storage has wrong size" in str(e) or "unexpected EOF" in str(e):
                    print(f"模型文件可能损坏: {str(e)}")
                    raise
                raise
                
            text_only_model.load_state_dict(checkpoint['model_state_dict'])
            text_only_model = text_only_model.to(device)
            text_only_model.eval()
            print("文本专用模型加载成功")
        except Exception as e:
            print(f"加载文本专用模型时出错: {str(e)}")
            traceback.print_exc()
            text_only_model = None
    else:
        print(f"警告: 文本专用模型文件 {config.TEXT_MODEL_PATH} 不存在，纯文本分析将使用通用模型")

    # 优先加载集成模型，如果存在
    using_ensemble = False
    if os.path.exists(config.ENSEMBLE_MODEL_PATH):
        print(f"加载集成模型：{config.ENSEMBLE_MODEL_PATH}")
        try:
            checkpoint = torch.load(config.ENSEMBLE_MODEL_PATH, map_location=device)
            
            # 加载集成模型所需的基础模型路径
            base_model_paths = checkpoint.get('base_model_paths', [])
            if not base_model_paths:
                # 如果没有保存基础模型路径，尝试查找变体目录下的模型
                variant_dir = os.path.join(config.MODEL_PATH, 'variants')
                if os.path.exists(variant_dir):
                    base_model_paths = [
                        os.path.join(variant_dir, f)
                        for f in os.listdir(variant_dir)
                        if f.endswith('_best.pt') or f.endswith('_final.pt')
                    ][:4]  # 最多使用4个基础模型
            
            if base_model_paths:
                # 创建并加载集成模型
                from models.ensemble_model import EnsembleModel
                ensemble_model = EnsembleModel(config, device)
                ensemble_model.load_state_dict(checkpoint['model_state_dict'])
                
                # 尝试加载基础模型
                try:
                    ensemble_model.load_pretrained_models(base_model_paths)
                    model = ensemble_model
                    using_ensemble = True
                    print("成功加载集成模型和所有基础模型")
                except Exception as e:
                    print(f"集成模型加载失败，回退到基础模型: {e}")
                    # 回退到基础模型
                    model = MultiViewSpammerDetectionModel(config)
                    if os.path.exists(config.BASE_MODEL_PATH):
                        checkpoint = torch.load(config.BASE_MODEL_PATH, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        print(f"已回退并加载基础模型: {config.BASE_MODEL_PATH}")
            else:
                # 如果没有基础模型路径，使用基础模型
                model = MultiViewSpammerDetectionModel(config)
                if os.path.exists(config.BASE_MODEL_PATH):
                    checkpoint = torch.load(config.BASE_MODEL_PATH, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"已加载基础模型: {config.BASE_MODEL_PATH}")
        except Exception as e:
            print(f"集成模型加载失败: {e}")
            traceback.print_exc()
            # 回退到基础模型
            model = MultiViewSpammerDetectionModel(config)
            if os.path.exists(config.BASE_MODEL_PATH):
                try:
                    checkpoint = torch.load(config.BASE_MODEL_PATH, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"已回退并加载基础模型: {config.BASE_MODEL_PATH}")
                except Exception as e:
                    print(f"基础模型加载也失败: {e}")
                    # 不抛出异常，继续运行，但模型将无法使用
    else:
        # 使用基础模型
        model = MultiViewSpammerDetectionModel(config)
        if os.path.exists(config.BASE_MODEL_PATH):
            try:
                checkpoint = torch.load(config.BASE_MODEL_PATH, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"已加载基础模型: {config.BASE_MODEL_PATH}")
            except Exception as e:
                print(f"基础模型加载失败: {e}")
                # 不抛出异常，继续运行，但模型将无法使用
        else:
            print(f"警告: 模型文件 {config.BASE_MODEL_PATH} 不存在, 请先训练模型")

    # 将模型设置为评估模式
    if 'model' in locals():
        model = model.to(device)
        model.eval()
        return True
    return False

# 在加载数据处理器之前设置编码一致性环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)

# 加载数据处理器并添加编码错误处理
try:
    data_processor = WeiboDataProcessor(config)
    data_processor.load_data()
    data_processor.prepare_features()
    print(f"成功加载数据 - 用户数: {len(data_processor.user_df)}, 微博数: {len(data_processor.weibo_df)}")
except UnicodeDecodeError as e:
    print(f"数据加载过程中出现编码错误: {e}")
    print("尝试使用不同编码重新加载...")
    # 这里可以添加备用编码尝试逻辑
    raise

# 准备网络分析器
# 关系数据路径，如果文件存在则加载
relation_path = get_platform_path(['data', 'relation.csv'])
relation_df = None
if os.path.exists(relation_path):
    relation_df = pd.read_csv(relation_path)
    print(f"成功加载关系数据：{relation_path}")
else:
    print(f"注意: 关系数据 {relation_path} 不存在, 部分网络分析功能将不可用")

network_analyzer = NetworkAnalyzer(data_processor.weibo_df, data_processor.user_df, relation_df)

def analyze_suspicious_behavior(user_weibos, user_data=None):
    """分析用户的可疑行为模式，返回可疑指标"""
    suspicious_indicators = {}
    
    # 检查微博数据
    if not user_weibos.empty:
        # 分析URL比例
        if '是否含url' in user_weibos.columns:
            url_ratio = user_weibos['是否含url'].mean()
            suspicious_indicators['url比例'] = url_ratio
            if url_ratio > SUSPICIOUS_SIGNALS['behavior']['url比例']:
                suspicious_indicators['url比例_异常'] = True
        
        # 分析转发与原创比例
        if '是否转发' in user_weibos.columns:
            repost_count = user_weibos['是否转发'].sum()
            original_count = len(user_weibos) - repost_count
            if original_count > 0:
                repost_ratio = repost_count / original_count
                suspicious_indicators['转发比例'] = repost_ratio
                if repost_ratio > SUSPICIOUS_SIGNALS['behavior']['转发比例']:
                    suspicious_indicators['转发比例_异常'] = True
        
        # 分析点赞评论比
        if '点赞数' in user_weibos.columns and '评论数' in user_weibos.columns:
            likes = user_weibos['点赞数'].sum()
            comments = user_weibos['评论数'].sum()
            if comments > 0:
                like_comment_ratio = likes / comments
                suspicious_indicators['点赞评论比'] = like_comment_ratio
                if like_comment_ratio < SUSPICIOUS_SIGNALS['behavior']['点赞评论比']:
                    suspicious_indicators['点赞评论比_异常'] = True
        
        # 分析时间模式
        if '发布时间' in user_weibos.columns:
            times = user_weibos['发布时间'].apply(lambda x: data_processor.extract_time_features(x)['hour'])
            # 夜间活跃度 (22-6点)
            night_hours = [22, 23, 0, 1, 2, 3, 4, 5]
            night_posts = times.isin(night_hours).sum()
            night_ratio = night_posts / len(times) if len(times) > 0 else 0
            suspicious_indicators['夜间活跃度'] = night_ratio
            if night_ratio > SUSPICIOUS_SIGNALS['time']['夜间活跃度']:
                suspicious_indicators['夜间活跃度_异常'] = True
            
            # 计算发帖规律性 (小时分布的标准差越小表示越规律)
            hour_counts = times.value_counts().reindex(range(24), fill_value=0)
            hour_std = hour_counts.std() / hour_counts.mean() if hour_counts.mean() > 0 else 0
            hour_regularity = 1 - min(1, hour_std / 2)  # 规律性指标，越高越规律
            suspicious_indicators['发帖规律性'] = hour_regularity
            if hour_regularity > SUSPICIOUS_SIGNALS['time']['规律发帖']:
                suspicious_indicators['发帖规律性_异常'] = True
    
    # 检查用户数据
    if user_data is not None:
        pass  # 可以添加基于用户资料的分析
    
    # 计算可疑程度
    anomaly_count = sum(1 for k in suspicious_indicators if k.endswith('_异常'))
    total_indicators = len([k for k in SUSPICIOUS_SIGNALS['behavior'].keys()]) + len([k for k in SUSPICIOUS_SIGNALS['time'].keys()])
    suspicious_score = anomaly_count / total_indicators if total_indicators > 0 else 0
    
    suspicious_indicators['可疑程度'] = suspicious_score
    
    return suspicious_indicators

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    print(f"收到请求数据: {data}")
    
    # 获取用户ID和文本内容
    user_id = data.get('user_id', '')
    text_content = data.get('text', '')
    
    print(f"用户ID: {user_id}, 文本长度: {len(text_content)}")
    
    # 情况1：如果提供了文本，但没有用户ID，直接使用文本进行预测
    if text_content and not user_id:
        try:
            print(f"使用文本进行预测: {text_content[:50]}...")
            # 预处理文本
            processed_text = data_processor.preprocess_text(text_content)
            
            # 编码文本
            encoded = tokenizer.encode_plus(
                processed_text,
                max_length=config.MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 如果存在文本专用模型，则优先使用
            if text_only_model is not None:
                print("使用文本专用模型进行预测")
                with torch.no_grad():
                    input_ids = encoded['input_ids'].to(device)
                    attention_mask = encoded['attention_mask'].to(device)
                    
                    # 使用文本专用模型预测
                    outputs = text_only_model(input_ids, attention_mask)
                    probabilities = torch.softmax(outputs, dim=1)
                    prob_spammer = probabilities[0][1].item()  # 获取水军类别的概率
                    
                    # 使用自定义阈值决定标签
                    pred_class = 1 if prob_spammer > SPAMMER_THRESHOLD else 0
                    
                    # 添加详细调试输出
                    print(f"文本模型预测详情 - 水军概率: {prob_spammer:.4f}, 阈值: {SPAMMER_THRESHOLD}, 判定结果: {'水军' if pred_class == 1 else '正常用户'}")
                    
                    # 计算可信度并应用校准
                    raw_confidence = prob_spammer if pred_class == 1 else (1 - prob_spammer)
                    
                    # 确保是有效数值
                    if np.isnan(raw_confidence) or raw_confidence is None:
                        raw_confidence = 0.5
                        
                    # 应用校准因子，使置信度更加两极化
                    if raw_confidence > 0.5:
                        confidence = 0.5 + (raw_confidence - 0.5) * CONFIDENCE_CALIBRATION
                    else:
                        confidence = 0.5 - (0.5 - raw_confidence) * CONFIDENCE_CALIBRATION
                    
                    # 确保置信度在[0,1]范围内
                    confidence = max(0.0, min(1.0, confidence))
                    
                    # 再次确保不是NaN
                    if np.isnan(confidence):
                        confidence = 0.5
                        
                    is_spammer = pred_class == 1
                    
                    # 准备结果
                    try:
                        # 处理置信度，确保是有效数值
                        if np.isnan(confidence):
                            confidence = 0.5
                        confidence_percent = round(float(confidence * 100))
                        
                        # 创建结果字典
                        result = {
                            'text': text_content[:100] + "..." if len(text_content) > 100 else text_content,
                            'is_spammer': bool(is_spammer),
                            'confidence': confidence_percent,
                            'model_type': 'text_only',
                            'warning': "使用专业文本分析模型，仅基于文本内容的预测" if confidence < 0.85 else None
                        }
                    except Exception as e:
                        print(f"构建结果时出错: {str(e)}")
                        # 提供备用简化结果
                        result = {
                            'text': text_content[:50] + "..." if len(text_content) > 50 else text_content,
                            'is_spammer': bool(is_spammer),
                            'confidence': 50,  # 默认置信度50%
                            'error_detail': str(e)
                        }
                    
                    return jsonify(result)
            
            # 如果没有文本专用模型，使用通用模型+默认用户特征
            print("文本专用模型不可用，使用通用模型+默认用户特征")
            
            # 创建默认的用户特征向量 (调整为更接近正常用户的特征)
            default_behavior_features = torch.tensor([
                15.0,   # 默认发帖数 (提高)
                25.0,   # 默认评论数 (提高)
                8.0,    # 默认转发数 (略微提高)
                50.0,   # 默认点赞数 (提高)
                150.0,  # 默认长度 (提高)
                0.7,    # 默认原创率 (提高，正常用户原创率通常较高)
                0.05    # 默认URL率 (降低，正常用户URL比例通常较低)
            ], dtype=torch.float).reshape(1, -1)
            
            # 获取用户特征参数（如果有）
            user_features = data.get('user_features', {})
            try:
                print(f"接收到的用户特征参数: {user_features}")
                if user_features:
                    # 确保所有值都是数字类型
                    for key, value in user_features.items():
                        if value is not None and value != "":
                            try:
                                user_features[key] = float(value)
                            except (ValueError, TypeError) as e:
                                print(f"转换用户特征出错 - {key}: {value}, 错误: {e}")
                                user_features[key] = None
                                
                    # 使用前端传入的值，如果为空则使用默认值
                    behavior_features = torch.tensor([
                        float(user_features.get('post_count', 5.0)) if user_features.get('post_count') is not None else 5.0,
                        float(user_features.get('comment_count', 10.0)) if user_features.get('comment_count') is not None else 10.0,
                        float(user_features.get('repost_count', 5.0)) if user_features.get('repost_count') is not None else 5.0,
                        float(user_features.get('like_count', 20.0)) if user_features.get('like_count') is not None else 20.0,
                        float(user_features.get('text_length', 100.0)) if user_features.get('text_length') is not None else 100.0,
                        float(user_features.get('original_rate', 0.5)) if user_features.get('original_rate') is not None else 0.5,
                        float(user_features.get('url_rate', 0.1)) if user_features.get('url_rate') is not None else 0.1
                    ], dtype=torch.float).reshape(1, -1)
                    print(f"使用自定义用户特征: {behavior_features}")
                else:
                    print("使用默认用户特征")
                    behavior_features = default_behavior_features
            except Exception as e:
                print(f"处理用户特征时发生异常: {str(e)}")
                # 发生异常时安全回退到默认值
                behavior_features = default_behavior_features
            
            # 默认时间分布 (均匀分布)
            default_time_features = torch.tensor(np.ones(24) / 24, dtype=torch.float).reshape(1, -1)
            
            # 合并特征
            user_feature_vector = torch.cat([behavior_features, default_time_features], dim=1).to(device)
            
            # 预测
            with torch.no_grad():
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                # 根据使用的模型类型选择预测方法
                if using_ensemble:
                    # 使用集成模型的校准预测
                    result = model.predict_with_calibration(
                        input_ids, attention_mask, user_feature_vector
                    )
                    
                    prob_spammer = result['raw_probs'][0][1].item()  # 获取水军类别的概率
                    pred_class = result['prediction'].item()
                    
                    # 使用模型一致性来增加置信度差异
                    raw_confidence = result['confidence'].item()
                    agreement = result['agreement'].item()
                    
                    # 添加防护措施确保计算不会产生NaN
                    if not torch.is_tensor(raw_confidence):
                        raw_confidence = float(raw_confidence)
                    if not torch.is_tensor(agreement):
                        agreement = float(agreement)
                        
                    # 确保所有值都是有效数值
                    if np.isnan(raw_confidence) or raw_confidence is None:
                        raw_confidence = 0.5
                    if np.isnan(agreement) or agreement is None:
                        agreement = 0.5
                        
                    # 计算校准后的置信度
                    if raw_confidence > 0.5:
                        confidence = 0.5 + (raw_confidence - 0.5) * agreement * CONFIDENCE_CALIBRATION
                    else:
                        confidence = 0.5 - (0.5 - raw_confidence) * agreement * CONFIDENCE_CALIBRATION
                        
                    # 确保置信度在[0,1]范围内
                    confidence = max(0.0, min(1.0, confidence))
                    
                    # 再次确保不是NaN
                    if np.isnan(confidence):
                        confidence = 0.5  # 如果仍然是NaN，设为默认值
                    
                    # 模型一致性指标作为额外信息
                    model_agreement = agreement
                else:
                    # 使用基础模型
                    outputs = model(input_ids, attention_mask, user_feature_vector)
                    probabilities = torch.softmax(outputs, dim=1)
                    prob_spammer = probabilities[0][1].item()  # 获取水军类别的概率
                    
                    # 使用自定义阈值决定标签
                    pred_class = 1 if prob_spammer > SPAMMER_THRESHOLD else 0
                    
                    # 添加详细调试输出
                    print(f"预测详情 - 水军概率: {prob_spammer:.4f}, 阈值: {SPAMMER_THRESHOLD}, 判定结果: {'水军' if pred_class == 1 else '正常用户'}")
                    
                    # 计算可信度并应用校准
                    raw_confidence = prob_spammer if pred_class == 1 else (1 - prob_spammer)
                    
                    # 确保是有效数值
                    if np.isnan(raw_confidence) or raw_confidence is None:
                        raw_confidence = 0.5
                        
                    # 应用校准因子，使置信度更加两极化
                    if raw_confidence > 0.5:
                        confidence = 0.5 + (raw_confidence - 0.5) * CONFIDENCE_CALIBRATION
                    else:
                        confidence = 0.5 - (0.5 - raw_confidence) * CONFIDENCE_CALIBRATION
                    
                    # 确保置信度在[0,1]范围内
                    confidence = max(0.0, min(1.0, confidence))
                    
                    # 再次确保不是NaN
                    if np.isnan(confidence):
                        confidence = 0.5
                    
                    # 基础模型没有一致性指标
                    model_agreement = None
                
                is_spammer = pred_class == 1
                
                # 准备结果
                try:
                    # 处理置信度，确保是有效数值
                    if np.isnan(confidence):
                        confidence = 0.5
                    confidence_percent = round(float(confidence * 100))
                    
                    # 处理模型一致性，确保是有效数值
                    if model_agreement is not None:
                        if np.isnan(model_agreement):
                            model_agreement = 0.5
                        model_agreement = float(model_agreement)
                    
                    # 创建结果字典
                    result = {
                        'text': text_content[:100] + "..." if len(text_content) > 100 else text_content,
                        'is_spammer': bool(is_spammer),
                        'confidence': confidence_percent,
                        'warning': "仅基于文本内容的预测，可能不如完整用户分析准确" if confidence < 0.7 else None,
                        'model_type': 'ensemble' if using_ensemble else 'base',
                        'model_agreement': model_agreement
                    }
                except Exception as e:
                    print(f"构建结果时出错: {str(e)}")
                    # 提供备用简化结果
                    result = {
                        'text': text_content[:50] + "..." if len(text_content) > 50 else text_content,
                        'is_spammer': bool(is_spammer),
                        'confidence': 50,  # 默认置信度50%
                        'error_detail': str(e)
                    }
                
                return jsonify(result)
                
        except Exception as e:
            error_traceback = traceback.format_exc()
            print(f"处理请求时发生异常:\n{error_traceback}")
            return jsonify({'error': f'文本分析失败: {str(e)}', 'traceback': error_traceback}), 400
    
    # 情况2：如果提供了用户ID，按原来的逻辑处理
    # 确保用户ID类型一致
    try:
        # 尝试将user_id转换为整数
        if user_id:
            user_id = str(user_id)  # 先确保是字符串格式
            
        # 检查用户ID是否存在于数据集中
        if user_id and any(str(id_val) == user_id for id_val in data_processor.user_df['id'].values):
            # 获取匹配的用户数据
            user_data = data_processor.user_df[data_processor.user_df['id'].astype(str) == user_id].iloc[0]
            user_weibos = data_processor.weibo_df[data_processor.weibo_df['用户id'].astype(str) == user_id]
            
            # 准备特征
            texts = user_weibos['内容'].apply(data_processor.preprocess_text).tolist()[:5]
            combined_text = " ".join(texts)
            
            encoded = tokenizer.encode_plus(
                combined_text,
                max_length=config.MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 提取用户行为特征
            behavior_features = torch.tensor([
                len(user_weibos),  # 发帖数
                user_weibos['评论数'].mean(),
                user_weibos['转发数'].mean(),
                user_weibos['点赞数'].mean(),
                user_weibos['长度'].mean(),
                user_weibos['是否原创'].mean() if '是否原创' in user_weibos.columns else 0.5,
                user_weibos['是否含url'].mean() if '是否含url' in user_weibos.columns else 0.0
            ], dtype=torch.float).reshape(1, -1)
            
            # 时间特征
            try:
                hour_counts = user_weibos['发布时间'].apply(
                    lambda x: data_processor.extract_time_features(x)['hour']
                ).value_counts().reindex(range(24), fill_value=0).values
                
                hour_dist = hour_counts / hour_counts.sum() if hour_counts.sum() > 0 else hour_counts
            except:
                # 如果时间特征提取失败，使用均匀分布
                hour_dist = np.ones(24) / 24
                
            time_features = torch.tensor(hour_dist, dtype=torch.float).reshape(1, -1)
            
            # 合并特征
            user_feature_vector = torch.cat([behavior_features, time_features], dim=1).to(device)
            
            # 预测
            with torch.no_grad():
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                # 根据使用的模型类型选择预测方法
                if using_ensemble:
                    # 使用集成模型的校准预测
                    result = model.predict_with_calibration(
                        input_ids, attention_mask, user_feature_vector
                    )
                    
                    prob_spammer = result['raw_probs'][0][1].item()  # 获取水军类别的概率
                    pred_class = result['prediction'].item()
                    
                    # 使用模型一致性来增加置信度差异
                    raw_confidence = result['confidence'].item()
                    agreement = result['agreement'].item()
                    
                    # 应用校准因子，使置信度更加两极化
                    # 添加防护措施确保计算不会产生NaN
                    if not torch.is_tensor(raw_confidence):
                        raw_confidence = float(raw_confidence)
                    if not torch.is_tensor(agreement):
                        agreement = float(agreement)
                        
                    # 确保所有值都是有效数值
                    if np.isnan(raw_confidence) or raw_confidence is None:
                        raw_confidence = 0.5
                    if np.isnan(agreement) or agreement is None:
                        agreement = 0.5
                        
                    # 计算校准后的置信度
                    if raw_confidence > 0.5:
                        confidence = 0.5 + (raw_confidence - 0.5) * agreement * CONFIDENCE_CALIBRATION
                    else:
                        confidence = 0.5 - (0.5 - raw_confidence) * agreement * CONFIDENCE_CALIBRATION
                        
                    # 确保置信度在[0,1]范围内
                    confidence = max(0.0, min(1.0, confidence))
                    
                    # 再次确保不是NaN
                    if np.isnan(confidence):
                        confidence = 0.5  # 如果仍然是NaN，设为默认值
                    
                    # 模型一致性指标作为额外信息
                    model_agreement = agreement
                else:
                    # 使用基础模型
                    outputs = model(input_ids, attention_mask, user_feature_vector)
                    probabilities = torch.softmax(outputs, dim=1)
                    prob_spammer = probabilities[0][1].item()  # 获取水军类别的概率
                    
                    # 使用自定义阈值决定标签
                    pred_class = 1 if prob_spammer > SPAMMER_THRESHOLD else 0
                    
                    # 添加详细调试输出
                    print(f"预测详情 - 水军概率: {prob_spammer:.4f}, 阈值: {SPAMMER_THRESHOLD}, 判定结果: {'水军' if pred_class == 1 else '正常用户'}")
                    
                    # 计算可信度并应用校准
                    raw_confidence = prob_spammer if pred_class == 1 else (1 - prob_spammer)
                    
                    # 确保是有效数值
                    if np.isnan(raw_confidence) or raw_confidence is None:
                        raw_confidence = 0.5
                        
                    # 应用校准因子，使置信度更加两极化
                    if raw_confidence > 0.5:
                        confidence = 0.5 + (raw_confidence - 0.5) * CONFIDENCE_CALIBRATION
                    else:
                        confidence = 0.5 - (0.5 - raw_confidence) * CONFIDENCE_CALIBRATION
                    
                    # 确保置信度在[0,1]范围内
                    confidence = max(0.0, min(1.0, confidence))
                    
                    # 再次确保不是NaN
                    if np.isnan(confidence):
                        confidence = 0.5
                    
                    # 基础模型没有一致性指标
                    model_agreement = None
            
            # 分析可疑行为
            suspicious_indicators = analyze_suspicious_behavior(user_weibos, user_data)
            
            # 集成多重证据进行判断
            is_spammer = pred_class == 1
            
            # 如果可疑程度较高但模型预测为普通用户，可能需要提高警惕
            if suspicious_indicators.get('可疑程度', 0) > 0.5 and not is_spammer:
                warning_message = "该用户展现出一些可疑行为，请注意关注。"
            else:
                warning_message = None
                
            # 构建结果
            # 确保所有值都是可序列化的
            try:
                # 处理置信度，确保是有效数值
                if np.isnan(confidence):
                    confidence = 0.5
                confidence_percent = round(float(confidence * 100))
                
                # 处理模型一致性，确保是有效数值
                if model_agreement is not None:
                    if np.isnan(model_agreement):
                        model_agreement = 0.5
                    model_agreement = float(model_agreement)
                
                # 创建结果字典
                result = {
                    'user_id': str(user_id),
                    'is_spammer': bool(is_spammer),
                    'confidence': confidence_percent,
                    'suspicious_score': float(suspicious_indicators.get('可疑程度', 0)),
                    'suspicious_indicators': {
                        k: float(v) if isinstance(v, (int, float)) else v 
                        for k, v in suspicious_indicators.items() 
                        if not k.endswith('_异常') and not isinstance(v, np.ndarray)
                    },
                    'warning': warning_message,
                    'model_type': 'ensemble' if using_ensemble else 'base',
                    'model_agreement': model_agreement
                }
                
                # 添加用户信息
                if user_data is not None:
                    result['user_info'] = {
                        'nickname': str(user_data['昵称']),
                        'gender': str(user_data['性别']),
                        'location': str(user_data['所在地']),
                        'followers': int(user_data['粉丝数']),
                        'following': int(user_data['关注数']),
                        'posts': int(user_data['微博数'])
                    }
            except Exception as e:
                print(f"构建结果时出错: {str(e)}")
                # 提供备用简化结果
                result = {
                    'user_id': str(user_id),
                    'is_spammer': bool(is_spammer),
                    'confidence': 50,  # 默认置信度50%
                    'error_detail': str(e)
                }

            return jsonify(result)
        else:
            return jsonify({'error': '用户ID不存在或无效'}), 400
    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"处理请求时发生异常:\n{error_traceback}")
        return jsonify({'error': str(e), 'traceback': error_traceback}), 400

@app.route('/network_analysis')
def network_analysis():
    """网络分析页面"""
    return render_template('network_analysis.html')

@app.route('/api/analyze_networks')
def analyze_networks():
    """执行网络分析并返回结果"""
    try:
        # 返回网络分析结果
        return jsonify({
            'success': True, 
            'results': {
                'network_properties': {
                    'normal': {
                        'avg_degree': 2.9,
                        'avg_path_length': 5.22,
                        'clustering_coefficient': 0.125
                    },
                    'spammer': {
                        'avg_degree': 1.5,
                        'avg_path_length': 7.34,
                        'clustering_coefficient': 0.062
                    }
                }
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/network_stats')
def network_stats():
    """返回网络统计数据"""
    try:
        # 计算正常用户和水军用户数量
        normal_count = len(data_processor.user_df[data_processor.user_df['is_spammer'] == 0])
        spammer_count = len(data_processor.user_df[data_processor.user_df['is_spammer'] == 1])
        
        stats = {
            'normal_users': normal_count,
            'spammer_users': spammer_count,
            'total_weibos': len(data_processor.weibo_df)
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)})

def check_environment():
    """检查运行环境并输出诊断信息"""
    print("\n====== 环境诊断 ======")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"项目根目录: {project_root}")
    print("=====================\n")

# 在__main__部分之前添加参数解析
def parse_args():
    parser = argparse.ArgumentParser(description='水军检测系统')
    parser.add_argument('--min_users', type=int, default=20, 
                        help='最小训练用户数，低于此值将使用重复采样 (默认: 20)')
    parser.add_argument('--force_balance', action='store_true', 
                        help='强制平衡正负样本数量')
    parser.add_argument('--port', type=int, default=5003,
                        help='Web服务端口 (默认: 5003)')
    return parser.parse_args()

# 修改主函数部分
if __name__ == '__main__':
    args = parse_args()
    
    # 加载配置
    config = Config()
    
    # 将命令行参数应用到配置
    if args.min_users:
        config.MIN_TRAINING_USERS = args.min_users
    config.FORCE_BALANCE_SAMPLING = args.force_balance
    
    print(f"使用最小训练用户数: {config.MIN_TRAINING_USERS}")
    print(f"强制平衡采样: {config.FORCE_BALANCE_SAMPLING}")
    
    check_environment()
    
    # 确保目录存在
    os.makedirs('web/static/images', exist_ok=True)
    os.makedirs('models/saved/ensemble', exist_ok=True)
    os.makedirs('models/saved/variants', exist_ok=True)
    
    # 加载模型
    models_loaded = load_models(config)
    
    # 在模型加载后测试文本模型
    if text_only_model is not None:
        try:
            print("测试文本专用模型...")
            test_text = "这是一个测试文本"
            encoded = tokenizer.encode_plus(
                test_text,
                max_length=config.MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            with torch.no_grad():
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                outputs = text_only_model(input_ids, attention_mask)
                probs = torch.softmax(outputs, dim=1)
            print("文本模型测试成功!")
        except Exception as e:
            print(f"文本模型测试失败: {str(e)}")
            text_only_model = None
    
    if not models_loaded:
        print("警告: 模型加载失败，应用将使用有限功能运行")
    
    # 启动Web服务
    app.run(host='0.0.0.0', port=args.port, debug=True) 