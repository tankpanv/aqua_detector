import os
import platform

class Config:
    # 根据当前平台设置路径分隔符
    SYSTEM_TYPE = platform.system()  # 'Windows', 'Linux', 或 'Darwin'
    
    # 基础路径设置
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_PATH, 'data')
    MODEL_PATH = os.path.join(BASE_PATH, 'models', 'saved')
    
    # 数据文件路径 - 使用os.path.join确保跨平台兼容
    USER_DETAIL_PATH = os.path.join(DATA_PATH, 'user_detail.csv')
    WEIBO_PATH = os.path.join(DATA_PATH, 'weibo.csv')
    RELATION_PATH = os.path.join(DATA_PATH, 'relation.csv')
    
    # 训练参数
    MIN_TRAINING_USERS = 20  # 每类最小训练用户数
    FORCE_BALANCE_SAMPLING = True  # 是否平衡正负样本
    
    # 模型路径
    TEXT_MODEL_PATH = os.path.join(MODEL_PATH, 'text_only_model.pt')
    BASE_MODEL_PATH = os.path.join(MODEL_PATH, 'best_model.pt')
    ENSEMBLE_MODEL_PATH = os.path.join(MODEL_PATH, 'ensemble', 'ensemble_best.pt')
    
    # 模型参数
    BERT_MODEL_NAME = 'bert-base-chinese'
    MAX_LEN = 128
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    EPOCHS = 5
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    DROPOUT_RATE = 0.1
    FREEZE_BERT = False  # 是否冻结BERT参数
    
    # 编码设置
    DEFAULT_ENCODING = 'utf-8'
    FALLBACK_ENCODING = 'gbk'  # Windows常用编码
    
    # 平台特定设置
    def __init__(self):
        if self.SYSTEM_TYPE == 'Windows':
            # Windows特定设置
            self.BATCH_SIZE = 8  # Windows通常内存较小，减小批量大小
            if not hasattr(self, 'MIN_TRAINING_USERS'):
                self.MIN_TRAINING_USERS = 20
        else:
            # Linux/Mac特定设置
            if not hasattr(self, 'MIN_TRAINING_USERS'):
                self.MIN_TRAINING_USERS = 50  # Linux通常有更多数据
    
    # 多视图融合参数
    TEXT_FEATURE_DIM = 768
    USER_FEATURE_DIM = 32
    RELATION_FEATURE_DIM = 16
    FUSION_OUTPUT_DIM = 256
    
    # Web服务参数
    HOST = "0.0.0.0"
    PORT = 5000
    
    # 训练参数
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    
    # 最小训练用户数参数
    MIN_TRAINING_USERS = 700  # 设置最小训练用户数
    # 如果实际用户数小于此值，则启用重复采样
    FORCE_BALANCE_SAMPLING = True  # 是否强制平衡正负样本 