import os

class Config:
    # 数据路径
    DATA_DIR = "WeiboSpammer/data"
    USER_DETAIL_PATH = os.path.join(DATA_DIR, "UserDetail.csv")
    WEIBO_PATH = os.path.join(DATA_DIR, "weibo.csv")
    
    # 模型参数
    BERT_MODEL_NAME = "bert-base-chinese"
    MAX_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    FREEZE_BERT = False  # 是否冻结BERT参数
    
    # 多视图融合参数
    TEXT_FEATURE_DIM = 768
    USER_FEATURE_DIM = 32
    RELATION_FEATURE_DIM = 16
    FUSION_OUTPUT_DIM = 256
    
    # 训练参数
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    
    # Web服务参数
    HOST = "0.0.0.0"
    PORT = 5000 