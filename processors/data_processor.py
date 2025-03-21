import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import jieba
import re
import datetime
from tqdm import tqdm

from config import Config

class WeiboDataProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
        
    def load_data(self):
        """加载微博数据和用户详情数据"""
        self.weibo_df = pd.read_csv(self.config.WEIBO_PATH)
        self.user_df = pd.read_csv(self.config.USER_DETAIL_PATH)
        print(f"加载了 {len(self.weibo_df)} 条微博数据")
        print(f"加载了 {len(self.user_df)} 条用户数据")
        
        # 列名修正：把最后一列'类别'作为水军标识，大于0的为水军
        self.user_df['is_spammer'] = (self.user_df['类别'] > 0).astype(int)
        
        # 合并用户标签，使用suffixes避免列名冲突
        self.weibo_df = self.weibo_df.merge(
            self.user_df[['id', 'is_spammer']], 
            left_on='用户id',  # 使用 weibo_df 的 '用户id' 列
            right_on='id',    # 使用 user_df 的 'id' 列
            how='left',
            suffixes=('', '_user')  # 避免列名冲突
        )
        
        # 处理合并后可能存在的缺失值
        self.weibo_df['is_spammer'] = self.weibo_df['is_spammer'].fillna(0)
        
        # 调试输出
        print(self.weibo_df.head())
    def preprocess_text(self, text):
        """文本预处理"""
        if pd.isna(text):
            return ""
        # 去除URL
        text = re.sub(r'http\S+', '', text)
        # 去除@用户
        text = re.sub(r'@\S+', '', text)
        # 去除特殊符号
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def extract_time_features(self, time_str):
        """从时间字符串中提取时间特征"""
        try:
            dt = datetime.datetime.strptime(time_str, "%Y/%m/%d %H:%M")
            return {
                'hour': dt.hour,
                'day_of_week': dt.weekday(),
                'month': dt.month
            }
        except:
            return {'hour': -1, 'day_of_week': -1, 'month': -1}
    
    def prepare_features(self):
        """准备多视图特征"""
        print("准备特征...")
        # 1. 文本视图特征
        self.weibo_df['processed_text'] = self.weibo_df['内容'].apply(self.preprocess_text)
        
        # 2. 用户行为视图特征
        user_post_count = self.weibo_df.groupby('用户id').size().reset_index(name='post_count')
        self.user_df = self.user_df.merge(user_post_count, left_on='id', right_on='用户id', how='left')
        self.user_df['post_count'] = self.user_df['post_count'].fillna(0)
        
        # 统计用户行为特征
        user_stats = self.weibo_df.groupby('用户id').agg({
            '评论数': 'mean',
            '转发数': 'mean',
            '点赞数': 'mean',
            '长度': 'mean',
            '是否原创': 'mean',
            '是否含url': 'mean'
        }).reset_index()
        # 合并并填充NaN
        self.user_df = self.user_df.merge(user_stats, left_on='id', right_on='用户id', how='left')
        stats_columns = ['评论数', '转发数', '点赞数', '长度', '是否原创', '是否含url']
        for col in stats_columns:
            self.user_df[col] = self.user_df[col].fillna(0)
        
        # 时间特征处理
        time_features = self.weibo_df['发布时间'].apply(self.extract_time_features)
        time_df = pd.DataFrame(time_features.tolist())
        self.weibo_df = pd.concat([self.weibo_df, time_df], axis=1)
        
        # 重采样平衡数据
        spammer_users = self.user_df[self.user_df['is_spammer'] == 1]['id'].astype(int).tolist()
        normal_users = self.user_df[self.user_df['is_spammer'] == 0]['id'].astype(int).tolist()
        
        if len(normal_users) > len(spammer_users):
            sampled_normal_users = np.random.choice(normal_users, len(spammer_users), replace=False)
            final_users = np.concatenate([sampled_normal_users, spammer_users])
        else:
            sampled_spammer_users = np.random.choice(spammer_users, len(normal_users), replace=False)
            final_users = np.concatenate([normal_users, sampled_spammer_users])
        
        final_users = list(set(final_users))
        existing_ids = set(self.user_df['id'].astype(int).unique())
        valid_final_users = [uid for uid in final_users if uid in existing_ids]
        self.final_user_df = self.user_df[self.user_df['id'].isin(valid_final_users)]
        print(f"有效训练用户数量: {len(self.final_user_df)}")
        
    def split_data(self):
        """分割数据为训练集、验证集和测试集"""
        user_ids = self.final_user_df['id'].values
        labels = self.final_user_df['is_spammer'].values
        
        # 先分割出测试集
        train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
            user_ids, labels, test_size=self.config.TEST_RATIO, random_state=self.config.RANDOM_SEED, stratify=labels
        )
        
        # 再分割训练集和验证集
        val_ratio_adjusted = self.config.VAL_RATIO / (self.config.TRAIN_RATIO + self.config.VAL_RATIO)
        train_ids, val_ids, train_labels, val_labels = train_test_split(
            train_val_ids, train_val_labels, test_size=val_ratio_adjusted, 
            random_state=self.config.RANDOM_SEED, stratify=train_val_labels
        )
        
        return {
            'train': {'user_ids': train_ids, 'labels': train_labels},
            'val': {'user_ids': val_ids, 'labels': val_labels},
            'test': {'user_ids': test_ids, 'labels': test_labels}
        }
        
    def create_datasets(self):
        """创建训练、验证和测试数据集"""
        split_data = self.split_data()
        
        train_dataset = WeiboSpammerDataset(
            self.weibo_df, self.user_df, split_data['train']['user_ids'], 
            split_data['train']['labels'], self.tokenizer, self.config
        )
        
        val_dataset = WeiboSpammerDataset(
            self.weibo_df, self.user_df, split_data['val']['user_ids'], 
            split_data['val']['labels'], self.tokenizer, self.config
        )
        
        test_dataset = WeiboSpammerDataset(
            self.weibo_df, self.user_df, split_data['test']['user_ids'], 
            split_data['test']['labels'], self.tokenizer, self.config
        )
        
        return train_dataset, val_dataset, test_dataset
        
    def get_dataloaders(self):
        """获取数据加载器"""
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=False
        )
        
        return train_loader, val_loader, test_loader


class WeiboSpammerDataset(Dataset):
    def __init__(self, weibo_df, user_df, user_ids, labels, tokenizer, config):
        self.weibo_df = weibo_df
        self.user_df = user_df
        self.user_ids = user_ids
        self.labels = labels
        self.tokenizer = tokenizer
        self.config = config
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        label = self.labels[idx]
        
        user_weibos = self.weibo_df[self.weibo_df['用户id'] == user_id]
        texts = user_weibos['processed_text'].tolist()[:5]
        combined_text = " ".join(texts) if texts else ""
        
        encoded = self.tokenizer.encode_plus(
            combined_text,
            max_length=self.config.MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        user_features = self.user_df[self.user_df['id'] == user_id].iloc[0]
        behavior_features = torch.tensor([
            user_features.get('post_count', 0),
            user_features.get('评论数', 0),
            user_features.get('转发数', 0),
            user_features.get('点赞数', 0),
            user_features.get('长度', 0),
            user_features.get('是否原创', 0),
            user_features.get('是否含url', 0)
        ], dtype=torch.float)
        
        # 处理时间特征，避免除以零
        hour_features = []
        for h in range(24):
            count = sum(1 for t in user_weibos['发布时间'].apply(
                lambda x: self.extract_time_feature(x)
            ) if t == h)
            hour_features.append(count)
        total = sum(hour_features)
        if total == 0:
            hour_dist = [1.0 / 24] * 24  # 均匀分布
        else:
            hour_dist = [h / total for h in hour_features]
        time_features = torch.tensor(hour_dist, dtype=torch.float)
        
        user_features = torch.cat([behavior_features, time_features])
        
        # 检查特征是否存在NaN
        assert not torch.isnan(user_features).any(), "用户特征包含NaN！"
        
        return {
            'input_ids': encoded['input_ids'].flatten(),
            'attention_mask': encoded['attention_mask'].flatten(),
            'user_features': user_features,
            'label': torch.tensor(label, dtype=torch.long)
        }
        
    def extract_time_feature(self, time_str):
        """提取时间中的小时"""
        try:
            if pd.isna(time_str):
                return -1
            dt = datetime.datetime.strptime(time_str, "%Y/%m/%d %H:%M")
            return dt.hour
        except:
            return -1 