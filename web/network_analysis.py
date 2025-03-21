import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import jieba
import re
from collections import Counter
import os
from datetime import datetime
import folium
from folium.plugins import HeatMap
import community as community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合Web服务器

class NetworkAnalyzer:
    def __init__(self, weibo_df, user_df, relation_df=None):
        """
        初始化网络分析器
        
        Args:
            weibo_df: 微博数据DataFrame
            user_df: 用户数据DataFrame
            relation_df: 关注关系数据DataFrame（可选）
        """
        self.weibo_df = weibo_df
        self.user_df = user_df
        self.relation_df = relation_df
        self.normal_users = self.user_df[self.user_df['is_spammer'] == 0]['id'].tolist()
        self.spammer_users = self.user_df[self.user_df['is_spammer'] == 1]['id'].tolist()
        
        # 创建输出目录
        os.makedirs('web/static/images', exist_ok=True)
        
    def build_network(self, users_type='all'):
        """
        构建用户社交网络
        
        Args:
            users_type: 'normal', 'spammer', 或 'all'
            
        Returns:
            networkx图对象
        """
        if self.relation_df is None:
            raise ValueError("关系数据未提供，无法构建网络")
            
        G = nx.DiGraph()
        
        # 根据类型筛选用户
        if users_type == 'normal':
            users = self.user_df[self.user_df['is_spammer'] == 0]['id'].tolist()
        elif users_type == 'spammer':
            users = self.user_df[self.user_df['is_spammer'] == 1]['id'].tolist()
        else:
            users = self.user_df['id'].tolist()
            
        # 筛选关系
        filtered_relations = self.relation_df[
            (self.relation_df['source_id'].isin(users)) & 
            (self.relation_df['target_id'].isin(users))
        ]
        
        # 添加节点
        for user_id in users:
            user_info = self.user_df[self.user_df['id'] == user_id].iloc[0]
            is_spammer = user_info['is_spammer']
            G.add_node(user_id, is_spammer=is_spammer)
            
        # 添加边
        for _, row in filtered_relations.iterrows():
            G.add_edge(row['source_id'], row['target_id'])
            
        return G
        
    def visualize_network(self, G, title, output_path):
        """
        可视化网络
        
        Args:
            G: networkx图对象
            title: 图表标题
            output_path: 输出文件路径
        """
        plt.figure(figsize=(10, 8))
        
        # 设置节点颜色（正常用户蓝色，虚假用户红色）
        node_colors = ['red' if G.nodes[node]['is_spammer'] else 'blue' for node in G.nodes()]
        
        # 使用spring布局
        pos = nx.spring_layout(G, seed=42)
        
        # 绘制节点和边
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True)
        
        # 添加标题
        plt.title(title)
        plt.axis('off')
        
        # 保存图像
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        return os.path.basename(output_path)
        
    def analyze_network_properties(self, G, network_type):
        """
        分析网络属性
        
        Args:
            G: networkx图对象
            network_type: 网络类型 ('normal' 或 'spammer')
            
        Returns:
            网络属性字典
        """
        # 基本度量
        avg_degree = sum(dict(G.degree()).values()) / len(G)
        if nx.is_directed(G):
            avg_in_degree = sum(dict(G.in_degree()).values()) / len(G)
            avg_out_degree = sum(dict(G.out_degree()).values()) / len(G)
        else:
            avg_in_degree = avg_out_degree = None
            
        # 连通分量
        scc = list(nx.strongly_connected_components(G)) if nx.is_directed(G) else list(nx.connected_components(G))
        largest_scc = max(scc, key=len)
        
        # 计算平均路径长度（在最大连通分量上）
        subgraph = G.subgraph(largest_scc)
        try:
            avg_path_length = nx.average_shortest_path_length(subgraph)
        except:
            avg_path_length = float('inf')
            
        # 计算聚类系数
        clustering_coef = nx.average_clustering(G.to_undirected())
        
        # 度分布
        degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
        degree_count = Counter(degree_sequence)
        deg, cnt = zip(*degree_count.items())
        
        # 可视化度分布
        plt.figure(figsize=(8, 6))
        plt.bar(deg, cnt, width=0.80, color='b')
        plt.title(f"{network_type} Network Degree Distribution")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        plt.yscale('log')
        plt.xscale('log')
        degree_plot_path = f'web/static/images/{network_type}_degree_dist.png'
        plt.savefig(degree_plot_path)
        plt.close()
        
        return {
            'avg_degree': avg_degree,
            'avg_in_degree': avg_in_degree,
            'avg_out_degree': avg_out_degree,
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_connected_components': len(scc),
            'largest_component_size': len(largest_scc),
            'avg_path_length': avg_path_length,
            'clustering_coefficient': clustering_coef,
            'degree_distribution_plot': os.path.basename(degree_plot_path)
        }
        
    def analyze_posting_patterns(self):
        """
        分析用户发帖模式
        
        Returns:
            包含图表路径的字典
        """
        # 添加发帖时间信息
        self.weibo_df['datetime'] = pd.to_datetime(self.weibo_df['发布时间'], format='%Y/%m/%d %H:%M')
        self.weibo_df['hour'] = self.weibo_df['datetime'].dt.hour
        self.weibo_df['month'] = self.weibo_df['datetime'].dt.month
        
        # 合并用户信息
        merged_df = self.weibo_df.merge(self.user_df[['id', 'is_spammer']], 
                                         left_on='用户id', right_on='id', how='left')
        
        # 按小时分析发帖频率
        plt.figure(figsize=(10, 6))
        hour_data = merged_df.groupby(['is_spammer', 'hour']).size().unstack(level=0)
        
        # 归一化
        hour_data = hour_data.div(hour_data.sum())
        
        sns.lineplot(data=hour_data)
        plt.title('Hourly Posting Patterns')
        plt.xlabel('Hour of Day')
        plt.ylabel('Normalized Frequency')
        plt.xticks(range(0, 24))
        plt.legend(['Normal Users', 'Spammers'])
        hourly_plot_path = 'web/static/images/hourly_posting_pattern.png'
        plt.savefig(hourly_plot_path)
        plt.close()
        
        # 按月分析发帖频率
        plt.figure(figsize=(10, 6))
        month_data = merged_df.groupby(['is_spammer', 'month']).size().unstack(level=0)
        
        # 归一化
        month_data = month_data.div(month_data.sum())
        
        sns.lineplot(data=month_data)
        plt.title('Monthly Posting Patterns')
        plt.xlabel('Month')
        plt.ylabel('Normalized Frequency')
        plt.xticks(range(1, 13))
        plt.legend(['Normal Users', 'Spammers'])
        monthly_plot_path = 'web/static/images/monthly_posting_pattern.png'
        plt.savefig(monthly_plot_path)
        plt.close()
        
        return {
            'hourly_plot': os.path.basename(hourly_plot_path),
            'monthly_plot': os.path.basename(monthly_plot_path)
        }
        
    def analyze_geographic_distribution(self):
        """
        分析用户地理分布
        
        Returns:
            地理分布图路径
        """
        # 提取地理位置信息（示例代码，实际实现需要根据具体数据调整）
        location_map = {
            '北京': [39.9042, 116.4074],
            '上海': [31.2304, 121.4737],
            '广州': [23.1291, 113.2644],
            '深圳': [22.5431, 114.0579],
            '杭州': [30.2741, 120.1551],
            '南京': [32.0603, 118.7969],
            '武汉': [30.5928, 114.3055],
            '成都': [30.5723, 104.0665],
            '重庆': [29.4316, 106.9123],
            '西安': [34.3416, 108.9398],
            '苏州': [31.2990, 120.5853],
            '天津': [39.3434, 117.3616],
            '郑州': [34.7466, 113.6253],
            '长沙': [28.2282, 112.9388],
            '青岛': [36.0671, 120.3826]
        }
        
        # 创建地图
        m_normal = folium.Map(location=[35.0, 105.0], zoom_start=4)
        m_spammer = folium.Map(location=[35.0, 105.0], zoom_start=4)
        
        # 分析正常用户
        normal_users = self.user_df[self.user_df['is_spammer'] == 0]
        
        normal_locations = []
        for _, user in normal_users.iterrows():
            location = user.get('所在地', '')
            if location in location_map:
                normal_locations.append(location_map[location])
                
        # 分析虚假用户
        spammer_users = self.user_df[self.user_df['is_spammer'] == 1]
        
        spammer_locations = []
        for _, user in spammer_users.iterrows():
            location = user.get('所在地', '')
            if location in location_map:
                spammer_locations.append(location_map[location])
        
        # 添加热力图
        HeatMap(normal_locations).add_to(m_normal)
        HeatMap(spammer_locations).add_to(m_spammer)
        
        # 保存地图
        normal_map_path = 'web/static/images/normal_users_map.html'
        spammer_map_path = 'web/static/images/spammer_users_map.html'
        
        m_normal.save(normal_map_path)
        m_spammer.save(spammer_map_path)
        
        return {
            'normal_map': os.path.basename(normal_map_path),
            'spammer_map': os.path.basename(spammer_map_path)
        }
        
    def analyze_text_similarity(self):
        """
        分析用户微博文本的同质性
        
        Returns:
            文本相似度分析结果
        """
        # 处理文本
        def clean_text(text):
            if pd.isna(text):
                return ""
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'@\S+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            return text.strip()
            
        self.weibo_df['clean_content'] = self.weibo_df['内容'].apply(clean_text)
        
        # 按用户聚合文本
        user_texts = self.weibo_df.groupby('用户id')['clean_content'].apply(' '.join).reset_index()
        
        # 合并用户信息
        user_texts = user_texts.merge(self.user_df[['id', 'is_spammer']], left_on='用户id', right_on='id', how='left')
        
        # 分别处理正常用户和虚假用户
        normal_texts = user_texts[user_texts['is_spammer'] == 0]['clean_content'].tolist()
        spammer_texts = user_texts[user_texts['is_spammer'] == 1]['clean_content'].tolist()
        
        # 简单的分词
        normal_words = [' '.join(jieba.cut(text)) for text in normal_texts]
        spammer_words = [' '.join(jieba.cut(text)) for text in spammer_texts]
        
        # 使用TF-IDF计算文本相似度
        normal_vectorizer = TfidfVectorizer(max_features=1000)
        normal_tfidf = normal_vectorizer.fit_transform(normal_words)
        normal_similarity = cosine_similarity(normal_tfidf)
        
        spammer_vectorizer = TfidfVectorizer(max_features=1000)
        spammer_tfidf = spammer_vectorizer.fit_transform(spammer_words)
        spammer_similarity = cosine_similarity(spammer_tfidf)
        
        # 计算平均相似度
        normal_avg_similarity = np.mean(normal_similarity)
        spammer_avg_similarity = np.mean(spammer_similarity)
        
        # 可视化相似度分布
        plt.figure(figsize=(10, 6))
        plt.hist(normal_similarity.flatten(), alpha=0.5, bins=50, label='Normal Users')
        plt.hist(spammer_similarity.flatten(), alpha=0.5, bins=50, label='Spammers')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Text Similarity Distribution')
        plt.legend()
        similarity_plot_path = 'web/static/images/text_similarity.png'
        plt.savefig(similarity_plot_path)
        plt.close()
        
        return {
            'normal_avg_similarity': normal_avg_similarity,
            'spammer_avg_similarity': spammer_avg_similarity,
            'similarity_plot': os.path.basename(similarity_plot_path)
        }
        
    def analyze_repost_network(self):
        """
        分析微博转发网络
        
        Returns:
            转发网络图路径
        """
        # 简化示例，实际需根据数据结构调整
        if '转发自' not in self.weibo_df.columns:
            print("缺少转发信息，无法构建转发网络")
            return None
            
        # 创建转发网络
        repost_network = nx.DiGraph()
        
        for _, post in self.weibo_df.iterrows():
            source_id = post['用户id']
            target_id = post.get('转发自')
            
            if not pd.isna(target_id):
                repost_network.add_edge(source_id, target_id)
                
        # 可视化转发网络
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(repost_network, seed=42)
        nx.draw(repost_network, pos, node_size=20, alpha=0.7)
        plt.title('Weibo Repost Network')
        repost_network_path = 'web/static/images/repost_network.png'
        plt.savefig(repost_network_path)
        plt.close()
        
        return os.path.basename(repost_network_path)
        
    def run_all_analyses(self):
        """
        运行所有分析并返回结果
        
        Returns:
            包含所有分析结果的字典
        """
        results = {}
        
        # 如果有关系数据，构建并分析网络
        if self.relation_df is not None:
            # 构建正常用户网络
            normal_network = self.build_network(users_type='normal')
            spammer_network = self.build_network(users_type='spammer')
            
            # 可视化网络
            results['normal_network_plot'] = self.visualize_network(
                normal_network, 'Normal Users Network', 'web/static/images/normal_network.png')
            results['spammer_network_plot'] = self.visualize_network(
                spammer_network, 'Spammer Network', 'web/static/images/spammer_network.png')
                
            # 分析网络属性
            results['normal_network_props'] = self.analyze_network_properties(normal_network, 'normal')
            results['spammer_network_props'] = self.analyze_network_properties(spammer_network, 'spammer')
            
            # 分析转发网络
            results['repost_network_plot'] = self.analyze_repost_network()
        
        # 分析发帖模式
        results['posting_patterns'] = self.analyze_posting_patterns()
        
        # 分析地理分布
        results['geographic_distribution'] = self.analyze_geographic_distribution()
        
        # 分析文本相似度
        results['text_similarity'] = self.analyze_text_similarity()
        
        return results 