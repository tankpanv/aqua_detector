import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from web.network_analysis import NetworkAnalyzer

def generate_sample_data():
    """生成示例数据，以便在真实数据不可用时使用"""
    # 创建示例用户数据
    user_data = {
        'id': list(range(1, 101)),
        '昵称': [f'用户{i}' for i in range(1, 101)],
        '性别': np.random.choice(['男', '女'], 100),
        '所在地': np.random.choice(['北京', '上海', '广州', '深圳', '杭州'], 100),
        '粉丝数': np.random.randint(100, 10000, 100),
        '关注数': np.random.randint(50, 500, 100),
        '微博数': np.random.randint(10, 1000, 100),
        'is_spammer': np.random.choice([0, 1], 100, p=[0.7, 0.3])  # 70%正常用户，30%水军
    }
    user_df = pd.DataFrame(user_data)
    
    # 创建示例微博数据
    weibo_rows = []
    for user_id in user_df['id']:
        user_info = user_df[user_df['id'] == user_id].iloc[0]
        is_spammer = user_info['is_spammer']
        
        # 每个用户生成1-10条微博
        num_posts = np.random.randint(1, 11)
        for _ in range(num_posts):
            # 生成发布时间（水军用户夜间发布概率更高）
            if is_spammer:
                hour = np.random.choice(range(24), p=[0.02]*16 + [0.05]*8)  # 夜间概率更高
            else:
                hour = np.random.choice(range(24), p=[0.03]*8 + [0.07]*8 + [0.03]*8)  # 工作时间概率更高
                
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 29)
            
            # 生成微博内容（水军内容重复性更高）
            if is_spammer:
                content_templates = [
                    "这个产品真的超赞！必须推荐给大家！",
                    "用了都说好，真的很不错！",
                    "强烈推荐，效果很好！",
                    "好评，下次还会购买！"
                ]
                content = np.random.choice(content_templates)
            else:
                content = f"这是用户{user_id}的第{_+1}条微博，分享一下日常生活。今天天气不错！"
                
            weibo_rows.append({
                'id': f'weibo_{len(weibo_rows) + 1}',
                '用户id': user_id,
                '内容': content,
                '发布时间': f'2023/{month:02d}/{day:02d} {hour:02d}:00',
                '转发数': np.random.randint(0, 100),
                '评论数': np.random.randint(0, 50),
                '点赞数': np.random.randint(0, 200),
                '长度': len(content),
                '是否原创': np.random.choice([0, 1], p=[0.3, 0.7]),
                '是否含url': np.random.choice([0, 1], p=[0.8, 0.2]),
                '是否转发': np.random.choice([0, 1], p=[0.7, 0.3])
            })
    
    weibo_df = pd.DataFrame(weibo_rows)
    
    # 创建用户关系数据
    relation_rows = []
    for user_id in user_df['id']:
        # 每个用户关注3-10个其他用户
        num_follows = np.random.randint(3, 11)
        followed_ids = np.random.choice([i for i in user_df['id'] if i != user_id], 
                                        min(num_follows, len(user_df)-1), 
                                        replace=False)
        
        for followed_id in followed_ids:
            relation_rows.append({
                'source_id': user_id,
                'target_id': followed_id
            })
    
    relation_df = pd.DataFrame(relation_rows)
    
    return user_df, weibo_df, relation_df

def ensure_dirs():
    """确保必要的目录存在"""
    os.makedirs('web/static/images', exist_ok=True)

def generate_sample_images():
    """生成示例图像用于演示"""
    ensure_dirs()
    
    # 生成示例网络图
    def generate_network_image(filename, is_normal=True):
        G = nx.barabasi_albert_graph(50 if is_normal else 30, 3 if is_normal else 2)
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, 
                              node_size=100, 
                              node_color='blue' if is_normal else 'red',
                              alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        plt.title(f"{'正常' if is_normal else '水军'}用户社交网络")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'web/static/images/{filename}', dpi=100)
        plt.close()
    
    # 生成正常用户网络和水军用户网络
    generate_network_image('normal_network.png', True)
    generate_network_image('spam_network.png', False)
    
    # 生成度分布图
    plt.figure(figsize=(10, 6))
    normal_degrees = np.random.power(2, 100) * 10
    spam_degrees = np.random.power(3, 100) * 5
    plt.hist(normal_degrees, bins=20, alpha=0.5, label='正常用户')
    plt.hist(spam_degrees, bins=20, alpha=0.5, label='水军用户')
    plt.xlabel('度数')
    plt.ylabel('频率')
    plt.title('用户网络度分布')
    plt.legend()
    plt.tight_layout()
    plt.savefig('web/static/images/degree_distribution.png')
    plt.close()
    
    # 生成小时活跃度图
    plt.figure(figsize=(10, 6))
    hours = np.arange(24)
    normal_activity = np.concatenate([
        np.linspace(0.01, 0.02, 6),  # 0-6点
        np.linspace(0.03, 0.07, 6),  # 6-12点
        np.linspace(0.08, 0.06, 6),  # 12-18点
        np.linspace(0.05, 0.02, 6)   # 18-24点
    ])
    spam_activity = np.concatenate([
        np.linspace(0.02, 0.03, 6),  # 0-6点
        np.linspace(0.03, 0.04, 6),  # 6-12点
        np.linspace(0.04, 0.05, 6),  # 12-18点
        np.linspace(0.05, 0.08, 6)   # 18-24点
    ])
    plt.plot(hours, normal_activity, 'b-', label='正常用户')
    plt.plot(hours, spam_activity, 'r-', label='水军用户')
    plt.xticks(range(0, 24, 2))
    plt.xlabel('小时')
    plt.ylabel('活跃度')
    plt.title('24小时活跃度分布')
    plt.legend()
    plt.tight_layout()
    plt.savefig('web/static/images/hourly_activity.png')
    plt.close()
    
    # 生成月度活跃度图
    plt.figure(figsize=(10, 6))
    months = np.arange(1, 13)
    normal_monthly = np.ones(12) / 12 + np.random.normal(0, 0.01, 12)
    spam_monthly = np.zeros(12)
    spam_monthly[np.random.choice(12, 3)] = 0.3  # 水军在特定月份爆发
    spam_monthly = spam_monthly + np.random.normal(0, 0.01, 12)
    spam_monthly = spam_monthly / spam_monthly.sum()
    
    plt.plot(months, normal_monthly, 'b-', label='正常用户')
    plt.plot(months, spam_monthly, 'r-', label='水军用户')
    plt.xticks(range(1, 13))
    plt.xlabel('月份')
    plt.ylabel('活跃度')
    plt.title('月度活跃分布')
    plt.legend()
    plt.tight_layout()
    plt.savefig('web/static/images/monthly_activity.png')
    plt.close()
    
    # 生成转发关系图
    plt.figure(figsize=(10, 8))
    G = nx.DiGraph()
    for i in range(30):
        G.add_node(i, type='normal' if i < 20 else 'spammer')
    
    # 添加边
    for i in range(20):  # 正常用户更多样化的转发
        targets = np.random.choice(30, np.random.randint(1, 4), replace=False)
        for target in targets:
            if target != i:
                G.add_edge(i, target)
    
    for i in range(20, 30):  # 水军用户更集中的转发
        targets = np.random.choice(range(20, 30), np.random.randint(1, 3), replace=False)
        for target in targets:
            if target != i:
                G.add_edge(i, target)
    
    pos = nx.spring_layout(G, seed=42)
    node_colors = ['blue' if G.nodes[n]['type'] == 'normal' else 'red' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, alpha=0.7)
    nx.draw_networkx_edges(G, pos, alpha=0.4, arrows=True)
    plt.title('微博转发关系网络')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('web/static/images/repost_patterns.png')
    plt.close()
    
    # 生成地域关系热图
    plt.figure(figsize=(12, 8))
    regions = ['北京', '上海', '广州', '深圳', '杭州', '南京', '武汉', '成都']
    normal_matrix = np.zeros((len(regions), len(regions)))
    spam_matrix = np.zeros((len(regions), len(regions)))
    
    # 正常用户地域关系：对角线和临近城市概率更高
    for i in range(len(regions)):
        normal_matrix[i, i] = np.random.uniform(0.3, 0.5)
        for j in range(len(regions)):
            if i != j:
                if abs(i-j) <= 2:  # 邻近城市
                    normal_matrix[i, j] = np.random.uniform(0.1, 0.2)
                else:
                    normal_matrix[i, j] = np.random.uniform(0.01, 0.05)
    
    # 水军用户地域关系：随机分布
    for i in range(len(regions)):
        for j in range(len(regions)):
            spam_matrix[i, j] = np.random.uniform(0.05, 0.2)
    
    # 绘制热图
    plt.subplot(1, 2, 1)
    sns.heatmap(normal_matrix, annot=True, fmt='.2f', xticklabels=regions, yticklabels=regions)
    plt.title('正常用户地域关系')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(spam_matrix, annot=True, fmt='.2f', xticklabels=regions, yticklabels=regions)
    plt.title('水军用户地域关系')
    
    plt.tight_layout()
    plt.savefig('web/static/images/region_connections.png')
    plt.close()
    
    # 生成内容同质性分析图
    plt.figure(figsize=(8, 6))
    similarity_data = {
        '用户类型': ['正常用户', '水军用户'],
        '内容同质性': [0.35, 0.82]
    }
    sns.barplot(x='用户类型', y='内容同质性', data=pd.DataFrame(similarity_data))
    plt.title('内容同质性对比')
    plt.ylabel('平均内容相似度')
    plt.ylim(0, 1)
    
    # 添加数值标签
    for i, v in enumerate(similarity_data['内容同质性']):
        plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('web/static/images/content_homophily.png')
    plt.close()

def generate_real_analysis_images():
    """尝试使用真实数据生成分析图像"""
    try:
        # 加载配置和数据
        config = Config()
        
        # 加载用户和微博数据
        try:
            user_df = pd.read_csv(config.USER_DETAIL_PATH)
            weibo_df = pd.read_csv(config.WEIBO_PATH)
            
            # 尝试加载关系数据
            relation_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                         'data/relation.csv')
            if os.path.exists(relation_path):
                relation_df = pd.read_csv(relation_path)
            else:
                relation_df = None
                print(f"关系数据文件不存在: {relation_path}, 将使用示例关系数据")
                
            # 确保用户数据包含is_spammer列
            if 'is_spammer' not in user_df.columns and '类别' in user_df.columns:
                user_df['is_spammer'] = (user_df['类别'] > 0).astype(int)
            
            print(f"已加载真实数据: {len(user_df)}个用户, {len(weibo_df)}条微博")
            
            # 创建网络分析器并运行分析
            analyzer = NetworkAnalyzer(weibo_df, user_df, relation_df)
            analyzer.run_all_analyses()
            print("使用真实数据生成分析图像完成")
            return True
        except Exception as e:
            print(f"使用真实数据生成分析图像时出错: {str(e)}")
            return False
    except Exception as e:
        print(f"加载配置或数据时出错: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始生成网络分析图像...")
    
    # 确保目录存在
    ensure_dirs()
    
    # 尝试使用真实数据生成图像
    success = generate_real_analysis_images()
    
    # 如果真实数据分析失败，使用示例数据生成图像
    if not success:
        print("使用示例数据生成分析图像...")
        generate_sample_images()
        print("示例分析图像生成完成")
    
    print("所有网络分析图像已生成完成") 