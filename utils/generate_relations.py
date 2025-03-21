import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def check_columns(df, required_columns, data_type='微博'):
    """检查数据框是否包含所需的列"""
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"警告: {data_type}数据缺少以下列: {missing_columns}")
        return False
    return True

def generate_relations(weibo_df, user_df, output_path='data/relation.csv'):
    """
    基于微博数据生成用户关系数据
    
    参数:
    - weibo_df: 微博数据DataFrame
    - user_df: 用户数据DataFrame
    - output_path: 输出文件路径
    """
    print("开始生成用户关系数据...")
    relations = []
    
    # 打印数据列信息
    print("\n微博数据列:", weibo_df.columns.tolist())
    print("用户数据列:", user_df.columns.tolist())
    
    # 确保用户ID列存在
    user_id_col = '用户id' if '用户id' in weibo_df.columns else 'userid'
    if user_id_col not in weibo_df.columns:
        print(f"错误: 找不到用户ID列 ({user_id_col})")
        return None
    
    # 确保用户数据中有is_spammer列
    if 'is_spammer' not in user_df.columns and '类别' in user_df.columns:
        print("将'类别'列转换为'is_spammer'列...")
        user_df['is_spammer'] = (user_df['类别'] > 0).astype(int)
    elif 'is_spammer' not in user_df.columns:
        print("警告: 用户数据中既没有'is_spammer'列也没有'类别'列，无法区分水军和正常用户")
        print("将随机分配用户类型用于演示...")
        np.random.seed(42)  # 确保可重复性
        user_df['is_spammer'] = np.random.choice([0, 1], size=len(user_df), p=[0.7, 0.3])
    
    # 分离水军和普通用户
    spammer_users = user_df[user_df['is_spammer'] == 1]['id'].values
    normal_users = user_df[user_df['is_spammer'] == 0]['id'].values
    
    print(f"\n检测到 {len(spammer_users)} 个水军用户和 {len(normal_users)} 个普通用户")
    
    # 1. 为普通用户生成更自然、分散的关系
    print("\n为普通用户生成关系网络...")
    
    # 普通用户更倾向于与其他普通用户连接
    for user_id in tqdm(normal_users):
        # 普通用户关系数量与粉丝数相关
        user_data = user_df[user_df['id'] == user_id]
        if '粉丝数' in user_data.columns:
            followers = user_data['粉丝数'].iloc[0]
            n_relations = int(np.clip(np.log2(followers + 1) * 0.7, 3, 15))
        else:
            n_relations = np.random.randint(3, 8)  # 默认关系数量
        
        # 普通用户更可能与其他普通用户连接
        normal_targets = np.random.choice(
            [u for u in normal_users if u != user_id],
            size=min(int(n_relations * 0.8), len(normal_users)-1),
            replace=False
        )
        
        # 少量与水军连接
        if len(spammer_users) > 0:
            spammer_targets = np.random.choice(
                spammer_users,
                size=min(int(n_relations * 0.2), len(spammer_users)),
                replace=False
            )
        else:
            spammer_targets = []
        
        for target_id in np.concatenate([normal_targets, spammer_targets]):
            # 普通用户关系权重分布更均匀
            weight = np.random.beta(2, 2) * 0.8 + 0.2  # 0.2-1.0之间，较均匀分布
            
            relations.append({
                'source_id': user_id,
                'target_id': target_id,
                'weight': weight,
                'relation_type': 'normal_user_relation'
            })
    
    # 2. 为水军用户生成特有的关系模式
    print("\n为水军用户生成关系网络...")
    
    # 水军互相连接的比例更高
    for user_id in tqdm(spammer_users):
        # 水军用户的关系数量更多
        n_relations = np.random.randint(10, 30)
        
        # 水军之间互相连接的概率更高
        other_spammers = [u for u in spammer_users if u != user_id]
        
        if other_spammers:
            spammer_targets = np.random.choice(
                other_spammers,
                size=min(int(n_relations * 0.7), len(other_spammers)),
                replace=False
            )
        else:
            spammer_targets = []
        
        # 也连接一些普通用户作为掩护
        if len(normal_users) > 0:
            normal_targets = np.random.choice(
                normal_users,
                size=min(int(n_relations * 0.3), len(normal_users)),
                replace=False
            )
        else:
            normal_targets = []
        
        for target_id in np.concatenate([spammer_targets, normal_targets]):
            # 水军用户的关系权重分布更极端
            if target_id in spammer_users:
                # 水军之间的连接权重高
                weight = np.random.beta(5, 1.5) * 0.5 + 0.5  # 0.5-1.0之间，偏高
            else:
                # 水军与普通用户的连接权重低
                weight = np.random.beta(1.5, 5) * 0.4 + 0.1  # 0.1-0.5之间，偏低
            
            relations.append({
                'source_id': user_id,
                'target_id': target_id,
                'weight': weight,
                'relation_type': 'spammer_relation'
            })
    
    # 3. 生成圈子结构（水军通常形成紧密圈子）
    print("\n生成水军圈子结构...")
    
    if len(spammer_users) >= 5:
        # 创建水军圈子
        n_circles = max(1, len(spammer_users) // 5)
        spammer_circles = np.array_split(spammer_users, n_circles)
        
        for i, circle in enumerate(spammer_circles):
            print(f"生成第{i+1}个水军圈子（{len(circle)}个用户）")
            
            # 圈子内部互相高度连接
            for source_id in circle:
                for target_id in circle:
                    if source_id != target_id:
                        # 圈子内部高权重连接
                        weight = np.random.uniform(0.8, 1.0)
                        
                        relations.append({
                            'source_id': source_id,
                            'target_id': target_id,
                            'weight': weight,
                            'relation_type': 'spammer_circle'
                        })
    
    # 4. 添加一些随机关系以增加网络复杂性
    print("\n添加随机关系以增加网络复杂性...")
    
    all_users = user_df['id'].values
    n_random_relations = int(len(all_users) * 0.5)  # 随机关系数量
    
    for _ in range(n_random_relations):
        source_id = np.random.choice(all_users)
        target_id = np.random.choice([u for u in all_users if u != source_id])
        weight = np.random.uniform(0.1, 0.3)  # 随机关系权重较低
        
        relations.append({
            'source_id': source_id,
            'target_id': target_id,
            'weight': weight,
            'relation_type': 'random'
        })
    
    # 5. 合并所有关系并处理
    if relations:
        all_relations = pd.DataFrame(relations)
        
        # 统计关系类型分布
        relation_counts = all_relations['relation_type'].value_counts()
        print("\n关系类型分布:")
        for rel_type, count in relation_counts.items():
            print(f"{rel_type}: {count}条")
        
        # 对重复关系进行合并，累加权重
        final_relations = all_relations.groupby(['source_id', 'target_id']).agg({
            'weight': 'max',  # 使用最大权重而不是累加
            'relation_type': lambda x: 'multiple' if len(set(x)) > 1 else x.iloc[0]
        }).reset_index()
        
        # 保存到文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_relations.to_csv(output_path, index=False)
        print(f"\n关系数据已保存到: {output_path}")
        print(f"共生成 {len(final_relations)} 条关系记录")
        
        # 计算并打印网络分析指标
        network_metrics = calculate_network_metrics(final_relations, user_df)
        print("\n网络分析指标:")
        for user_type, metrics in network_metrics.items():
            print(f"{user_type}用户:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        return final_relations
    else:
        print("\n警告: 未能生成任何关系数据")
        return None

def calculate_network_metrics(relations_df, user_df):
    """计算网络分析指标，用于验证生成的关系网络特征"""
    try:
        import networkx as nx
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加所有用户
        for user_id in user_df['id'].values:
            spammer = user_df[user_df['id'] == user_id]['is_spammer'].iloc[0]
            G.add_node(user_id, is_spammer=bool(spammer))
        
        # 添加边
        for _, row in relations_df.iterrows():
            G.add_edge(row['source_id'], row['target_id'], weight=row['weight'])
        
        # 区分水军和普通用户子图
        normal_nodes = [n for n, d in G.nodes(data=True) if not d.get('is_spammer', False)]
        spammer_nodes = [n for n, d in G.nodes(data=True) if d.get('is_spammer', False)]
        
        normal_subgraph = G.subgraph(normal_nodes)
        spammer_subgraph = G.subgraph(spammer_nodes)
        
        # 计算指标
        metrics = {
            '普通': {
                '平均出度': np.mean([d for n, d in G.out_degree(normal_nodes)]) if normal_nodes else 0,
                '平均入度': np.mean([d for n, d in G.in_degree(normal_nodes)]) if normal_nodes else 0,
                '密度': nx.density(normal_subgraph) if normal_nodes else 0,
                '出入度比': np.mean([G.out_degree(n) / max(1, G.in_degree(n)) for n in normal_nodes]) if normal_nodes else 0
            },
            '水军': {
                '平均出度': np.mean([d for n, d in G.out_degree(spammer_nodes)]) if spammer_nodes else 0,
                '平均入度': np.mean([d for n, d in G.in_degree(spammer_nodes)]) if spammer_nodes else 0,
                '密度': nx.density(spammer_subgraph) if spammer_nodes else 0,
                '出入度比': np.mean([G.out_degree(n) / max(1, G.in_degree(n)) for n in spammer_nodes]) if spammer_nodes else 0
            }
        }
        
        return metrics
    except ImportError:
        print("警告: 未安装networkx库，无法计算网络指标")
        return {'普通': {}, '水军': {}}
    except Exception as e:
        print(f"计算网络指标时出错: {str(e)}")
        return {'普通': {}, '水军': {}}

if __name__ == "__main__":
    # 从配置文件加载路径
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config import Config
    
    config = Config()
    
    # 加载数据
    print("加载用户和微博数据...")
    try:
        user_df = pd.read_csv(config.USER_DETAIL_PATH)
        weibo_df = pd.read_csv(config.WEIBO_PATH)
        
        # 生成关系数据
        generate_relations(weibo_df, user_df)
    except Exception as e:
        print(f"错误: {str(e)}")
        print("请确保数据文件存在且格式正确") 