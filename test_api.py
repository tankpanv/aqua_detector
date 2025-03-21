import requests
import json
import time

def test_detection_api(user_id):
    """测试用户检测API"""
    url = 'http://localhost:5000/detect'
    
    print(f"测试用户ID: {user_id}")
    
    data = {
        'user_id': user_id
    }
    
    try:
        response = requests.post(url, json=data)
        
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("成功获取检测结果:")
            result = response.json()
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # 打印主要检测结果
            print("\n摘要:")
            print(f"用户ID: {result.get('user_id')}")
            print(f"是否水军: {result.get('is_spammer')}")
            print(f"置信度: {result.get('confidence')}%")
            print(f"模型类型: {result.get('model_type')}")
            if 'model_agreement' in result and result['model_agreement'] is not None:
                print(f"模型一致性: {result['model_agreement']:.2f}")
            print(f"可疑得分: {result.get('suspicious_score', 0):.2f}")
        else:
            print("错误:")
            print(response.text)
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    # 从数据中找一个用户ID测试
    import pandas as pd
    
    try:
        # 尝试加载用户数据
        user_df = pd.read_csv("data/user_detail.csv")
        # 获取前5个用户ID进行测试
        test_ids = user_df['id'].astype(str).values[:5]
        
        for user_id in test_ids:
            test_detection_api(user_id)
            time.sleep(1)  # 等待1秒
            print("="*50)
    except Exception as e:
        print(f"无法加载用户数据: {e}")
        # 使用一个可能存在的用户ID进行测试
        test_detection_api("6941098054") 