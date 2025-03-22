import os
import platform
import locale
import chardet

def get_system_info():
    """获取系统信息"""
    return {
        'system': platform.system(),
        'release': platform.release(),
        'python_version': platform.python_version(),
        'default_encoding': locale.getpreferredencoding(),
    }

def get_platform_path(path_components):
    """创建跨平台兼容的路径"""
    # 转换所有正斜杠为系统适合的分隔符
    path = os.path.join(*path_components)
    # 确保路径存在
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    return path

def detect_file_encoding(file_path, default='utf-8'):
    """检测文件编码"""
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read(10000))
        return result['encoding'] if result['encoding'] else default
    except Exception:
        return default

def safe_read_csv(file_path, encodings=None):
    """安全读取CSV文件，自动处理编码问题"""
    import pandas as pd
    
    if encodings is None:
        encodings = ['utf-8', 'gbk', 'latin1', locale.getpreferredencoding()]
    
    # 先尝试检测编码
    detected_encoding = detect_file_encoding(file_path)
    if detected_encoding:
        encodings.insert(0, detected_encoding)
    
    # 遍历尝试不同编码
    last_error = None
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            break
    
    # 所有编码都失败，抛出最后的错误
    raise last_error 