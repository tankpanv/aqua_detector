#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阿里云OSS环境检查脚本 (单文件版)
用于检查上传脚本的环境配置是否正确
"""

import os
import sys
import re

def check_python_version():
    """检查Python版本"""
    print("🐍 检查Python版本...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 6:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro} (符合要求)")
        return True
    else:
        print(f"❌ Python版本: {version.major}.{version.minor}.{version.micro} (需要3.6+)")
        return False

def check_oss2_installation():
    """检查oss2库是否安装"""
    print("\n📦 检查oss2库安装状态...")
    try:
        import oss2
        print(f"✅ oss2库已安装，版本: {oss2.__version__}")
        return True
    except ImportError:
        print("❌ oss2库未安装")
        print("💡 安装命令:")
        print("   pip install oss2")
        print("   或者: conda install -c conda-forge oss2")
        return False

def check_upload_script():
    """检查上传脚本和配置"""
    print("\n📄 检查上传脚本...")
    
    if not os.path.exists('upload_to_oss.py'):
        print("❌ 上传脚本 upload_to_oss.py 不存在")
        return False
    
    # 检查文件是否可读
    try:
        with open('upload_to_oss.py', 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 检查基本功能是否存在
            if 'def upload_file_to_oss' not in content:
                print("❌ 上传脚本格式不正确")
                return False
            
            print("✅ 上传脚本存在且格式正确")
            
            # 检查配置是否已设置
            print("\n⚙️ 检查脚本内配置...")
            
            # 提取配置信息
            config_issues = []
            
            # 检查ACCESS_KEY_ID
            access_key_match = re.search(r"ACCESS_KEY_ID\s*=\s*['\"]([^'\"]*)['\"]", content)
            if not access_key_match or not access_key_match.group(1) or access_key_match.group(1) == 'your_access_key_id':
                config_issues.append("ACCESS_KEY_ID 未配置或使用默认值")
            else:
                access_key = access_key_match.group(1)
                print(f"   🔑 AccessKey ID: {access_key[:8]}***")
            
            # 检查ACCESS_KEY_SECRET
            secret_match = re.search(r"ACCESS_KEY_SECRET\s*=\s*['\"]([^'\"]*)['\"]", content)
            if not secret_match or not secret_match.group(1) or secret_match.group(1) == 'your_access_key_secret':
                config_issues.append("ACCESS_KEY_SECRET 未配置或使用默认值")
            
            # 检查BUCKET_NAME
            bucket_match = re.search(r"BUCKET_NAME\s*=\s*['\"]([^'\"]*)['\"]", content)
            if not bucket_match or not bucket_match.group(1) or bucket_match.group(1) == 'your_bucket_name':
                config_issues.append("BUCKET_NAME 未配置或使用默认值")
            else:
                bucket_name = bucket_match.group(1)
                print(f"   📦 存储桶: {bucket_name}")
            
            # 检查ENDPOINT
            endpoint_match = re.search(r"ENDPOINT\s*=\s*['\"]([^'\"]*)['\"]", content)
            if endpoint_match and endpoint_match.group(1):
                endpoint = endpoint_match.group(1)
                print(f"   🌍 节点: {endpoint}")
            else:
                config_issues.append("ENDPOINT 未配置")
            
            # 检查PUBLIC_DIR
            public_dir_match = re.search(r"PUBLIC_DIR\s*=\s*['\"]([^'\"]*)['\"]", content)
            if public_dir_match and public_dir_match.group(1):
                public_dir = public_dir_match.group(1)
                print(f"   📁 目标目录: {public_dir}")
            
            if config_issues:
                print("⚠️ 配置问题:")
                for issue in config_issues:
                    print(f"   - {issue}")
                print("💡 请编辑脚本开头的 'OSS配置信息' 部分")
                return False
            else:
                print("✅ 脚本配置检查通过")
                return True
            
    except Exception as e:
        print(f"❌ 无法读取上传脚本: {e}")
        return False

def check_network_connectivity():
    """检查网络连接（简单测试）"""
    print("\n🌐 检查网络连接...")
    
    try:
        import urllib.request
        import socket
        
        # 测试连接阿里云OSS
        socket.setdefaulttimeout(10)
        response = urllib.request.urlopen('https://oss.console.aliyun.com', timeout=10)
        if response.getcode() == 200:
            print("✅ 网络连接正常，可以访问阿里云")
            return True
        else:
            print("⚠️ 网络连接可能有问题")
            return False
    except Exception as e:
        print(f"⚠️ 网络连接测试失败: {e}")
        print("💡 请检查网络连接或防火墙设置")
        return False

def print_usage_examples():
    """打印使用示例"""
    print("\n📖 使用示例:")
    print("=" * 50)
    print("# 1. 上传单个文件")
    print("python upload_to_oss.py --file /path/to/image.jpg")
    print()
    print("# 2. 上传文件并重命名")
    print("python upload_to_oss.py --file document.pdf --object-name my-doc.pdf")
    print()
    print("# 3. 批量上传")
    print("python upload_to_oss.py --file file1.txt file2.jpg file3.pdf")
    print()
    print("# 4. 查看配置")
    print("python upload_to_oss.py --show-config")
    print()
    print("# 5. 查看帮助")
    print("python upload_to_oss.py --help")

def print_config_guide():
    """打印配置指南"""
    print("\n🔧 配置指南:")
    print("=" * 50)
    print("1. 用文本编辑器打开 upload_to_oss.py")
    print("2. 找到脚本开头的 'OSS配置信息' 部分")
    print("3. 修改以下配置项:")
    print("   - ACCESS_KEY_ID: 您的阿里云AccessKey ID")
    print("   - ACCESS_KEY_SECRET: 您的阿里云AccessKey Secret")
    print("   - BUCKET_NAME: OSS存储桶名称")
    print("   - ENDPOINT: OSS服务节点地址")
    print("   - PUBLIC_DIR: 上传目标目录（可选）")

def main():
    """主函数"""
    print("🔧 阿里云OSS上传工具环境检查 (单文件版)")
    print("=" * 60)
    
    checks = [
        ("Python版本", check_python_version),
        ("oss2库安装", check_oss2_installation),
        ("上传脚本和配置", check_upload_script),
        ("网络连接", check_network_connectivity),
    ]
    
    passed = 0
    total = len(checks)
    
    for name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print(f"❌ {name}检查时发生错误: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 检查结果: {passed}/{total} 项通过")
    
    if passed == total:
        print("🎉 所有检查通过！环境配置正确，可以开始使用OSS上传工具。")
        print_usage_examples()
    else:
        print("⚠️ 部分检查未通过，请根据上述提示解决问题后重新检查。")
        
        if passed < 2:  # 基础环境有问题
            print("\n🔧 优先解决事项:")
            print("1. 确保Python 3.6+已安装")
            print("2. 安装oss2库: pip install oss2")
        elif passed < 3:  # 配置问题
            print("\n🔧 优先解决事项:")
            print("1. 确保上传脚本文件完整")
            print("2. 正确配置脚本内的OSS信息")
            print_config_guide()
        else:  # 网络问题
            print("\n🔧 优先解决事项:")
            print("1. 检查网络连接")
            print("2. 确认防火墙设置")

if __name__ == '__main__':
    main() 