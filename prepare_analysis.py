#!/usr/bin/env python3
"""
准备分析图像脚本

这个脚本用于在启动Web应用前独立生成所有必要的网络分析图像
"""

import os
import sys
from web.generate_analysis import ensure_dirs, generate_real_analysis_images, generate_sample_images

if __name__ == "__main__":
    print("开始准备网络分析图像...")
    
    # 确保目录存在
    ensure_dirs()
    
    # 尝试使用真实数据生成图像
    success = generate_real_analysis_images()
    
    # 如果真实数据分析失败，使用示例数据生成图像
    if not success:
        print("使用示例数据生成分析图像...")
        generate_sample_images()
        print("示例分析图像生成完成")
    
    print("所有网络分析图像已生成，现在可以启动Web应用了。")
    print("运行以下命令启动Web应用：")
    print("  python web/app.py") 