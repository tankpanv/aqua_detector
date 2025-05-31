#!/usr/bin/env python3
"""
水军检测系统 - 最简化版本
只包含用户认证功能，不加载AI模型
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"当前目录: {current_dir}")
print(f"项目根目录: {project_root}")

# 导入简化的认证系统
from web.simple_auth import (
    init_db, get_db, close_db, login_required, get_current_user,
    authenticate_user, login_user, logout_user, create_user,
    update_user_profile, change_password, save_detection_history,
    get_detection_history
)
from web.simple_forms import (
    validate_login_form, validate_register_form, validate_profile_form,
    validate_change_password_form
)

app = Flask(__name__)

# 配置密钥
app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600 * 24 * 7  # 7天

# 注册数据库关闭函数
app.teardown_appcontext(close_db)

# 初始化数据库需要在应用上下文中
with app.app_context():
    print("初始化数据库...")
    init_db()
    print("数据库初始化完成")

# 添加模板上下文处理器
@app.context_processor
def inject_user():
    """向模板注入当前用户信息"""
    return dict(current_user=get_current_user())

@app.route('/')
@login_required
def index():
    return render_template('index.html')

# 用户认证路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    current_user = get_current_user()
    if current_user:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        form_data = validate_login_form()
        
        if form_data['valid']:
            user = authenticate_user(form_data['data']['username'], form_data['data']['password'])
            if user:
                login_user(user, form_data['data']['remember_me'])
                next_page = request.args.get('next')
                if not next_page or not next_page.startswith('/'):
                    next_page = url_for('index')
                return redirect(next_page)
            else:
                flash('用户名或密码错误', 'error')
        else:
            for field, error in form_data['errors'].items():
                flash(error, 'error')
    
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    current_user = get_current_user()
    if current_user:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        form_data = validate_register_form()
        
        if form_data['valid']:
            success, message = create_user(**form_data['data'])
            if success:
                flash('恭喜您！注册成功，请使用新账户登录。', 'success')
                return redirect(url_for('login'))
            else:
                flash(message, 'error')
        else:
            for field, error in form_data['errors'].items():
                flash(error, 'error')
    
    return render_template('auth/register.html')

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def profile():
    user = get_current_user()
    return render_template('auth/profile.html', user=user)

@app.route('/edit_profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    if request.method == 'POST':
        form_data = validate_profile_form()
        
        if form_data['valid']:
            current_user = get_current_user()
            success, message = update_user_profile(current_user['id'], **form_data['data'])
            if success:
                flash('个人资料已更新', 'success')
                return redirect(url_for('profile'))
            else:
                flash(message, 'error')
        else:
            for field, error in form_data['errors'].items():
                flash(error, 'error')
    
    return render_template('auth/edit_profile.html')

@app.route('/change_password', methods=['GET', 'POST'])
@login_required
def change_password():
    if request.method == 'POST':
        form_data = validate_change_password_form()
        
        if form_data['valid']:
            current_user = get_current_user()
            success, message = change_password(current_user['id'], **form_data['data'])
            if success:
                flash('密码已成功修改', 'success')
                return redirect(url_for('profile'))
            else:
                flash(message, 'error')
        else:
            for field, error in form_data['errors'].items():
                flash(error, 'error')
    
    return render_template('auth/change_password.html')

@app.route('/history')
@login_required
def detection_history():
    """查看检测历史"""
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    current_user = get_current_user()
    history_data = get_detection_history(current_user['id'], page, per_page)
    
    return render_template('auth/history.html', history=history_data)

@app.route('/detect', methods=['POST'])
@login_required
def detect():
    """简化的检测接口，返回模拟结果"""
    data = request.get_json()
    
    # 模拟检测结果
    result = {
        'is_spammer': False,
        'confidence': 85,
        'message': '当前为演示模式，AI模型尚未加载',
        'model_type': 'demo'
    }
    
    # 保存检测历史
    current_user = get_current_user()
    if current_user:
        input_content = data.get('text', '') or data.get('user_id', '')
        save_detection_history(current_user['id'], '演示模式', input_content, result)
    
    return jsonify(result)

@app.route('/network_analysis')
@login_required
def network_analysis():
    """网络分析页面"""
    return render_template('network_analysis.html')

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='水军检测系统最简版')
    parser.add_argument('--port', type=int, default=8888, help='Web服务端口')
    args = parser.parse_args()
    
    print(f"启动Web服务，端口: {args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=True) 