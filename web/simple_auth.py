#!/usr/bin/env python3
"""
简化的用户认证系统
不依赖Flask-SQLAlchemy、Flask-Login等外部包
使用内置的sqlite3和Flask session
"""

import sqlite3
import hashlib
import os
from datetime import datetime
from functools import wraps
from flask import session, request, redirect, url_for, flash, g

# 数据库文件路径
DATABASE = 'aqua_detector.db'

def get_db():
    """获取数据库连接"""
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

def close_db(e=None):
    """关闭数据库连接"""
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_db():
    """初始化数据库"""
    db = get_db()
    
    # 创建用户表
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            real_name TEXT,
            phone TEXT,
            organization TEXT,
            role TEXT DEFAULT 'user',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            detection_count INTEGER DEFAULT 0,
            last_detection TIMESTAMP
        )
    ''')
    
    # 创建检测历史表
    db.execute('''
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            detection_type TEXT NOT NULL,
            input_content TEXT,
            result TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    db.commit()
    
    # 检查是否存在管理员账户
    admin = db.execute('SELECT * FROM users WHERE username = ?', ('admin',)).fetchone()
    if not admin:
        # 创建默认管理员账户
        password_hash = hash_password('admin123')
        db.execute('''
            INSERT INTO users (username, email, password_hash, real_name, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', 'admin@example.com', password_hash, '系统管理员', 'admin'))
        db.commit()
        print("默认管理员账户创建成功 - 用户名: admin, 密码: admin123")

def hash_password(password):
    """密码哈希"""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def verify_password(password, password_hash):
    """验证密码"""
    return hash_password(password) == password_hash

def login_required(f):
    """登录装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录后访问此页面', 'info')
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """获取当前用户"""
    if 'user_id' not in session:
        return None
    
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE id = ?', (session['user_id'],)).fetchone()
    return dict(user) if user else None

def create_user(username, email, password, real_name=None, phone=None, organization=None):
    """创建用户"""
    db = get_db()
    
    # 检查用户名是否已存在
    existing_user = db.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
    if existing_user:
        return False, "该用户名已被使用"
    
    # 检查邮箱是否已存在
    existing_email = db.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
    if existing_email:
        return False, "该邮箱已被注册"
    
    # 创建用户
    password_hash = hash_password(password)
    try:
        db.execute('''
            INSERT INTO users (username, email, password_hash, real_name, phone, organization)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, email, password_hash, real_name, phone, organization))
        db.commit()
        return True, "注册成功"
    except Exception as e:
        return False, f"注册失败: {str(e)}"

def authenticate_user(username, password):
    """用户认证"""
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
    
    if user and verify_password(password, user['password_hash']):
        # 更新最后登录时间
        db.execute('UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (user['id'],))
        db.commit()
        return dict(user)
    return None

def login_user(user, remember=False):
    """用户登录"""
    session['user_id'] = user['id']
    session['username'] = user['username']
    if remember:
        session.permanent = True

def logout_user():
    """用户登出"""
    session.pop('user_id', None)
    session.pop('username', None)

def update_user_profile(user_id, real_name=None, phone=None, organization=None):
    """更新用户资料"""
    db = get_db()
    try:
        db.execute('''
            UPDATE users SET real_name = ?, phone = ?, organization = ?
            WHERE id = ?
        ''', (real_name, phone, organization, user_id))
        db.commit()
        return True, "资料更新成功"
    except Exception as e:
        return False, f"更新失败: {str(e)}"

def change_password(user_id, old_password, new_password):
    """修改密码"""
    db = get_db()
    user = db.execute('SELECT password_hash FROM users WHERE id = ?', (user_id,)).fetchone()
    
    if not user or not verify_password(old_password, user['password_hash']):
        return False, "当前密码错误"
    
    try:
        new_password_hash = hash_password(new_password)
        db.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_password_hash, user_id))
        db.commit()
        return True, "密码修改成功"
    except Exception as e:
        return False, f"密码修改失败: {str(e)}"

def save_detection_history(user_id, detection_type, input_content, result):
    """保存检测历史"""
    db = get_db()
    try:
        import json
        result_json = json.dumps(result, ensure_ascii=False)
        
        db.execute('''
            INSERT INTO detection_history (user_id, detection_type, input_content, result)
            VALUES (?, ?, ?, ?)
        ''', (user_id, detection_type, input_content, result_json))
        
        # 更新用户检测次数
        db.execute('''
            UPDATE users SET detection_count = detection_count + 1, last_detection = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (user_id,))
        
        db.commit()
        return True
    except Exception as e:
        print(f"保存检测历史失败: {str(e)}")
        return False

def get_detection_history(user_id, page=1, per_page=10):
    """获取检测历史"""
    db = get_db()
    offset = (page - 1) * per_page
    
    # 获取总数
    total = db.execute('SELECT COUNT(*) FROM detection_history WHERE user_id = ?', (user_id,)).fetchone()[0]
    
    # 获取分页数据
    history = db.execute('''
        SELECT * FROM detection_history WHERE user_id = ?
        ORDER BY created_at DESC LIMIT ? OFFSET ?
    ''', (user_id, per_page, offset)).fetchall()
    
    # 解析结果JSON
    import json
    history_list = []
    for record in history:
        record_dict = dict(record)
        try:
            record_dict['result'] = json.loads(record_dict['result'])
        except:
            record_dict['result'] = {}
        history_list.append(record_dict)
    
    return {
        'items': history_list,
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': (total + per_page - 1) // per_page,
        'has_prev': page > 1,
        'has_next': page * per_page < total,
        'prev_num': page - 1 if page > 1 else None,
        'next_num': page + 1 if page * per_page < total else None
    } 