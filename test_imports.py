#!/usr/bin/env python3
# 测试用户认证系统所需的包

print("检查用户认证系统依赖...")

try:
    from flask_sqlalchemy import SQLAlchemy
    print("✓ Flask-SQLAlchemy 可用")
except ImportError as e:
    print("✗ Flask-SQLAlchemy 不可用:", e)

try:
    from flask_login import LoginManager
    print("✓ Flask-Login 可用")
except ImportError as e:
    print("✗ Flask-Login 不可用:", e)

try:
    from flask_wtf import FlaskForm
    print("✓ Flask-WTF 可用")
except ImportError as e:
    print("✗ Flask-WTF 不可用:", e)

try:
    from wtforms import StringField
    print("✓ WTForms 可用")
except ImportError as e:
    print("✗ WTForms 不可用:", e)

try:
    from werkzeug.security import generate_password_hash
    print("✓ Werkzeug 密码哈希 可用")
except ImportError as e:
    print("✗ Werkzeug 密码哈希 不可用:", e)

print("\n检查完成！") 