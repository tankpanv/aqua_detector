#!/usr/bin/env python3
"""
简化的表单验证系统
不依赖WTForms，使用基本的表单验证
"""

import re
from flask import request

class FormValidator:
    """简单的表单验证器"""
    
    def __init__(self):
        self.errors = {}
    
    def validate_required(self, field_name, value, message="此字段为必填项"):
        """验证必填字段"""
        if not value or not value.strip():
            self.errors[field_name] = message
            return False
        return True
    
    def validate_length(self, field_name, value, min_len=None, max_len=None):
        """验证长度"""
        if value:
            length = len(value)
            if min_len and length < min_len:
                self.errors[field_name] = f"长度不能少于{min_len}个字符"
                return False
            if max_len and length > max_len:
                self.errors[field_name] = f"长度不能超过{max_len}个字符"
                return False
        return True
    
    def validate_email(self, field_name, value):
        """验证邮箱格式"""
        if value:
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, value):
                self.errors[field_name] = "请输入有效的邮箱地址"
                return False
        return True
    
    def validate_equal(self, field_name, value1, value2, message="两次输入不一致"):
        """验证两个字段是否相等"""
        if value1 != value2:
            self.errors[field_name] = message
            return False
        return True
    
    def has_errors(self):
        """是否有验证错误"""
        return len(self.errors) > 0
    
    def get_errors(self):
        """获取所有错误"""
        return self.errors

def validate_login_form():
    """验证登录表单"""
    validator = FormValidator()
    
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    remember_me = request.form.get('remember_me') == 'on'
    
    validator.validate_required('username', username, '请输入用户名')
    validator.validate_required('password', password, '请输入密码')
    
    return {
        'valid': not validator.has_errors(),
        'errors': validator.get_errors(),
        'data': {
            'username': username,
            'password': password,
            'remember_me': remember_me
        }
    }

def validate_register_form():
    """验证注册表单"""
    validator = FormValidator()
    
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip()
    real_name = request.form.get('real_name', '').strip()
    phone = request.form.get('phone', '').strip()
    organization = request.form.get('organization', '').strip()
    password = request.form.get('password', '')
    password2 = request.form.get('password2', '')
    
    # 验证必填字段
    validator.validate_required('username', username, '请输入用户名')
    validator.validate_required('email', email, '请输入邮箱')
    validator.validate_required('password', password, '请输入密码')
    validator.validate_required('password2', password2, '请确认密码')
    
    # 验证长度
    validator.validate_length('username', username, 3, 64)
    validator.validate_length('password', password, 6, 128)
    validator.validate_length('real_name', real_name, 0, 64)
    validator.validate_length('phone', phone, 0, 20)
    validator.validate_length('organization', organization, 0, 120)
    
    # 验证邮箱格式
    validator.validate_email('email', email)
    
    # 验证密码确认
    validator.validate_equal('password2', password, password2, '两次输入的密码不一致')
    
    return {
        'valid': not validator.has_errors(),
        'errors': validator.get_errors(),
        'data': {
            'username': username,
            'email': email,
            'real_name': real_name or None,
            'phone': phone or None,
            'organization': organization or None,
            'password': password
        }
    }

def validate_profile_form():
    """验证个人资料表单"""
    validator = FormValidator()
    
    real_name = request.form.get('real_name', '').strip()
    phone = request.form.get('phone', '').strip()
    organization = request.form.get('organization', '').strip()
    
    # 验证长度
    validator.validate_length('real_name', real_name, 0, 64)
    validator.validate_length('phone', phone, 0, 20)
    validator.validate_length('organization', organization, 0, 120)
    
    return {
        'valid': not validator.has_errors(),
        'errors': validator.get_errors(),
        'data': {
            'real_name': real_name or None,
            'phone': phone or None,
            'organization': organization or None
        }
    }

def validate_change_password_form():
    """验证修改密码表单"""
    validator = FormValidator()
    
    old_password = request.form.get('old_password', '')
    password = request.form.get('password', '')
    password2 = request.form.get('password2', '')
    
    # 验证必填字段
    validator.validate_required('old_password', old_password, '请输入当前密码')
    validator.validate_required('password', password, '请输入新密码')
    validator.validate_required('password2', password2, '请确认新密码')
    
    # 验证新密码长度
    validator.validate_length('password', password, 6, 128)
    
    # 验证密码确认
    validator.validate_equal('password2', password, password2, '两次输入的密码不一致')
    
    return {
        'valid': not validator.has_errors(),
        'errors': validator.get_errors(),
        'data': {
            'old_password': old_password,
            'password': password
        }
    } 