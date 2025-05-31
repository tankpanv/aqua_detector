from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()

class User(UserMixin, db.Model):
    """用户模型"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    real_name = db.Column(db.String(64), nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    organization = db.Column(db.String(120), nullable=True)
    role = db.Column(db.String(20), nullable=False, default='user')  # user, admin
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # 用户使用记录
    detection_count = db.Column(db.Integer, default=0)  # 检测次数
    last_detection = db.Column(db.DateTime, nullable=True)  # 最后检测时间
    
    def set_password(self, password):
        """设置密码"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """验证密码"""
        return check_password_hash(self.password_hash, password)
    
    def is_admin(self):
        """检查是否为管理员"""
        return self.role == 'admin'
    
    def increment_detection_count(self):
        """增加检测次数"""
        self.detection_count += 1
        self.last_detection = datetime.utcnow()
        db.session.commit()
    
    def __repr__(self):
        return f'<User {self.username}>'

class DetectionHistory(db.Model):
    """检测历史记录"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    detection_type = db.Column(db.String(20), nullable=False)  # 'text' or 'user_id'
    input_content = db.Column(db.Text, nullable=True)  # 输入的文本或用户ID
    result = db.Column(db.JSON, nullable=False)  # 检测结果
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref=db.backref('detection_history', lazy=True))
    
    def __repr__(self):
        return f'<DetectionHistory {self.id}>' 