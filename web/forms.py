from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField, SelectField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError, Optional
from web.models import User

class LoginForm(FlaskForm):
    """登录表单"""
    username = StringField('用户名', validators=[DataRequired(), Length(1, 64)])
    password = PasswordField('密码', validators=[DataRequired()])
    remember_me = BooleanField('记住我')
    submit = SubmitField('登录')

class RegistrationForm(FlaskForm):
    """注册表单"""
    username = StringField('用户名', validators=[
        DataRequired(), 
        Length(3, 64, message='用户名长度必须在3-64个字符之间')
    ])
    email = StringField('邮箱', validators=[
        DataRequired(), 
        Email(message='请输入有效的邮箱地址')
    ])
    real_name = StringField('真实姓名', validators=[
        Optional(), 
        Length(0, 64, message='姓名长度不能超过64个字符')
    ])
    phone = StringField('手机号', validators=[
        Optional(), 
        Length(0, 20, message='手机号长度不能超过20个字符')
    ])
    organization = StringField('所属机构', validators=[
        Optional(), 
        Length(0, 120, message='机构名称长度不能超过120个字符')
    ])
    password = PasswordField('密码', validators=[
        DataRequired(),
        Length(6, 128, message='密码长度必须在6-128个字符之间')
    ])
    password2 = PasswordField('确认密码', validators=[
        DataRequired(), 
        EqualTo('password', message='两次输入的密码不一致')
    ])
    submit = SubmitField('注册')
    
    def validate_username(self, username):
        """验证用户名是否已存在"""
        user = User.query.filter_by(username=username.data).first()
        if user:
            raise ValidationError('该用户名已被使用，请选择其他用户名')
    
    def validate_email(self, email):
        """验证邮箱是否已存在"""
        user = User.query.filter_by(email=email.data).first()
        if user:
            raise ValidationError('该邮箱已被注册，请使用其他邮箱')

class ProfileForm(FlaskForm):
    """个人资料编辑表单"""
    real_name = StringField('真实姓名', validators=[
        Optional(), 
        Length(0, 64, message='姓名长度不能超过64个字符')
    ])
    phone = StringField('手机号', validators=[
        Optional(), 
        Length(0, 20, message='手机号长度不能超过20个字符')
    ])
    organization = StringField('所属机构', validators=[
        Optional(), 
        Length(0, 120, message='机构名称长度不能超过120个字符')
    ])
    submit = SubmitField('更新资料')

class ChangePasswordForm(FlaskForm):
    """修改密码表单"""
    old_password = PasswordField('当前密码', validators=[DataRequired()])
    password = PasswordField('新密码', validators=[
        DataRequired(),
        Length(6, 128, message='密码长度必须在6-128个字符之间')
    ])
    password2 = PasswordField('确认新密码', validators=[
        DataRequired(), 
        EqualTo('password', message='两次输入的密码不一致')
    ])
    submit = SubmitField('修改密码') 