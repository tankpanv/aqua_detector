# 阿里云OSS文件上传工具（单文件版）

这是一个用于将文件上传到阿里云OSS存储桶的Python脚本。**所有配置信息都已集成在脚本内部，无需额外配置文件。**

## 功能特性

- ✅ 支持单文件和批量文件上传
- ✅ 自动分片上传大文件（大于100MB）
- ✅ 上传进度显示
- ✅ 文件完整性验证（MD5校验）
- ✅ 自动重试机制
- ✅ 详细的错误处理
- ✅ **单文件集成配置**（无需外部配置文件）
- ✅ 中文界面和emoji提示

## 安装依赖

### 方法1：使用pip安装
```bash
pip install oss2
```

### 方法2：使用conda安装
```bash
conda install -c conda-forge oss2
```

## 配置设置

**不需要创建单独的配置文件！** 直接编辑 `upload_to_oss.py` 脚本开头的配置部分：

1. 用文本编辑器打开 `upload_to_oss.py`
2. 找到脚本开头的 "OSS配置信息" 部分
3. 修改以下配置项：

```python
# OSS访问凭证
ACCESS_KEY_ID = 'your_access_key_id'        # 替换为您的AccessKey ID
ACCESS_KEY_SECRET = 'your_access_key_secret' # 替换为您的AccessKey Secret

# OSS存储桶配置
BUCKET_NAME = 'your_bucket_name'            # 替换为您的存储桶名称
ENDPOINT = 'https://oss-cn-beijing.aliyuncs.com'  # 根据地域修改

# 上传目录设置
PUBLIC_DIR = 'public'  # 目标目录前缀，可根据需要修改
```

## 使用方法

### 基本用法

#### 1. 上传单个文件（使用原文件名）
```bash
python upload_to_oss.py --file /path/to/image.jpg
```

#### 2. 上传文件并指定OSS对象名称
```bash
python upload_to_oss.py --file /path/to/document.pdf --object-name my-document.pdf
```

#### 3. 批量上传多个文件
```bash
python upload_to_oss.py --file file1.txt file2.jpg file3.pdf
```

#### 4. 显示配置信息
```bash
python upload_to_oss.py --show-config
```

#### 5. 查看帮助信息
```bash
python upload_to_oss.py --help
```

### 高级用法

#### 上传到指定目录
脚本会自动将文件上传到配置的目录下（默认为 `public`）。

例如，上传 `image.jpg` 后，OSS中的完整路径为：`public/image.jpg`

#### 大文件上传
- 文件大于100MB时自动使用分片上传
- 显示实时上传进度
- 支持断点续传（网络中断时自动重试）

## 输出示例

```
🚀 开始上传 1 个文件到阿里云OSS...
📦 目标存储桶: wealthgarden
📁 目标目录: public
============================================================

[1/1] 🔄 处理文件: /path/to/image.jpg
📁 准备上传文件: /path/to/image.jpg
📊 文件大小: 2.5 MB
🔐 文件MD5: d41d8cd98f00b204e9800998ecf8427e
☁️ 开始上传到: wealthgarden/public/image.jpg
📤 使用普通上传...
📊 上传进度: 100.0% (2.5 MB/2.5 MB)
✅ 上传成功!
📝 OSS对象名称: public/image.jpg
🌐 访问URL: https://wealthgarden.oss-cn-beijing.aliyuncs.com/public/image.jpg
🏷️ ETag: "d41d8cd98f00b204e9800998ecf8427e"
✅ 文件大小验证通过: 2.5 MB

============================================================
📊 上传完成！成功: 1/1
⏱️ 总耗时: 3.45 秒
🎉 所有文件上传成功！
```

## 错误处理

脚本包含完整的错误处理机制：

- **文件不存在**：检查文件路径是否正确
- **网络超时**：自动重试最多3次
- **权限错误**：检查AccessKey权限设置
- **存储桶不存在**：检查存储桶名称和地域设置
- **配置未设置**：检查脚本内的配置信息是否正确

## 部署和移植

### 优势
- ✅ **单文件部署**：只需复制一个 `upload_to_oss.py` 文件
- ✅ **无依赖配置文件**：所有配置都在脚本内部
- ✅ **便于移植**：可以直接在其他服务器或环境中运行
- ✅ **版本控制友好**：可以为不同环境创建不同版本的脚本

### 在其他环境使用
1. 将 `upload_to_oss.py` 复制到目标环境
2. 安装依赖：`pip install oss2`
3. 编辑脚本开头的配置信息
4. 直接运行即可

## 安全注意事项

1. **保护AccessKey**：
   - 不要将包含真实AccessKey的脚本提交到公共代码仓库
   - 可以创建一个模板版本（AccessKey使用占位符）用于分享
   - 定期更换AccessKey

2. **权限最小化**：
   - 只授予必要的OSS权限
   - 建议使用RAM子账号而非主账号

3. **版本管理**：
   - 为不同环境创建不同的脚本副本
   - 使用 `.gitignore` 忽略包含敏感信息的版本

## 常见问题

### Q: 如何查看当前配置？
A: 使用 `python upload_to_oss.py --show-config` 命令

### Q: 如何修改配置？
A: 直接编辑脚本开头的 "OSS配置信息" 部分，无需外部配置文件

### Q: 上传失败怎么办？
A: 检查网络连接、AccessKey权限、存储桶配置等，脚本会自动重试3次

### Q: 如何上传到不同的目录？
A: 修改脚本中的 `PUBLIC_DIR` 配置项

### Q: 支持哪些文件类型？
A: 支持所有文件类型，没有限制

### Q: 如何在不同环境使用不同配置？
A: 为每个环境创建一个单独的脚本副本，分别配置不同的参数

## 文件结构

```
.
├── upload_to_oss.py         # 主上传脚本（包含所有配置）
├── check_oss_setup.py       # 环境检查脚本（可选）
└── OSS_UPLOAD_README.md     # 使用说明
```

## 版本信息

- 版本：v1.1 (单文件版)
- 更新内容：合并配置文件，便于部署和移植
- 作者：AI Assistant
- 最后更新：2024年

## 从旧版本升级

如果您之前使用的是包含独立 `oss_config.py` 配置文件的版本：

1. 备份您的 `oss_config.py` 中的配置信息
2. 下载新版本的 `upload_to_oss.py`
3. 将备份的配置信息填入新脚本开头的配置部分
4. 删除旧的 `oss_config.py` 文件
5. 测试新版本脚本

---

**注意：** 请确保您的阿里云账号有足够的OSS存储空间和权限。 