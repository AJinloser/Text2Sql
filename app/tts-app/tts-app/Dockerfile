# 使用官方 Python 镜像作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app/tts-app

# 安装 Python 依赖
# RUN pip install --no-cache-dir -r requirements.txt

# 把对应文件复制到容器的 /app 目录
# COPY . /app

# 创建输出目录
# RUN mkdir -p /app/output/audio

# 暴露端口（如果你的 Python 服务监听某个端口）
EXPOSE 5000

# 启动 Python 服务(docker-compose中的command会覆盖这里)
CMD ["python", "tts_ws.py"]