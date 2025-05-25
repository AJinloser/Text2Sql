# -*- coding:utf-8 -*-
#
#   author: iflytek
#
#  本demo测试时运行的环境为：Windows + Python3.7
#  本demo测试成功运行时所安装的第三方库及其版本如下：
#   cffi==1.12.3
#   gevent==1.4.0
#   greenlet==0.4.15
#   pycparser==2.19
#   six==1.12.0
#   websocket==0.2.1
#   websocket-client==0.56.0
#   合成小语种需要传输小语种文本、使用小语种发音人vcn、tte=unicode以及修改文本编码方式
#  错误码链接：https://www.xfyun.cn/document/error-code （code返回错误码时必看）
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import time
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import _thread as thread
import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from functools import partial

app = FastAPI()

STATUS_FIRST_FRAME = 0  # 第一帧的标识
STATUS_CONTINUE_FRAME = 1  # 中间帧标识
STATUS_LAST_FRAME = 2  # 最后一帧的标识

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))  # 代码文件在Dify_flow/utils中
out_path = os.path.join(base_dir, 'output/audio')
pcm_path = os.path.join(out_path, 'audio_file.pcm')
# wav_path = os.path.join(out_path, 'audio_file.wav')

wav_file = ""

APPID='0eca74cf'
APISecret='NDlkMzEyMDgzNzY0MmI3ZTcyNjI1MDE2'
APIKey='086e9de3b1d7555e44e83fc48cd998b6'

"""
音色
vcn: 女生：xiaoyan, aisxping, aisjinger, 男生：aisbabyxu, aisjiuxu, 感情男：x4_lingfeizhe_emo, 感情女：x4_lingxiaoyao_em
"""
vcn = 'x4_lingxiaoyao_em'


class Ws_Param(object):
    # 初始化
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.Text = Text

        # 公共参数(common)
        self.CommonArgs = {"app_id": self.APPID}
        # 业务参数(business)，更多个性化参数可在官网查看
        self.BusinessArgs = {"aue": "raw", "auf": "audio/L16;rate=16000", "vcn": vcn, "tte": "utf8"}
        self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-8')), "UTF8")}
        #使用小语种须使用以下方式，此处的unicode指的是 utf16小端的编码方式，即"UTF-16LE"”
        #self.Data = {"status": 2, "text": str(base64.b64encode(self.Text.encode('utf-16')), "UTF8")}

    # 生成url
    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + "ws-api.xfyun.cn" + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + "/v2/tts " + "HTTP/1.1"
        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')

        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            self.APIKey, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        # 将请求的鉴权参数组合为字典
        v = {
            "authorization": authorization,
            "date": date,
            "host": "ws-api.xfyun.cn"
        }
        # 拼接鉴权参数，生成url
        url = url + '?' + urlencode(v)
        # print("date: ",date)
        # print("v: ",v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        # print('websocket url :', url)
        return url

def on_message(ws, message):
    try:
        message =json.loads(message)
        code = message["code"]
        sid = message["sid"]
        audio = message["data"]["audio"]
        audio = base64.b64decode(audio)
        status = message["data"]["status"]
        print(message)
        if status == 2:
            print("ws is closed")
            ws.close()
        if code != 0:
            errMsg = message["message"]
            print("sid:%s call error:%s code is:%s" % (sid, errMsg, code))
        else:

            with open(pcm_path, 'ab') as f:
                f.write(audio)

    except Exception as e:
        print("receive msg,but parse exception:", e)


# 收到websocket错误的处理
def on_error(ws, error):
    print("### error:", error)


# 收到websocket关闭的处理
def on_close(ws, close_status_code, close_msg):
    print("### closed ###")


# 收到websocket连接建立的处理
def on_open(ws, wsParam):
    def run(*args):
        d = {"common": wsParam.CommonArgs,
             "business": wsParam.BusinessArgs,
             "data": wsParam.Data,
             }
        d = json.dumps(d)
        print("------>开始发送文本数据")
        ws.send(d)
        if os.path.exists(pcm_path):
            os.remove(pcm_path)

    thread.start_new_thread(run, ())


def play_pcm():
    import pyaudio

    # PCM 参数
    sample_rate = 16000  # 采样率
    channels = 1  # 单声道
    sample_width = 2  # 16-bit，每个采样点占 2 字节

    # 初始化 PyAudio
    p = pyaudio.PyAudio()

    # 打开音频流
    stream = p.open(format=p.get_format_from_width(sample_width),
                    channels=channels,
                    rate=sample_rate,
                    output=True)

    # 读取并播放 PCM 数据
    with open(pcm_path, "rb") as f:
        data = f.read()
        stream.write(data)

    # 关闭音频流
    stream.stop_stream()
    stream.close()
    p.terminate()


def pcm2wav():
    global wav_file
    # 将 PCM 文件转换为 WAV 文件
    import wave

    # PCM 参数
    sample_rate = 16000  # 采样率
    channels = 1  # 单声道
    sample_width = 2  # 16-bit，每个采样点占 2 字节

    # 读取 PCM 数据
    with open(pcm_path, "rb") as pcm:
        pcm_data = pcm.read()

    # 写入 WAV 文件    
    # 获取当前时间并格式化
    import pytz
    timezone = pytz.timezone("Asia/Shanghai")
    current_time = datetime.now(timezone)
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M-%S")
    # 生成文件名
    wav_file = f"file_{formatted_time}.wav"
    wav_path = os.path.join(out_path, wav_file)    
    
    with wave.open(wav_path, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_data)

    print("PCM 文件已转换为 WAV 文件：", wav_path)


from pathlib import Path

def delete_oldest_wav_files_recursive(directory, max_files=10):
    """
    递归检查目录及其子目录，如果 .wav 文件数量超过 max_files，则删除最旧的文件。
    
    :param directory: 要检查的目录路径
    :param max_files: 每个目录中允许的最大 .wav 文件数量
    """
    # 将目录路径转换为 Path 对象
    dir_path = Path(directory)

    # 递归遍历目录及其子目录
    for root, _, files in os.walk(dir_path):
        # 获取当前目录中的所有 .wav 文件及其修改时间
        wav_files = [(f, os.path.getmtime(os.path.join(root, f))) 
                     for f in files if f.endswith(".wav")]

        # 如果 .wav 文件数量超过最大值
        if len(wav_files) > max_files:
            print(f"Found {len(wav_files)} .wav files in {root} (exceeds {max_files}). Deleting the oldest files...")

            # 按修改时间排序（最旧的文件在前）
            wav_files.sort(key=lambda x: x[1])

            # 计算需要删除的文件数量
            files_to_delete = len(wav_files) - max_files

            # 删除最旧的文件
            for i in range(files_to_delete):
                file_path = os.path.join(root, wav_files[i][0])
                os.remove(file_path)
                print(f"Deleted {file_path}")


# @app.post("/tts/")
@app.post("/")
async def main(request: Request):
    data = await request.json()
    text = data.get("text", "")
    wsParam = Ws_Param(APPID=APPID, APISecret=APISecret, APIKey=APIKey, Text=text)
    websocket.enableTrace(False)
    wsUrl = wsParam.create_url()
    ws = websocket.WebSocketApp(wsUrl, on_message=on_message, on_error=on_error, on_close=on_close)
    # ws.on_open = on_open
    ws.on_open = partial(on_open, wsParam=wsParam)  # 使用 partial 传递 wsParam 参数
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
    if os.path.exists(pcm_path):
        pcm2wav()
        delete_oldest_wav_files_recursive(out_path, max_files=10)

        # static_url = "http://nginx-file-server/audio_file.wav"  # docker同一个网段可以直接访问容器名称
        # static_url = "http://host.docker.internal:8000/audio/audio_file.wav"  # 这是docker容器内访问的地址
        static_url = f"http://localhost:8000/audio/{wav_file}"  #dify服务可以直接访问主机url
        # return {"result": f"<audio controls> <source src='{static_url}' type='audio/wav'> 您的浏览器不支持音频播放</audio>"}
        return f"<audio controls> <source src='{static_url}' type='audio/wav'> 您的浏览器不支持音频播放</audio>"
    else:
        return {"error": "音频文件未生成"}

    # return {"result": "<audio controls> <source src='audio-file.mp3' type='audio/mpeg'> 出错就是不支持播放音乐</audio>"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)