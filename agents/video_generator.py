import http.client
import json
import base64
import time
import requests
import os

def encode_base64(file_path):
    with open(file_path, 'rb') as image_file:
        image_data = image_file.read()
        base64_encoded_data = base64.b64encode(image_data)
        base64_string = base64_encoded_data.decode('utf-8')
        return "data:image/png;base64," + base64_string

def download_video(url, save_path='./video.mp4'):
    """
    从URL下载视频文件
    
    参数:
    url (str): 视频直链URL
    save_path (str): 本地保存路径（默认当前目录video.mp4）
    """
    try:
        # 设置请求头模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 发起请求并开启流传输
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()  # 检查请求是否成功
        
        # 获取文件大小（字节）
        total_size = int(response.headers.get('content-length', 0))
        
        # 写入文件
        with open(save_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                # 显示进度条（可选）
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\r下载进度: {progress:.2f}%", end='')
        print("\n下载完成！保存至:", os.path.abspath(save_path))
    
    except Exception as e:
        print("下载失败:", str(e))

class VideoGenerator:
    def __init__(
        self,
        base_url: str = "https://yunwu.ai/v1",
        api_key: str = "sk-bapMAyiji5uA2O91yTv6iBIXmp3e0JsMWByxgzHkzOd9jRIF",
    ):
        self.base_url = base_url
        self.api_key = api_key

    def generate_video(
        self,
        prompt: str = "",
        image_paths: list = [],
        save_path: str = None,
    ):
        conn = http.client.HTTPSConnection("yunwu.ai")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        images = []
        for image_path in image_paths:
            if image_path:
                images.append(encode_base64(image_path))
        payload = json.dumps({
            "prompt": prompt,
            "model": "veo2-fast-frames" if len(images) == 2 else "veo3-fast-frames",
            "images": images,
            "enhance_prompt": True
        })

        conn.request("POST", "/v1/video/create", payload, headers)
        res = conn.getresponse()

        response_data = json.loads(res.read().decode("utf-8"))
        print(response_data)
        task_id = response_data["id"]
        boundary = ''
        payload = ''
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'Content-type': 'multipart/form-data; boundary={}'.format(boundary)
        }


        while True:
            conn.request("GET", f"/v1/video/query?id={task_id}", payload, headers)
            res = conn.getresponse()
            data = json.loads(res.read().decode("utf-8"))
            if data["status"] == "completed":
                print(data)
                video_url = data["video_url"]
                download_video(video_url, save_path)
                break
            elif  data["status"] == "failed":
                print("视频生成失败:")
                break
            else:
                print(data)
                print("视频生成中，请稍候...")
                time.sleep(5)
                continue


# download_video("https://filesystem.site/cdn/20250716/rL4aPvlQ16AdW6eJSn5Mbdhaq3o3gW.mp4", ".working_dir/videos/0.mp4")


# generator = VideoGenerator()
# generator.generate_video(
#     prompt=r"夜幕下的华北平原，战地记者举着自拍杆出现在镜头中央，表情兴奋而紧张，眼神坚定地看着镜头。背景中可见远处火光冲天，正太铁路沿线多处爆炸的火焰照亮夜空，八路军战士们的身影在黑暗中快速移动，手持工具和炸药包冲向铁路线。记者身后可见铁轨被撬起的画面，枕木堆积如山准备销毁。",
#     image_paths=[
#         "1.png"
#     ],
#     save_path="video1.mp4"
# )