import logging
import requests
import base64
import mimetypes
from tenacity import retry


@retry
def download_image(url, save_path):
    try:
        logging.info(f"Downloading image from {url} to {save_path}")

        response = requests.get(url, stream=True)
        response.raise_for_status()  # 检查请求是否成功

        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                file.write(chunk)

    except Exception as e:
        logging.error(f"Error downloading image: {e}")
        raise e


def image_to_base64(image_path, mime: bool = True) -> str:
    with open(image_path, 'rb') as image_file:
        b64 = base64.b64encode(image_file.read()).decode('utf-8')

    if mime:
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None:
            mime_type = 'application/octet-stream'
        return f"data:{mime_type};base64,{b64}"

    return b64

