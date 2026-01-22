import requests
import json
from typing import List, Optional, Tuple
from PIL import Image
import config
from tools.image_utils import ImageToolbox

class GeminiClient:
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY
        self.api_url = config.GEMINI_API_URL
        self.site_id = config.SITE_TOTAL_ID
        self.headers = {
            'x-goog-api-key': self.api_key,
            'Content-Type': 'application/json',
            'Cookie': f'SITE_TOTAL_ID={self.site_id}'
        }

    def generate_content(
        self, 
        prompt_text: str, 
        images: Optional[List[Image.Image]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        向 Gemini API 发送多模态请求。
        
        参数:
            prompt_text (str): 文本提示词。
            images (List[Image.Image], optional): PIL 图片对象列表，支持多图。
        
        返回:
            Tuple[bool, Optional[str]]: (是否成功, 模型回复或错误信息)
        """
        
        # 1. 构造基础 parts (文本部分)
        parts = [{"text": prompt_text}]

        # 2. 如果有图片，将所有图片转为 Base64 并加入 parts
        if images:
            for img in images:
                try:
                    # 复用 image_utils 中的转换工具
                    encoded_image = ImageToolbox.pil_to_base64(img)
                    
                    # 将图片加入 parts 列表前面 (Gemini 习惯图片在前)
                    parts.insert(0, {
                        "inlineData": {
                            "data": encoded_image,
                            "mimeType": "image/jpeg"
                        }
                    })
                except Exception as e:
                    return False, f"错误：处理图片时发生错误: {e}"

        # 3. 构造请求体
        request_data = {
            "contents": [{
                "parts": parts
            }]
        }

        # 4. 发送 API 请求
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(request_data),
                timeout=180 
            )

            if response.status_code == 200:
                response_json = response.json()
                try:
                    # 按照您提供的逻辑解析响应
                    model_response_text = response_json['candidates'][0]['content']['parts'][0]['text']
                    return True, model_response_text
                except (IndexError, KeyError):
                    return False, f"错误：解析失败。响应内容: {response.text}"
            else:
                return False, f"API 失败 ({response.status_code}): {response.text}"

        except requests.exceptions.RequestException as e:
            return False, f"网络请求异常: {e}"