import requests
import json
import base64
from io import BytesIO
from typing import List, Optional, Tuple
from PIL import Image

# 假设这是你的本地配置模块
import config 
# 假设这是你的工具模块
from tools.image_utils import ImageToolbox

class GeminiClientV2:
    def __init__(self):
        self.api_key = config.GEMINI_API_KEY_V2
        # 如果 config 中没有定义 MODEL_NAME，默认使用 curl 中的模型
        self.model_name = getattr(config, 'GEMINI_MODEL', 'gemini-3-pro-preview')
        
        # 构造 API URL (注意：这里使用 generateContent 以获得一次性响应，而非流式)
        # 原始 curl 为: .../models/gemini-3-pro-preview:streamGenerateContent
        # 这里的 base URL 对应 aiplatform 或 generativelanguage，取决于你的 API Key 类型
        # 按照你的 curl 示例，使用的是 aiplatform 格式
        self.base_url = "https://aiplatform.googleapis.com/v1/publishers/google/models"
        self.api_url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"

        # 配置代理
        # 从 config 读取 PROXY_URL，如果 config 里没写，可以使用默认值
        proxy_url = getattr(config, 'PROXY_URL', '')
        self.proxies = {
            "http": proxy_url,
            "https": proxy_url
        }

        self.headers = {
            'Content-Type': 'application/json'
        }

    def generate_content(
        self, 
        prompt_text: str, 
        images: Optional[List[Image.Image]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        向 Gemini API (V2) 发送多模态请求。
        支持 VPN 代理，接口签名与 V1 保持一致。
        """
        
        # 1. 构造 parts
        parts = [{"text": prompt_text}]

        # 2. 处理图片
        if images:
            # Gemini 对图片的顺序敏感，通常建议图片放在文本之前或根据语境
            # 这里按照原逻辑，将图片插入到 parts 最前面
            for img in reversed(images): # 使用 reversed 保证插入后顺序正确
                try:
                    # 使用 ImageToolbox 或本地逻辑转 base64
                    encoded_image = ImageToolbox.pil_to_base64(img)
                    
                    parts.insert(0, {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": encoded_image
                        }
                    })
                except Exception as e:
                    return False, f"错误：处理图片时发生错误: {e}"

        # 3. 构造请求体 (结构匹配 curl 中的 contents -> role/parts)
        request_data = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts
                }
            ],
            # 可以添加生成配置，例如 token 限制或 temperature
            # "generationConfig": {
            #     "temperature": 0.7,
            #     "maxOutputTokens": 2048
            # }
        }

        # 4. 发送请求 (带代理)
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=json.dumps(request_data),
                proxies=self.proxies, # 关键：添加代理
                timeout=1800
            )

            if response.status_code == 200:
                response_json = response.json()
                try:
                    # 解析响应
                    # 路径通常为 candidates[0].content.parts[0].text
                    candidate = response_json['candidates'][0]
                    
                    # 检查是否有 finishReason 为 SAFETY 或其他拦截
                    if candidate.get('finishReason') not in ['STOP', None]:
                        # 这是一个警告，但我们仍然尝试获取文本
                        pass

                    model_response_text = candidate['content']['parts'][0]['text']
                    return True, model_response_text
                except (IndexError, KeyError, TypeError) as e:
                    return False, f"错误：解析响应失败。JSON结构: {str(e)}。原始内容: {response.text[:200]}"
            else:
                return False, f"API 失败 ({response.status_code}): {response.text}"

        except requests.exceptions.ProxyError:
            return False, "错误：代理连接失败，请检查 VPN 设置。"
        except requests.exceptions.RequestException as e:
            return False, f"网络请求异常: {e}"

# --- 测试代码 ---

def test_gemini_v2():
    """
    测试函数，用于验证 GeminiClientV2
    """
    print("--- 开始测试 GeminiClientV2 ---")
    
    # 1. 初始化客户端
    client = GeminiClientV2()
    print(f"初始化完成，使用代理: {client.proxies['http']}")

    # 2. 测试纯文本
    print("\n[测试 1] 纯文本请求...")
    prompt = "Explain how AI works in a few words"
    success, result = client.generate_content(prompt)
    
    if success:
        print(f"✅ 成功:\n{result}")
    else:
        print(f"❌ 失败: {result}")

    # 3. 测试图片 (可选)
    # 请修改下面的 image_path 为你本地存在的图片路径
    image_path = "/mnt/afs_ocr/tongronglei/workspace/mathocr/2_eval/test_ocr/tmp/11-522417e5-78e1-43c7-b972-49a3d607e008.jpeg" 
    
    import os
    if os.path.exists(image_path):
        print(f"\n[测试 2] 图片+文本请求 (路径: {image_path})...")
        try:
            img = Image.open(image_path)
            prompt_img = "OCR this image."
            success_img, result_img = client.generate_content(prompt_img, images=[img])
            
            if success_img:
                print(f"✅ 成功:\n{result_img}")
            else:
                print(f"❌ 失败: {result_img}")
        except Exception as e:
            print(f"加载测试图片失败: {e}")
    else:
        print(f"\n[跳过测试 2] 未找到测试图片: {image_path}")

if __name__ == "__main__":
    # 为了让代码直接运行，这里临时 mock config 和 ImageToolbox (如果你没有这两个文件)
    # 如果你有真实环境，请注释掉下面这几行 Mock 代码
    
    # --- Mock Start ---
    class MockConfig:
        GEMINI_API_KEY = "YOUR_REAL_API_KEY_HERE" # ⚠️ 请在此填入真实 Key 测试，或确保 config.py 存在
        PROXY_URL = ""
    
    if not hasattr(config, 'GEMINI_API_KEY_V2'):
        config = MockConfig()
        
    if not hasattr(ImageToolbox, 'pil_to_base64'):
        # 简单的 patch，防止报错
        class ImageToolbox:
            @staticmethod
            def pil_to_base64(img):
                buffered = BytesIO()
                img.convert('RGB').save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
    # --- Mock End ---

    test_gemini_v2()