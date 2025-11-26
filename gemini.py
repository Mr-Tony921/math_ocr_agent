import requests
import json
import base64
import os
from typing import Tuple, Optional

# --- API 配置 ---
# 请根据实际情况填写你的密钥和 URL
API_KEY = 'sk-'  # 替换为你的 API 密钥
SITE_TOTAL_ID = 'd011e11efbb1a4bf9163830b0f22e8e7'
API_URL = 'https://api.ppchat.vip/v1beta/models/gemini-3-pro-preview:generateContent'
# ------------------

def generate_content_with_image(
    image_path: str, 
    prompt_text: str
) -> Tuple[bool, Optional[str]]:
    """
    向 API 发送包含图片和文本的多模态请求。

    参数:
        image_path (str): 本地图片文件的完整路径。
        prompt_text (str): 用户的文本提示。

    返回:
        Tuple[bool, Optional[str]]: (是否成功, 模型回复文本或错误信息)。
    """
    
    # 1. 检查图片文件是否存在
    if not os.path.exists(image_path):
        return False, f"错误：未找到文件路径 '{image_path}'"

    # 2. 读取文件并进行 Base64 编码
    try:
        with open(image_path, "rb") as image_file:
            # 以 Base64 字符串形式对二进制数据进行编码
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        
        # 简单地从文件名推断 MIME 类型（实际应用中可能需要更精确的MIME类型检测）
        mime_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        
        file_ext = os.path.splitext(image_path)[1].lower()
        mime_type = mime_type_map.get(file_ext, "image/jpeg") # 默认为 image/jpeg
        
    except Exception as e:
        return False, f"错误：读取或编码图片时发生错误: {e}"

    # 3. 构造请求体 (Request Payload)
    headers = {
        'x-goog-api-key': API_KEY,
        'Content-Type': 'application/json',
        'Cookie': f'SITE_TOTAL_ID={SITE_TOTAL_ID}'
    }

    # API 请求体中，图片和文本作为不同的 'parts' 放在 'contents' 列表中
    request_data = {
        "contents": [{
            "parts": [
                {
                    "inlineData": {
                        "data": encoded_image,
                        "mimeType": mime_type
                    }
                },
                {
                    "text": prompt_text
                }
            ]
        }]
    }

    # 4. 发送 API 请求
    try:
        response = requests.post(
            API_URL, 
            headers=headers, 
            data=json.dumps(request_data),
            timeout=1800 # 设置超时时间
        )

        # 5. 检查响应状态和解析结果
        if response.status_code == 200:
            response_json = response.json()
            
            # 提取回复文本 (与你之前的功能 2 逻辑相同)
            try:
                # 路径是: candidates[0] -> content -> parts[0] -> text
                model_response_text = response_json['candidates'][0]['content']['parts'][0]['text']
                return True, model_response_text
            except (IndexError, KeyError):
                # 如果成功的响应中解析不到文本，则返回解析错误
                return False, f"错误：成功调用但无法解析回复文本。原始响应: {response.text}"
        else:
            # API 调用失败
            return False, f"API 调用失败，状态码: {response.status_code}. 错误信息: {response.text}"

    except requests.exceptions.RequestException as e:
        return False, f"请求发生错误: {e}"

# --- 示例使用 ---
if __name__ == '__main__':
    # **请替换为你的本地图片文件路径**
    # 为了测试，请确保你有一个名为 'example_image.jpg' 的图片文件
    # 或者将路径替换为你的实际文件路径
    
    # 确保有一个图片文件用于测试
    test_image_path = "/mnt/afs/tongronglei/code/judge_data/test_ocr/images/11.png"
    test_prompt = "OCR this image。"
    
    # 为了让代码能运行，我们在这里假设文件存在，并提醒用户替换
    if not os.path.exists(test_image_path):
        print(f"**注意：测试文件 '{test_image_path}' 不存在。请替换为你的实际图片路径进行测试。**")
        print("请在运行前创建一个名为 'example_image.jpg' 的测试文件，或者更改代码中的 `test_image_path` 变量。")
    
    # 假设测试文件存在，开始调用
    print(f"正在向 API 发送请求，文件路径: {test_image_path}")
    
    # 调用函数
    success, result = generate_content_with_image(test_image_path, test_prompt)

    # 输出结果
    print("\n--- API 调用结果 ---")
    print(f"调用是否成功 (bool): {success}")
    if success:
        print(f"模型回复文本 (str):\n{result}")
    else:
        print(f"错误或失败信息:\n{result}")