import base64
import requests

# vLLM 服务器地址
API_URL = "http://10.119.24.58:8000/v1/chat/completions"

# # 读取本地图片（示例：./test.jpg）
# image_path = "/mnt/afs/tongronglei/code/judge_data/test_ocr/images/11.png"
# with open(image_path, "rb") as f:
#     img_b64 = base64.b64encode(f.read()).decode()

# payload = {
#     "model": "Qwen3-VL-235B-A22B-Thinking",
#     "messages": [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "text", "text": "OCR this image."},
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{img_b64}"
#                     }
#                 }
#             ]
#         }
#     ]
# }

# response = requests.post(API_URL, json=payload)
# print(response.json()["choices"][0]["message"]["content"].split("</think>")[-1].strip())

def generate_with_proxy(image_path, text_prompt, model_name="Qwen3-VL-235B-A22B-Thinking"):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ]
            }
        ]
    }
    
    response = requests.post(API_URL, json=payload)

    return response.json()["choices"][0]["message"]["content"].split("</think>")[-1].strip()