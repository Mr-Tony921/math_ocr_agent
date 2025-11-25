import base64
import requests

from zai import ZhipuAiClient

def generate_with_proxy(image_path, text_prompt, model_name="glm-4.5v"):
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    client = ZhipuAiClient(api_key="")  # 填写您自己的APIKey
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ],
        thinking={
            "type": "enabled"
        }
    )

    result = response.choices[0].message.content
    return result