import json
import requests
import re
from PIL import Image
from tools.image_utils import ImageToolbox
import config
from utils.prompts import OCR_SYSTEM_PROMPT, OCR_EXTRACT_PROMPT

class LocalOCRClient:
    def __init__(self):
        self.endpoint = config.LOCAL_OCR_URL
        self.template = {
            "system": "<|im_start|>system\n{}<|im_end|>\n",
            "user_start": "<|im_start|>user\n",
            "assistant_start": "<|im_start|>assistant\n",
            "im_end": "<|im_end|>\n",
            "img_tag": "<img></img>\n"
        }

    def _build_query(self, prompt_text: str, has_image: bool) -> str:
        """构建 Sensechat 格式的查询字符串"""
        query = self.template["system"].format(OCR_SYSTEM_PROMPT)
        query += self.template["user_start"]
        if has_image:
            query += self.template["img_tag"]
        query += prompt_text + self.template["im_end"]
        query += self.template["assistant_start"]
        return query

    def generate_content(self, pil_img: Image.Image, prompt: str = OCR_EXTRACT_PROMPT) -> str:
        """
        核心对外接口：输入 PIL 图片，输出 OCR 原始字符串结果
        """
        # 1. 准备图片数据
        img_b64 = ImageToolbox.pil_to_base64(pil_img)
        
        # 2. 构建输入文本
        query = self._build_query(prompt, has_image=True)
        
        # 3. 构造 Payload
        payload = {
            "inputs": query,
            "parameters": {
                "max_new_tokens": 4096,
                "do_sample": False,
                "temperature": 0.01,
                "top_p": 0.25,
                "top_k": 1,
                "repetition_penalty": 1,
                "add_special_tokens": False,
                "skip_special_tokens": True,
            },
            "multimodal_params": {
                "images": [{"type": "base64", "data": img_b64}]
            }
        }

        # 4. 发起请求
        try:
            response = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # 适配返回格式
            if isinstance(result, list):
                return result[0]["generated_text"]
            return result.get("generated_text", [""])[0]
            
        except Exception as e:
            print(f"❌ OCR API 请求失败: {str(e)}")
            return ""

    @staticmethod
    def parse_results(text: str) -> dict:
        """解析 XML 标签内容"""
        def _extract(tag):
            match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.S)
            return match.group(1).strip() if match else ""

        return {
            "question_text": _extract("st_question"),
            "question_id": _extract("st_question_id"),
            "question_type": _extract("st_question_type"),
            "answer_text": _extract("st_answer"),
            "final_answer": _extract("st_final_answer")
        }