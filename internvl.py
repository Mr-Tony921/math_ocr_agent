import base64
import json
import requests
import os
import io
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from itertools import cycle
import time
from PIL import Image

# 配置常量
CONFIG = {
    "API_ENDPOINTS": {
        "test": ["http://10.119.16.170:8000/generate"]
    },
    "SYSTEM_PROMPTS": {
        "common": "你是商汤科技开发的日日新多模态大模型，英文名叫Sensechat，是一个有用无害的人工智能助手。",
        "think": "Reason step by step and place the thought process within the <think></think> tags, and provide the final conclusion at the end.",
    },
}

PROMPT_TEMPLATE = {
    "system": "<|im_start|>system\n{}<|im_end|>\n",
    "user_start": "<|im_start|>user\n",
    "assistant_start": "<|im_start|>assistant\n",
    "im_end": "<|im_end|>\n",
    "img_tag": "<img></img>\n",
    "audio_tag": "<audio></audio>\n"
}

def get_next_endpoint(api_version):
    """带版本感知的轮询器，维护每个版本的独立迭代器"""
    if not hasattr(get_next_endpoint, "_cycles"):
        get_next_endpoint._cycles = {}
    
    if api_version not in get_next_endpoint._cycles:
        endpoints = CONFIG["API_ENDPOINTS"].get(api_version, [])
        get_next_endpoint._cycles[api_version] = cycle(endpoints)
    
    return next(get_next_endpoint._cycles[api_version])

def handle_url(url, min_edge=28):
    """处理本地或远程文件URL：读取图像、短边<min_edge则放大、保持原格式输出base64"""
    if not url.startswith("file://"):
        raise ValueError("handle_url 目前只处理 file:// 路径")

    img_path = url[7:]
    img = Image.open(img_path)

    # 原始格式（如 'PNG', 'JPEG'）
    orig_format = img.format
    if orig_format is None:
        # 如果 PIL 识别不出来，就从扩展名猜
        ext = os.path.splitext(img_path)[1].lower().replace(".", "")
        # 常见格式映射
        mapping = {
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "png": "PNG",
            "webp": "WEBP",
            "bmp": "BMP"
        }
        orig_format = mapping.get(ext, "PNG")   # 默认 PNG

    w, h = img.size
    short_edge = min(w, h)

    # 等比放大
    if 0 < short_edge < min_edge:
        scale = min_edge / short_edge
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

    # 转 base64（保持原始格式）
    buffer = io.BytesIO()
    img.save(buffer, format=orig_format)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return img_base64

def api_request(version, messages, max_retries=3):
    """带重试机制的API请求"""
    # endpoint = CONFIG["API_ENDPOINTS"][version]
    query = ""
    images = []
    system_prompt = messages[0]["content"] if messages[0]["role"] == "system" else ""
    
    # 构建查询模板
    if system_prompt:
        query += PROMPT_TEMPLATE["system"].format(system_prompt)
    
    for message in messages[1 if system_prompt else 0:]:
        if message["role"] == "user":
            query += PROMPT_TEMPLATE["user_start"]
            if isinstance(message["content"], list):
                img_count = 0
                text_content = ""
                
                for content in message["content"]:
                    if content["type"] == "image_url":
                        query += PROMPT_TEMPLATE["img_tag"]
                        img_count += 1
                        images.append({
                            "type": "base64",
                            "data": handle_url(content["image_url"])
                        })
                    elif content["type"] == "text":
                        query += content["text"]
                
                # if img_count >= 2:
                #     query += f"用户本轮上传了{img_count}张图\n"
                query += PROMPT_TEMPLATE["im_end"]
            else:
                query += message["content"] + PROMPT_TEMPLATE["im_end"]
        elif message["role"] == "assistant":
            query += PROMPT_TEMPLATE["assistant_start"] + message["content"] + PROMPT_TEMPLATE["im_end"]
    
    query += PROMPT_TEMPLATE["assistant_start"]
    
    payload = {
        "inputs": query,
        "parameters": {
            "max_new_tokens": 1024,
            "do_sample": False,
            "temperature": 0.01,
            "top_p": 0.25,
            "top_k": 1,
            "repetition_penalty": 1,
            "add_special_tokens": False,
            "add_spaces_between_special_tokens": False,
            "skip_special_tokens": True,
            "image_patch_max_num": -1,
        }
    }
    
    multimodal_params = {}
    if images:
        multimodal_params["images"] = images
    if multimodal_params:
        payload["multimodal_params"] = multimodal_params
    
    endpoint = get_next_endpoint(version)

    for attempt in range(max_retries):
        try:
            
            response = requests.post(
                endpoint,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=1800
            )
            response.raise_for_status()
            result = response.json()
            return result[0]["generated_text"] if isinstance(result, list) else result["generated_text"][0]
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"API request failed after {max_retries} attempts: {str(e)}")
                return None
            continue

def process_sample(sample, question, api_version="test", system_prompt=CONFIG["SYSTEM_PROMPTS"]["common"]):
    """处理单个样本"""
    try:
        messages = [{"role": "system", "content": system_prompt}]
        img_path = "file://" + sample
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": img_path},
                {"type": "text", "text": question},
            ]
        })
        
        response = api_request(api_version, messages)
        if not response:
            return None
        
        return response
    except Exception as e:
        print(f"Error processing sample: {str(e)}")
        return None

def run_evaluation(api_version, system_prompt):
    """执行单个评估任务"""
    print(f"\nStarting evaluation - Model: {api_version}")
    
    # 准备处理队列
    images_path = "/mnt/afs/tongronglei/code/judge_data/test_ocr/images"
    samples_to_process = []
    for filename in os.listdir(images_path):
        full_path = os.path.abspath((os.path.join(images_path, filename)))
        if os.path.isfile(full_path):
            samples_to_process.append(full_path)
    
    # 执行并行处理
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {}
        with tqdm(total=len(samples_to_process)) as pbar:
            # 提交任务
            for sample in samples_to_process:
                future = executor.submit(
                    process_sample,
                    sample,
                    api_version,
                    system_prompt
                )
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time:.2f} seconds")


def _is_valid_result(result):
    """校验结果有效性"""
    if not result:
        return False
    # 检查所有轮次是否都有有效回答
    for turn in result:
        if not turn['result']['answer'] or turn['result']['answer'].strip() == "":
            return False
    return True


def main():
    """主执行函数"""
    # 遍历所有组合
    for api_version in CONFIG["API_ENDPOINTS"].keys():
        system_prompt = CONFIG["SYSTEM_PROMPTS"]["common"]  ## system_prompt改下
   
        try:
            run_evaluation(api_version, system_prompt)
        except Exception as e:
            print(f"Failed to evaluate {api_version}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
