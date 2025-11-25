import base64
import json
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import re
import time
import os

from prompts import vision_prompt, block_prompt
from internvl import process_sample
from doubao import generate_with_proxy
from mimetypes import guess_type

# Configuration for number of threads
NUM_THREADS = 40  # You can change this to control the number of threads

output_file = "./infer_result-doubao-merge-v1612-9.json"

def extract_tag(tag_name, xml_str):
    pattern = r'<{0}>(.*?)</{0}>'.format(tag_name)
    match = re.search(pattern, xml_str, re.DOTALL)
    return match.group(1).strip() if match else None

def replace_tag_content(text: str, tag: str, new_content: str) -> str:
    pattern = rf"(<{tag}>)(.*?)(</{tag}>)"

    def repl(match):
        return match.group(1) + new_content + match.group(3)

    return re.sub(pattern, repl, text, count=1, flags=re.DOTALL)


def local_image_to_data_url(image_path):
    """将本地图片文件转换为Base64编码的data URL。"""
    # 根据文件扩展名猜测MIME类型
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # 如果找不到，则使用默认值

    # 读取并编码图片文件
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # 构建data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def parse_model_output(text: str) -> str:
    """
    去除字符串中 <think>...</think> 部分，并对剩余内容 strip。
    """
    # DOTALL 让 . 可以匹配换行
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()

def process_item(content_json, output_file):
    current_data = content_json[0]

    block_image = current_data['image_path']
    block = process_sample(block_image, block_prompt)

    image_data_url = local_image_to_data_url(block_image)
    # merge = vision_prompt.format(full_result=block)
    # merge = vision_prompt.format(full_result=block, block_prompt=block_prompt)
    st_question = extract_tag("st_question", block)
    st_answer = extract_tag("st_answer", block)
    st_final_answer = extract_tag("st_final_answer", block)
    total_slots = 0
    answered_slots = 0
    if st_final_answer:
        parts = re.split(r"[;；]", st_final_answer)
        parts = [p.strip() for p in parts]
        total_slots = len(parts)
        answered_slots = sum(1 for p in parts if p != "")
    ocr_result = f"<st_question>{st_question}</st_question>\n<st_answer>{st_answer}</st_answer>\n<st_final_answer>{st_final_answer}</st_final_answer>"
    merge = vision_prompt.format(full_result=ocr_result, block_prompt=block_prompt, total_slots=total_slots, answered_slots=answered_slots)

    content = [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": merge}
    ]
    messages = [{"role": "user", "content": content}]
    success, llm_output = generate_with_proxy(messages)
    if success:
        tmp = parse_model_output(llm_output['data']['response_content']['choices'][0]['message']['content'])
        st_question = extract_tag("st_question", tmp)
        st_answer = extract_tag("st_answer", tmp)
        st_final_answer = extract_tag("st_final_answer", tmp)
        parsed_result = block
        if st_question:
            parsed_result = replace_tag_content(parsed_result, "st_question", st_question)
        if st_answer:
            parsed_result = replace_tag_content(parsed_result, "st_answer", st_answer)
        if st_final_answer:
            parsed_result = replace_tag_content(parsed_result, "st_final_answer", st_final_answer)
    else:
        parsed_result = ""

    ref = current_data.get('ref_answer', '')  # 安全获取参考答案
        
    # 构造结果JSON
    result_json = {
        "id": current_data['id'],
        "imgs": current_data['image_path'],
        "question": "",  # 不再使用原始问题
        "ref_answer": ref,
        "audio_key": "",
        "result": {
            "answer": parsed_result,  # 使用解析后的结果
            "comment": "",
            "system_prompt": "",
            "score": 0,
            "level": 3
        }
    }

    # 写入结果到输出文件
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump([result_json], f, ensure_ascii=False)
        f.write('\n')  # 添加换行符分隔JSON对象

    return [result_json]

if __name__ == "__main__":
    input_file = "/mnt/afs/tongronglei/code/judge_data/test_ocr/output.json"

    # 清空输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('')

    # 加载输入数据
    with open(input_file, 'r', encoding='utf-8') as f:
        json_lists = json.load(f)

    start_time = time.time()

    # 使用线程池处理数据
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(process_item, item,   output_file) for item in json_lists]
        for future in tqdm(futures, total=len(json_lists)):
            future.result()  # 等待任务完成

    end_time = time.time()
    print(f"Processing complete. Total time: {end_time - start_time:.2f} seconds")
