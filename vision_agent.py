import os
import re
import base64

from prompts import vision_prompt, block_prompt, crop_prompt
from internvl import process_sample
from doubao import generate_with_proxy

from mimetypes import guess_type

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

def extract_tag_content(text: str, tag: str) -> str:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def find_files_with_prefix(folder, prefix):
    """
    在 folder 目录中查找所有以 prefix 开头的文件（不递归），返回绝对路径 list。
    """
    result = []
    for name in os.listdir(folder):
        full_path = os.path.join(folder, name)
        if os.path.isfile(full_path) and name.startswith(prefix):
            result.append(full_path)

    # 提取数字并排序
    def extract_number(path):
        # 假设文件名格式是 prefix_数字.xxx
        base = os.path.basename(path)
        match = re.search(rf"{re.escape(prefix)}_(\d+)", base)
        return int(match.group(1)) if match else float('inf')

    result.sort(key=extract_number)
    return result

def main():
    os.environ["OPENAI_API_KEY"] = ""

    question_dir = "/mnt/afs/tongronglei/code/judge_data/test_ocr/ocr_agent/images/Q_8/"

    block_image = find_files_with_prefix(question_dir, "block")
    assert len(block_image) == 1, "Expected exactly one block image"

    block = process_sample(block_image[0], block_prompt)

    image_data_url = local_image_to_data_url(block_image[0])
    merge = vision_prompt.format(full_result=block)
    content = [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": merge}
    ]
    messages = [{"role": "user", "content": content}]
    success, llm_output = generate_with_proxy(messages, "doubao-1.5-thinking-vision-pro-250428")
    if success:
        print(llm_output['data']['response_content']['choices'][0]['message']['content'])
        print("-"*160)
        print(llm_output['data']['response_content']['choices'][0]['message']['reasoning_content'])

# def main():
#     os.environ["OPENAI_API_KEY"] = ""

#     parent_dir = "/mnt/afs/tongronglei/code/judge_data/test_ocr/ocr_agent/images/"

#     # 找到所有以 Q_ 开头的文件夹
#     q_folders = [
#         os.path.join(parent_dir, d)
#         for d in os.listdir(parent_dir)
#         if os.path.isdir(os.path.join(parent_dir, d)) and d.startswith("Q_")
#     ]

#     # 按数字大小排序
#     def extract_number(folder_name):
#         match = re.search(r"Q_(\d+)", folder_name)
#         return int(match.group(1)) if match else 0

#     q_folders.sort(key=lambda x: extract_number(os.path.basename(x)))

#     # 遍历每个 Q_ 文件夹
#     for question_dir in q_folders:
#         print(f"Processing folder: {question_dir}")

#         # 找到 block 开头的图片
#         block_images = find_files_with_prefix(question_dir, "block")
#         if len(block_images) != 1:
#             print(f"Skipping {question_dir}, expected exactly one block image, found {len(block_images)}")
#             continue

#         block = process_sample(block_images[0], block_prompt)

#         # 转成 data_url
#         image_data_url = local_image_to_data_url(block_images[0])

#         # 构建 prompt
#         merge = vision_prompt.format(full_result=block)
#         content = [
#             {"type": "image_url", "image_url": {"url": image_data_url}},
#             {"type": "text", "text": merge}
#         ]
#         messages = [{"role": "user", "content": content}]

#         # 调用 LLM
#         success, llm_output = generate_with_proxy(messages, "doubao-1.5-thinking-vision-pro-250428")
#         if success:
#             print(llm_output['data']['response_content']['choices'][0]['message']['content'])
#             print("-"*160)
#             # print(llm_output['data']['response_content']['choices'][0]['message']['reasoning_content'])

if __name__ == "__main__":
    main()