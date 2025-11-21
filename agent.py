import os
import re

from prompts import merge_prompt, block_prompt, crop_prompt
from internvl import process_sample
from doubao import generate_with_proxy

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
    os.environ["OPENAI_API_KEY"] = "5b2cec881cccf499887845abc06d3c17"

    question_dir = "/mnt/afs/tongronglei/code/judge_data/test_ocr/ocr_agent/images/Q_15/"

    block_image = find_files_with_prefix(question_dir, "block")
    assert len(block_image) == 1, "Expected exactly one block image"

    crop_images = find_files_with_prefix(question_dir, "crop")
    assert len(crop_images) >= 1, "Expected at least one crop image or use block result only"

    block = process_sample(block_image[0], block_prompt)
    print(block)
    print("-"*160)

    crops = []
    for crop_image in crop_images:
        crop_single = extract_tag_content(process_sample(crop_image, crop_prompt), "st_handwritten")
        if crop_single.endswith("。"):
            crop_single = crop_single[:-1]
        crops.append(crop_single)
    crop = ";".join(crops)
    print(crop)
    print("-"*160)

    merge = merge_prompt.format(num_crop=len(crops), full_result=block, crop_results=crop)
    messages = [{"role": "user", "content": merge}]
    success, llm_output = generate_with_proxy(messages)
    if success:
        print(llm_output['data']['response_content']['choices'][0]['message']['content'])
        print("-"*160)
        # print(llm_output['data']['response_content']['choices'][0]['message']['reasoning_content'])


if __name__ == "__main__":
    main()