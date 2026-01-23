import requests
import base64
import concurrent.futures
import time
import os
from pathlib import Path

def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def call_exam_agent(img_path, url):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            b64_data = get_image_base64(img_path)
            response = requests.post(url, json={"image_base64": b64_data}, timeout=600)
            
            # 只要不是 200，或者业务 code 不是 200，都视为失败
            if response.status_code == 200:
                res_json = response.json()
                if res_json.get('code') == 200:
                    return res_json.get('data')
            
            print(f"[RETRY] {os.path.basename(img_path)} (Attempt {attempt+1})")
            time.sleep(2)
        except Exception as e:
            if attempt == max_retries - 1:
                return f"Final Error: {str(e)}"
            time.sleep(2)
    return "Failed after retries"

def main():
    # --- 配置区 ---
    SERVER_URL = "http://10.120.1.3:8888/evaluate"
    IMG_DIR = "/mnt/afs_ocr/tongronglei/workspace/mathocr/2_eval/test_ocr/tmp"
    CONCURRENT_THREADS = 4
    
    EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    
    # 1. 扫描图片
    image_files = [
        str(f) for f in Path(IMG_DIR).iterdir() 
        if f.suffix.lower() in EXTENSIONS
    ]
    
    print(f"Loaded {len(image_files)} images. Starting pressure test (Threads: {CONCURRENT_THREADS})...")
    
    # 2. 并发执行并捕获返回
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_THREADS) as executor:
        # 提交任务，并建立 future 到 path 的映射
        future_to_img = {executor.submit(call_exam_agent, img, SERVER_URL): img for img in image_files}
        
        for future in concurrent.futures.as_completed(future_to_img):
            img_path = future_to_img[future]
            base_name = os.path.basename(img_path)
            
            # 这里是关键：通过 future.result() 获取 call_exam_agent 的 return 值
            try:
                agent_output = future.result()
                
                # 标准化输出格式
                print("\n" + "="*60)
                print(f"[IMAGE_NAME]: {base_name}")
                print("-" * 30)
                print(agent_output)  # 这里打印出你的 XML 结果
                print("-" * 30)
                print(f"[END_OF_RESULT]: {base_name}")
                print("="*60 + "\n")
                
            except Exception as e:
                print(f"Thread execution error for {base_name}: {e}")

if __name__ == "__main__":
    main()