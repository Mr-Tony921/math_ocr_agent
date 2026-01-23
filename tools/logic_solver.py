import os
import json
import hashlib
import threading
import io
import re
import time
from PIL import Image
import config

# 导入提示词
from utils.prompts import (
    CHOICE_SOLVE_PROMPT,
    COMPLETION_SOLVE_PROMPT
)

class LogicSolver:
    _file_lock = threading.Lock()  # 全局文件锁，防止多线程同时写文件冲突

    def __init__(self, gemini_client):
        """
        :param gemini_client: 传入线程独立的 GeminiClient 实例
        """
        self.client = gemini_client
        self.cache_file = config.CACHE_FILE_PATH
        self.cache_data = self._load_cache()

    def _load_cache(self):
        """加载本地缓存"""
        if not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[LogicSolver] ⚠️ 缓存文件损坏或读取失败: {e}")
            return {}

    def _save_cache(self):
        """持久化缓存 (线程安全)"""
        with self._file_lock:
            try:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[LogicSolver] ⚠️ 写入缓存失败: {e}")

    def _compute_hash(self, pil_image, question_text):
        """计算唯一指纹: MD5(Image Bytes + Question Text)"""
        img_byte_arr = io.BytesIO()
        # 统一转为 JPEG 计算 hash，避免格式差异
        pil_image.save(img_byte_arr, format='JPEG')
        img_bytes = img_byte_arr.getvalue()
        
        combined = img_bytes + question_text.strip().encode('utf-8')
        return hashlib.md5(combined).hexdigest()

    def _update_memory_cache(self, key, value):
        """更新内存并触发写盘"""
        with self._file_lock:
            self.cache_data[key] = value
        self._save_cache()

    def _generate_with_retry(self, prompt, image, cache_key, max_retries=3):
        """
        核心通用方法：
        1. 查缓存
        2. 带重试的 API 调用
        3. 强制检查 <answer> 标签
        """
        # 1. 优先查缓存
        if cache_key in self.cache_data:
            print(f"[LogicSolver] 🔥 命中缓存: {self.cache_data[cache_key]}")
            return self.cache_data[cache_key]

        print(f"[LogicSolver] 缓存未命中，调用 Gemini (Max {max_retries} Retries)...")
        
        for attempt in range(max_retries):
            try:
                success, raw_response = self.client.generate_content(prompt, [image])
                
                # 情况 A: API 调用本身失败 (网络或鉴权问题)
                if not success:
                    print(f"[LogicSolver] Retry {attempt+1}/{max_retries}: API 调用失败 - {raw_response}")
                    time.sleep(1) # 简单避退
                    continue

                # 情况 B: API 成功，但正则没匹配到 (格式错误)
                match = re.search(r'<answer>(.*?)</answer>', raw_response, re.IGNORECASE | re.DOTALL)
                if match:
                    final_res = match.group(1).strip()
                    # 成功获取，写入缓存并返回
                    self._update_memory_cache(cache_key, final_res)
                    return final_res
                else:
                    # 获取了内容但没有 tag，视为失败，触发重试
                    print(f"[LogicSolver] Retry {attempt+1}/{max_retries}: 响应缺少 <answer> 标签. Raw len: {len(raw_response)}")
            
            except Exception as e:
                print(f"[LogicSolver] Retry {attempt+1}/{max_retries}: 发生未知异常: {e}")
                time.sleep(1)

        print(f"[LogicSolver] ❌ 达到最大重试次数 ({max_retries})，解题失败。")
        return ""

    def solve_choice(self, enhanced_img, question_text):
        """
        选择题解题入口
        """
        cache_key = self._compute_hash(enhanced_img, question_text)
        solve_prompt = CHOICE_SOLVE_PROMPT.format(question_text=question_text)
        
        # 调用通用重试逻辑
        return self._generate_with_retry(solve_prompt, enhanced_img, cache_key)

    def solve_completion(self, enhanced_img, question_text):
        """
        填空题解题入口
        """
        cache_key = self._compute_hash(enhanced_img, question_text)
        solve_prompt = COMPLETION_SOLVE_PROMPT.format(question_text=question_text)
        
        # 调用通用重试逻辑
        return self._generate_with_retry(solve_prompt, enhanced_img, cache_key)