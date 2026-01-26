import os

# 本地 OCR (Sensechat VLM 兼容) 配置
LOCAL_OCR_URL = "http://10.120.7.6:8000/generate"

# 本地 Grounding 服务配置
LOCAL_GROUNDING_URL = "http://10.120.2.61:8888/cut_question"

# Gemini API 配置
GEMINI_API_KEY = 'sk-nz9CAzZK3q9F7Ofk2FbL6bBnB9v80VyKoY7lHoczPJo0f7eo'
GEMINI_API_URL = 'https://api.ppchat.vip/v1beta/models/gemini-3-pro-preview:generateContent'
SITE_TOTAL_ID = 'd011e11efbb1a4bf9163830b0f22e8e7'

# 业务题型配置
TARGET_TYPES = ["选择题", "填空题"]

# --- 路径配置 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_DIR = os.path.join(BASE_DIR, "debug")
CACHE_FILE_PATH = os.path.join(DEBUG_DIR, "logic_cache.json")

# --- 确保 Debug 目录存在 ---
if not os.path.exists(DEBUG_DIR):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    print(f"[Config] Debug directory ensured at: {DEBUG_DIR}")