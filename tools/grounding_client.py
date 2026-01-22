import time
import requests
from collections import OrderedDict
from PIL import Image
import config
from tools.image_utils import ImageToolbox

CACHE_MAX_SIZE = 100

GROUNDING_TYPE_MAP = {
    "figure": "graph",
    "table": "sheet",
    "answer": "answering_area",
    "stem": "question_stem",
    "blank": "answer_region"
}

class GroundingClient:
    def __init__(self):
        self.url = config.LOCAL_GROUNDING_URL
        self.cache = OrderedDict()  # LRU Cache
        self.max_cache_size = CACHE_MAX_SIZE

    def _normalize_box(self, x: int, y: int, w: int, h: int, img_w: int, img_h: int):
        """将绝对坐标(x, y, w, h)转换为归一化坐标[x1, y1, x2, y2] (0-1000)"""
        if img_w == 0 or img_h == 0:
            return [0, 0, 0, 0]
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        return [
            int(max(0, min(1000, (x1 / img_w) * 1000))),
            int(max(0, min(1000, (y1 / img_h) * 1000))),
            int(max(0, min(1000, (x2 / img_w) * 1000))),
            int(max(0, min(1000, (y2 / img_h) * 1000)))
        ]

    def get_bboxes(self, image: Image.Image, target_type: str):
        """
        获取指定类型的坐标列表
        :param target_type: 'table','figure','answer','blank', 'stem'
        """
        if target_type not in GROUNDING_TYPE_MAP:
            print(f"⚠️  不支持的类型: {target_type}")
            return []
        
        api_key = GROUNDING_TYPE_MAP[target_type]
        img_hash = ImageToolbox.get_image_hash(image)

        # 1. 检查缓存
        if img_hash in self.cache:
            self.cache.move_to_end(img_hash)
            api_result = self.cache[img_hash]
        else:
            # 2. 请求服务 (带 3 次重试)
            api_result = self._request_service(image, img_hash)
        
        if not api_result or api_key not in api_result:
            return []

        # 3. 解析结果
        normalized_bboxes = []
        img_w, img_h = image.size
        for item in api_result[api_key]:
            # 处理嵌套结构 (graphloc / sheetloc)
            box_data = item
            if "graphloc" in item: box_data = item["graphloc"]
            elif "sheetloc" in item: box_data = item["sheetloc"]
            
            x, y = box_data.get("x", 0), box_data.get("y", 0)
            w, h = box_data.get("width", 0), box_data.get("height", 0)
            
            bbox = self._normalize_box(x, y, w, h, img_w, img_h)
            normalized_bboxes.append(bbox)
        
        return normalized_bboxes

    def _request_service(self, image: Image.Image, img_hash: str):
        """内部请求逻辑"""
        image_base64 = ImageToolbox.pil_to_base64(image)
        payload = {"image_base64": image_base64}
        
        for attempt in range(3):
            try:
                resp = requests.post(self.url, json=payload, timeout=30)
                resp.raise_for_status()
                resp_json = resp.json()
                
                if resp_json.get("code") == 0:
                    result = resp_json.get("result", {})
                    # 更新缓存
                    self.cache[img_hash] = result
                    if len(self.cache) > self.max_cache_size:
                        self.cache.popitem(last=False)
                    return result
                else:
                    print(f"❌ 服务错误: {resp_json.get('msg')}")
            except Exception as e:
                print(f"🔄 重试 Grounding ({attempt+1}/3): {e}")
                time.sleep(2)
        return None