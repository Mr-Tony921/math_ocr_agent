from PIL import Image

class CropTool:
    @staticmethod
    def crop_by_normalized_bbox(image: Image.Image, bbox: list):
        """
        根据归一化坐标 [x1, y1, x2, y2] 裁剪图片
        """
        w, h = image.size
        # 还原坐标
        left = int(bbox[0] * w / 1000)
        top = int(bbox[1] * h / 1000)
        right = int(bbox[2] * w / 1000)
        bottom = int(bbox[3] * h / 1000)
        
        # 增加少量 padding 避免切到字迹边缘
        padding = 5
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(w, right + padding)
        bottom = min(h, bottom + padding)
        
        return image.crop((left, top, right, bottom))