import hashlib
import io
import base64
import os
from PIL import Image, ImageOps, ImageEnhance, ImageFile

# 解决大图读取报错问题
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageToolbox:
    @staticmethod
    def get_mime_type(file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
        }
        return mime_map.get(ext, "image/jpeg")

    @staticmethod
    def get_image_hash(image: Image.Image) -> str:
        """计算图片内容的MD5作为缓存Key"""
        img_byte_arr = io.BytesIO()
        # 统一转换为 RGB 和 PNG 格式计算 hash，保证同一张图结果一致
        img_copy = image.copy().convert('RGB')
        img_copy.save(img_byte_arr, format='PNG')
        return hashlib.md5(img_byte_arr.getvalue()).hexdigest()

    @staticmethod
    def pil_to_base64(img: Image.Image, format: str = "JPEG") -> str:
        """将 PIL Image 对象转换为 Base64 字符串"""
        buffered = io.BytesIO()
        # 如果是 RGBA (PNG)，转成 RGB 再存 JPEG
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @staticmethod
    def internvl_ocr_augment(img: Image.Image) -> Image.Image:
        """
        1. 分辨率增强 (针对小图双三次插值放大)
        2. 灰度融合 (保留 70% 彩色 + 30% 灰度，增强文字边缘)
        3. 对比度增强
        4. 锐化
        """
        img = img.convert("RGB")

        # 1. 分辨率增强
        w, h = img.size
        if min(w, h) < 28:
            img = img.resize((w * 2, h * 2), Image.BICUBIC)

        # 2. 灰度融合
        gray = ImageOps.grayscale(img)
        img = Image.blend(img, gray.convert("RGB"), alpha=0.3)

        # 3. 对比度
        img = ImageEnhance.Contrast(img).enhance(1.3)

        # 4. 锐化
        img = ImageEnhance.Sharpness(img).enhance(1.2)

        return img