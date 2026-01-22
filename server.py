import base64
import binascii
from fastapi import FastAPI, Body
from pydantic import BaseModel, validator
import uvicorn

# 从 agent_core.py 引入 run_evaluation
from agent_core import run_evaluation

app = FastAPI(title="exam_ocr_agent")

class EvaluationRequest(BaseModel):
    image_base64: str

    @validator("image_base64")
    def check_base64_format(cls, v):
        """校验 Base64 字符串的合法性"""
        if not v:
            raise ValueError("Base64 string cannot be empty")
        
        # 如果带有 Data URI 前缀 (如 data:image/jpeg;base64,)，先去掉
        if "," in v:
            v = v.split(",")[1]
            
        try:
            # 尝试解码前几个字节，检查是否为合法的 Base64 编码
            base64.b64decode(v[:32], validate=True)
        except binascii.Error:
            raise ValueError("Invalid Base64 encoding")
        return v

@app.post("/evaluate")
async def evaluate_api(request: EvaluationRequest = Body(...)):
    try:
        # 直接透传验证后的 base64 字符串给核心函数
        # 注意：这里 request.image_base64 已经是经过 validator 处理过的纯字符串
        raw_xml_res = run_evaluation(request.image_base64)
        
        if not raw_xml_res:
            return {
                "code": 500, 
                "message": "Evaluation engine failed to process the image", 
                "data": None
            }

        return {
            "code": 200,
            "message": "success",
            "data": raw_xml_res
        }

    except Exception as e:
        # 捕获运行时异常，返回 500
        return {
            "code": 500, 
            "message": f"Core Logic Error: {str(e)}", 
            "data": None
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)