import uuid
import contextvars
import builtins
import time
from fastapi import FastAPI, Body, Request
from pydantic import BaseModel, validator

# 1. 强力日志追踪：劫持内置 print
request_id_ctx = contextvars.ContextVar("request_id", default="INIT")
_original_print = builtins.print

def scoped_print(*args, **kwargs):
    rid = request_id_ctx.get()
    # 增加毫秒级时间，方便观察 agent_core 内部耗时
    t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    _original_print(f"[{t}] [{rid}]", *args, **kwargs)

builtins.print = scoped_print

from agent_core import run_evaluation

app = FastAPI(title="exam_ocr_agent")

# 2. 中间件：生成并注入 ID
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = str(uuid.uuid4())[:8]
    token = request_id_ctx.set(rid)
    try:
        return await call_next(request)
    finally:
        request_id_ctx.reset(token)

class EvaluationRequest(BaseModel):
    image_base64: str
    # ... validator 保持不变 ...

# 3. 核心接口：去掉 async 是解决 TIMEOUT 的关键
@app.post("/evaluate")
def evaluate_api(request: EvaluationRequest = Body(...)):
    """
    注意：这里去掉了 async，FastAPI 会在线程池中运行此函数。
    即使 run_evaluation 跑 10 分钟，主线程依然能响应心跳。
    """
    try:
        print(f"--- New Request ---")
        res = run_evaluation(request.image_base64)
        print(f"--- Task Completed ---")
        return {"code": 200, "message": "success", "data": res}
    except Exception as e:
        print(f"Error: {e}")
        return {"code": 500, "message": str(e), "data": None}