import io
import os
import re
import time
import base64
from PIL import Image

# --- 配置变量 ---
MAX_RETRIES = 3

# --- 导入工具 ---
from tools.crop_utils import CropTool
from tools.image_utils import ImageToolbox
from tools.ocr_client import LocalOCRClient
from tools.grounding_client import GroundingClient
# from tools.gemini_client import GeminiClient
from tools.gemini_client_v2 import GeminiClientV2 as GeminiClient
from tools.logic_solver import LogicSolver

# --- 导入 Prompt ---
from utils.prompts import (
    CHOICE_VISUAL_VERIFY_PROMPT,
    COMPLETION_STRUCTURE_CHECK_PROMPT,
    COMPLETION_VISUAL_VERIFY_PROMPT,
    COMPLETION_VISUAL_BATCH_VERIFY_PROMPT
)
import config

# ==========================================
# 辅助函数
# ==========================================

def _safe_split_answers(text):
    """初步分割 OCR 结果"""
    if not text: return []
    normalized = text.replace('；', ';')
    return [x.strip() for x in normalized.split(';') if x.strip()]

def _safe_join_answers(parts):
    """组装最终输出字符串"""
    return ";".join([p if p else "" for p in parts])

def _gemini_generate_with_retry(gemini_service, prompt, visual_materials, mode="tag", retry_count=MAX_RETRIES):
    """
    通用 Gemini 生成助手，支持重试和格式校验
    mode: 
      - "tag": 检查是否存在 <answer> 标签
      - "yes_no": 检查是否存在 YES 或 NO
      - "batch_yes_no": 批量模式，通常只需判断 API 是否成功返回内容
    """
    for attempt in range(retry_count):
        try:
            success, raw_res = gemini_service.generate_content(prompt, visual_materials)
            if not success:
                print(f"[Retry] API 请求失败，正在进行第 {attempt+1} 次重试...")
                continue
            
            # 格式校验
            if mode == "tag":
                if re.search(r'<answer>(.*?)</answer>', raw_res, re.IGNORECASE | re.DOTALL):
                    return True, raw_res
            elif mode == "yes_no":
                upper_res = raw_res.upper()
                if "YES" in upper_res or "NO" in upper_res:
                    return True, raw_res
            elif mode == "batch_yes_no":
                # 批量模式只要有返回内容且包含索引特征即可（如 "1:"）
                if raw_res.strip():
                    return True, raw_res
            
            print(f"[Retry] 响应格式不符，正在进行第 {attempt+1} 次重试...内容预览: {raw_res[:50]}")
        except Exception as e:
            print(f"[Retry] 发生异常: {e}")
        
        time.sleep(0.5) # 短暂规避
    
    return False, ""

# ==========================================
# 核心步骤逻辑
# ==========================================

def run_ocr(base64_image, ocr_service):
    """第一步：OCR 识别"""
    img_data = base64.b64decode(base64_image)
    raw_img = Image.open(io.BytesIO(img_data))
    enhanced_img = ImageToolbox.internvl_ocr_augment(raw_img)
    
    print(f"[OCR] 发送请求至本地节点...")
    raw_text = ocr_service.generate_content(enhanced_img)
    parsed = ocr_service.parse_results(raw_text)
    
    print(f"[OCR] 识别完成: Type={parsed['question_type']}, Ans={parsed['final_answer']}")
    return enhanced_img, parsed

def run_step_choice(enhanced_img, parsed, logic_solver, gemini_service, grounding_service):
    """选择题深度校验"""
    ocr_answer = parsed.get('final_answer', '').strip().upper()
    question_text = parsed.get('question_text', '')
    
    print(f"[Judge] 启动逻辑解题...")
    gemini_solve_res = logic_solver.solve_choice(enhanced_img, question_text)
    
    if not gemini_solve_res:
        gemini_solve_res = ocr_answer

    if ocr_answer == gemini_solve_res.upper():
        print(f"[Judge] 结果一致，信任 OCR 识别值")
        return parsed

    print(f"[Judge] 发现冲突 ({ocr_answer} vs {gemini_solve_res})，执行视觉复核...")
    
    bboxes = grounding_service.get_bboxes(enhanced_img, "answer")
    visual_materials = [CropTool.crop_by_normalized_bbox(enhanced_img, bbox) for bbox in bboxes] if bboxes else [enhanced_img]

    verify_prompt = CHOICE_VISUAL_VERIFY_PROMPT.format(gemini_solve_res=gemini_solve_res)
    
    # 使用重试助手
    success, v_res = _gemini_generate_with_retry(gemini_service, verify_prompt, visual_materials, mode="yes_no")
    
    if success and "YES" in v_res.strip().upper():
        print(f"[Judge] 视觉复核确认，修正答案为: {gemini_solve_res}")
        parsed['answer_text'] = gemini_solve_res
        parsed['final_answer'] = gemini_solve_res
    else:
        print(f"[Judge] 视觉复核未通过或超时，维持原始 OCR 结果")

    return parsed

def run_step_completion(enhanced_img, parsed, logic_solver, gemini_service, grounding_service):
    """填空题深度校验"""
    print(f"[Judge] 启动填空题深度校验")
    ocr_raw = parsed.get('final_answer', '').strip()
    question_text = parsed.get('question_text', '')
    
    # 1. 逻辑解题 (LogicSolver 内部已包含针对 solve_completion 的重试)
    logic_ans_str = logic_solver.solve_completion(enhanced_img, question_text)
    logical_parts = [p.strip() for p in logic_ans_str.split('||')] if logic_ans_str else [""]
    logical_count = len(logical_parts)
    logical_ans_str = " || ".join(logical_parts) 
    
    # 2. 状态检查
    blank_bboxes = grounding_service.get_bboxes(enhanced_img, "blank")
    answer_bboxes = grounding_service.get_bboxes(enhanced_img, "answer")
    current_parts = _safe_split_answers(ocr_raw)
    
    struct_consistent = (logical_count == len(blank_bboxes) == len(current_parts))
    content_consistent = (len(answer_bboxes) == len([p for p in current_parts if p.strip()]))
    
    if struct_consistent and content_consistent:
        print("[Judge] ✨ 维度数据匹配，跳过结构重构")
    else:
        # 3. 结构重构 (带重试)
        print("[Judge] ⚠️ 数据分歧，触发结构重构...")
        struct_prompt = COMPLETION_STRUCTURE_CHECK_PROMPT.format(
            question_text=question_text, logical_ans=logical_ans_str,
            logical_count=logical_count, ocr_raw=ocr_raw,
            blank_count=len(blank_bboxes), answer_count=len(answer_bboxes)
        )
        s_success, raw_struct = _gemini_generate_with_retry(gemini_service, struct_prompt, [enhanced_img], mode="tag")
        if s_success:
            match = re.search(r'<answer>(.*?)</answer>', raw_struct, re.IGNORECASE | re.DOTALL)
            if match:
                reconstructed = [p.strip() for p in match.group(1).split('||')]
                current_parts = ["" if "[EMPTY]" in p.upper() else p for p in reconstructed]

    if len(current_parts) != logical_count:
        current_parts = (current_parts + [""] * logical_count)[:logical_count]

    # 4. 视觉核验
    final_verified_parts = list(current_parts)
    need_verify_indices = [i for i, (c, l) in enumerate(zip(current_parts, logical_parts)) 
                           if c and l and c.replace('$', '').replace(' ', '') != l.replace('$', '').replace(' ', '')]

    if not need_verify_indices:
        print("[Judge] 识别一致")
    elif len(need_verify_indices) == 1 and logical_count == 1 and len(answer_bboxes) == 1:
        # 单空核验 (带重试)
        print("[Judge] 启动单空精准核验...")
        visual_src = [CropTool.crop_by_normalized_bbox(enhanced_img, answer_bboxes[0])] if answer_bboxes else [enhanced_img]
        v_prompt = COMPLETION_VISUAL_VERIFY_PROMPT.format(correct_val=logical_parts[0], current_val=current_parts[0])
        vv_success, vv_res = _gemini_generate_with_retry(gemini_service, v_prompt, visual_src, mode="yes_no")
        if vv_success and "YES" in vv_res.upper():
            final_verified_parts[0] = logical_parts[0]
    else:
        # 批量核验 (带重试)
        print(f"[Judge] 执行大图批量核验 (待核验: {len(need_verify_indices)} 项)...")
        verify_list_str = "\n".join([f"{i+1}: 转录='{current_parts[i]}', 标准='{logical_parts[i]}'" for i in need_verify_indices])
        batch_prompt = COMPLETION_VISUAL_BATCH_VERIFY_PROMPT.format(verify_list=verify_list_str)
        vb_success, b_res = _gemini_generate_with_retry(gemini_service, batch_prompt, [enhanced_img], mode="batch_yes_no")
        if vb_success:
            for i in need_verify_indices:
                if re.search(rf"{i+1}:\s*YES", b_res, re.IGNORECASE):
                    final_verified_parts[i] = logical_parts[i]

    final_ans = _safe_join_answers(final_verified_parts)
    parsed['final_answer'] = final_ans
    parsed['answer_text'] = final_ans
    return parsed

# ==========================================
# 主入口
# ==========================================

def run_evaluation(base64_image):
    """评估主入口"""
    local_ocr_service = LocalOCRClient()
    local_grounding_service = GroundingClient()
    local_gemini_service = GeminiClient()
    local_logic_solver = LogicSolver(local_gemini_service)
    
    enhanced_img, result = run_ocr(base64_image, local_ocr_service)
    
    q_type = result.get('question_type', '')
    if q_type == "选择题":
        result = run_step_choice(enhanced_img, result, local_logic_solver, local_gemini_service, local_grounding_service)
    elif q_type == "填空题" or q_type == "小学口算题":
        result = run_step_completion(enhanced_img, result, local_logic_solver, local_gemini_service, local_grounding_service)
    
    print(f"[Final] 判定答案: {result['final_answer']}")
    
    return (f"<st_question>{result.get('question_text','')}</st_question>\n"
            f"<st_question_id>{result.get('question_id','')}</st_question_id>\n"
            f"<st_question_type>{result.get('question_type','')}</st_question_type>\n"
            f"<st_answer>{result.get('answer_text','')}</st_answer>\n"
            f"<st_final_answer>{result.get('final_answer','')}</st_final_answer>")

if __name__ == "__main__":
    # images_path = "/mnt/afs_ocr/tongronglei/workspace/mathocr/2_eval/test_ocr/fuduji"
    # if os.path.exists(images_path):
    #     sample_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    #     for filename in sample_files:
    #         print(f"\n{'='*20} Processing: {filename} {'='*20}")
    #         try:
    #             with open(os.path.join(images_path, filename), "rb") as img_f:
    #                 img_b64 = base64.b64encode(img_f.read()).decode('utf-8')
    #             run_evaluation(img_b64)
    #         except Exception as e:
    #             print(f"[Fatal] {e}")
    #         print("-" * 60)

    filename = "/mnt/afs_ocr/tongronglei/workspace/mathocr/2_eval/test_ocr/tmp/11-522417e5-78e1-43c7-b972-49a3d607e008.jpeg"
    with open(filename, "rb") as img_f:
        img_b64 = base64.b64encode(img_f.read()).decode('utf-8')
        run_evaluation(img_b64)