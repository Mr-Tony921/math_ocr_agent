import os
import re
from PIL import Image
from tools.crop_utils import CropTool
from tools.image_utils import ImageToolbox
from tools.ocr_client import LocalOCRClient
from tools.grounding_client import GroundingClient
from tools.gemini_client import GeminiClient
from utils.prompts import (
    CHOICE_SOLVE_PROMPT, 
    CHOICE_VISUAL_VERIFY_PROMPT,
    COMPLETION_SOLVE_PROMPT,
    COMPLETION_STRUCTURE_CHECK_PROMPT,
    COMPLETION_VISUAL_VERIFY_PROMPT
)
import config

# 初始化全局工具实例
ocr_service = LocalOCRClient()
grounding_service = GroundingClient()
gemini_service = GeminiClient()

def run_ocr(image_path):
    """
    第一步：加载、增强并执行本地 OCR 识别
    """
    raw_img = Image.open(image_path)
    enhanced_img = ImageToolbox.internvl_ocr_augment(raw_img)
    
    print(f"[OCR] 发送请求至本地节点...")
    raw_text = ocr_service.generate_content(enhanced_img)
    parsed = ocr_service.parse_results(raw_text)
    
    print(f"[OCR] 识别完成: Type={parsed['question_type']}, Ans={parsed['final_answer']}")
    return enhanced_img, parsed

def run_step_choice(enhanced_img, parsed):
    """
    选择题深度校验逻辑：逻辑解题 + 视觉核对
    """
    ocr_answer = parsed.get('final_answer', '').strip().upper()
    question_text = parsed.get('question_text', '')
    
    # 1. Gemini 逻辑解题
    print(f"[Judge] 启动 Gemini 逻辑解题...")
    solve_prompt = CHOICE_SOLVE_PROMPT.format(question_text=question_text)
    gemini_solve_res = ocr_answer
    
    try:
        success, raw_response = gemini_service.generate_content(solve_prompt, [enhanced_img])
        if success:
            match = re.search(r'<answer>(.*?)</answer>', raw_response, re.IGNORECASE | re.DOTALL)
            if match:
                gemini_solve_res = match.group(1).strip()
                print(f"[Judge] 逻辑解题结果: {gemini_solve_res}")
    except Exception as e:
        print(f"[Error] 解题环节异常: {e}")

    # 2. 结果一致性对比
    if ocr_answer == gemini_solve_res.upper():
        print(f"[Judge] 结果一致，信任 OCR 识别值")
        return parsed

    # 3. 结果冲突，启动视觉复核
    print(f"[Judge] 发现冲突 ({ocr_answer} vs {gemini_solve_res})，执行视觉复核...")
    
    bboxes = grounding_service.get_bboxes(enhanced_img, "answer")
    visual_materials = []
    if bboxes:
        for i, bbox in enumerate(bboxes):
            crop_img = CropTool.crop_by_normalized_bbox(enhanced_img, bbox)
            visual_materials.append(crop_img)
            # crop_img.save(f"debug/answer_{i}.jpg")
    
    if not visual_materials:
        visual_materials = [enhanced_img]

    verify_prompt = CHOICE_VISUAL_VERIFY_PROMPT.format(
        gemini_solve_res=gemini_solve_res,
        ocr_answer=ocr_answer
    )
    
    try:
        v_success, v_res = gemini_service.generate_content(verify_prompt, visual_materials)
        if v_success and "YES" in v_res.strip().upper():
            print(f"[Judge] 视觉复核确认，修正答案为: {gemini_solve_res}")
            # 同步更新字典中的答案字段
            parsed['answer_text'] = gemini_solve_res
            parsed['final_answer'] = gemini_solve_res
        else:
            print(f"[Judge] 视觉复核未通过，维持原始 OCR 结果")
    except Exception as e:
        print(f"[Error] 视觉复核异常: {e}")

    return parsed

def _safe_split_answers(text):
    """初步分割 OCR 结果"""
    if not text: return []
    # 兼容中英文分号
    normalized = text.replace('；', ';')
    return [x.strip() for x in normalized.split(';') if x.strip()]

def _safe_join_answers(parts):
    """组装最终输出字符串"""
    return "；".join([p if p else "" for p in parts])

def run_step_completion(enhanced_img, parsed):
    """
    填空题深度校验逻辑
    """
    print(f"[Judge] 启动填空题深度校验...")
    
    ocr_raw = parsed.get('final_answer', '').strip()
    question_text = parsed.get('question_text', '')
    
    # 1. 视觉检测统计
    blank_bboxes = grounding_service.get_bboxes(enhanced_img, "blank")
    answer_bboxes = grounding_service.get_bboxes(enhanced_img, "answer")
    blank_count, answer_count = len(blank_bboxes), len(answer_bboxes)
    
    current_parts = _safe_split_answers(ocr_raw)
    ocr_count = len(current_parts)
    print(f"[Judge] 数量统计: OCR={ocr_count}, Blank={blank_count}, Answer={answer_count}")

    # 判定单空模式
    is_clean_single = (ocr_count == 1 and blank_count == 1 and answer_count == 1)
    if is_clean_single:
        print("[Judge] 命中单空模式，将执行精准切片核验")

    # 2. 结构重构 (多空或数量不匹配时触发)
    if not is_clean_single and (ocr_count != blank_count or blank_count > 1):
        print("[Judge] 结构存在不确定性，执行 Gemini 结构修复...")
        struct_prompt = COMPLETION_STRUCTURE_CHECK_PROMPT.format(
            ocr_raw=ocr_raw, blank_count=blank_count, answer_count=answer_count
        )
        try:
            success, raw_struct = gemini_service.generate_content(struct_prompt, [enhanced_img])
            if success:
                match = re.search(r'<answer>(.*?)</answer>', raw_struct, re.IGNORECASE | re.DOTALL)
                if match:
                    parts = [p.strip() for p in match.group(1).split('||')]
                    current_parts = ["" if "[EMPTY]" in p.upper() else p for p in parts]
                    print(f"[Judge] 结构修复完成: {current_parts}")
        except Exception as e:
            print(f"[Error] 结构修复异常: {e}")

    # 3. 逻辑解题
    print("[Judge] 执行逻辑解题获取标准答案...")
    solve_prompt = COMPLETION_SOLVE_PROMPT.format(question_text=question_text)
    logical_parts = []
    try:
        success, raw_solve = gemini_service.generate_content(solve_prompt, [enhanced_img])
        if success:
            match = re.search(r'<answer>(.*?)</answer>', raw_solve, re.IGNORECASE | re.DOTALL)
            if match:
                logical_parts = [p.strip() for p in match.group(1).split('||')]
                print(f"[Judge] 标准答案: {logical_parts}")
    except Exception as e:
        print(f"[Error] 逻辑解题异常: {e}")

    # 4. 逐空比对与视觉核验
    final_parts = []
    max_len = max(len(current_parts), len(logical_parts))
    
    for i in range(max_len):
        curr = current_parts[i] if i < len(current_parts) else ""
        logic = logical_parts[i] if i < len(logical_parts) else ""
        
        if not curr:
            final_parts.append("")
            continue
        if not logic or curr.replace(' ','') == logic.replace(' ',''):
            final_parts.append(curr)
            continue
            
        # 冲突核验
        print(f"[Judge] 空位 #{i+1} 冲突: '{curr}' vs '{logic}'，执行视觉核验...")
        visual_src = [enhanced_img]
        if is_clean_single and answer_bboxes:
            try:
                visual_src = [CropTool.crop_by_normalized_bbox(enhanced_img, answer_bboxes[0])]
                print(f"[Judge] 使用局部切片核验")
            except: pass

        verify_prompt = COMPLETION_VISUAL_VERIFY_PROMPT.format(correct_val=logic, current_val=curr)
        try:
            v_success, v_res = gemini_service.generate_content(verify_prompt, visual_src)
            if v_success and "YES" in v_res.upper():
                print(f"[Judge] 视觉确认修正: {logic}")
                final_parts.append(logic)
            else:
                final_parts.append(curr)
        except:
            final_parts.append(curr)

    # 5. 更新结果
    final_ans = _safe_join_answers(final_parts)
    parsed['final_answer'] = final_ans
    parsed['answer_text'] = final_ans
    return parsed

def run_evaluation(image_path):
    """
    评估主入口
    """
    if not os.path.exists(image_path):
        return None

    # 1. 基础 OCR
    enhanced_img, result = run_ocr(image_path)
    
    # 2. 深度校验路由
    q_type = result.get('question_type', '')
    if q_type == "选择题":
        result = run_step_choice(enhanced_img, result)
    elif q_type == "填空题":
        run_step_completion(enhanced_img, result)
    
    print(f"[Final] 判定答案: {result['final_answer']}")
    return result

if __name__ == "__main__":
    test_file = "/mnt/afs/tongronglei/code/judge_data/test_ocr/fuduji/img_v3_02u6_37aeefe9-8b0f-44ad-b1e5-d03c95ee192g.png"
    final_parsed_result = run_evaluation(test_file)

    # images_path = "/mnt/afs/tongronglei/code/judge_data/test_ocr/tmp"
    
    # if os.path.exists(images_path):
    #     sample_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    #     for filename in sample_files:
    #         full_path = os.path.join(images_path, filename)
    #         print(f"\n{'='*20} Processing: {filename} {'='*20}")
            
    #         try:
    #             final_parsed_result = run_evaluation(full_path)
    #         except Exception as e:
    #             print(f"[Fatal] 文件处理崩溃: {e}")
                
    #         print("-" * 60)