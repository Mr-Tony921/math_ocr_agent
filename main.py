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
    COMPLETION_VISUAL_VERIFY_PROMPT,
    COMPLETION_VISUAL_BATCH_VERIFY_PROMPT
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
        gemini_solve_res=gemini_solve_res
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
    填空题深度校验主逻辑
    逻辑流：逻辑解题(定基调) -> 双维度一致性检查 -> 结构重构(按需) -> 视觉核验(切片/批量)
    """
    print(f"[Judge] 启动填空题深度校验")
    
    ocr_raw = parsed.get('final_answer', '').strip()
    question_text = parsed.get('question_text', '')
    
    # --- 1. 逻辑解题：确立“法理”上的空数和标准答案 ---
    print("[Judge] Step 1: 执行逻辑解题...")
    solve_prompt = COMPLETION_SOLVE_PROMPT.format(question_text=question_text)
    logical_parts = []
    try:
        success, raw_solve = gemini_service.generate_content(solve_prompt, [enhanced_img])
        if success:
            match = re.search(r'<answer>(.*?)</answer>', raw_solve, re.IGNORECASE | re.DOTALL)
            if match:
                # 预处理：统一转为列表
                logical_parts = [p.strip() for p in match.group(1).split('||')]
    except Exception as e:
        print(f"[Error] 逻辑解题异常: {e}")
    
    # 确定逻辑预期空数，至少为1
    logical_count = len(logical_parts) if logical_parts else 1
    logical_ans_str = " || ".join(logical_parts)
    print(f"[Judge] 逻辑基准确认: 预期空数={logical_count}, 参考答案={logical_parts}")

    # --- 2. 状态检查：双维度判定是否需要结构重构 ---
    blank_bboxes = grounding_service.get_bboxes(enhanced_img, "blank")
    answer_bboxes = grounding_service.get_bboxes(enhanced_img, "answer")
    
    current_parts = _safe_split_answers(ocr_raw)
    
    # 维度 A：结构统计 (坑位总数)
    ocr_total_count = len(current_parts)
    blank_count = len(blank_bboxes)
    
    # 维度 B：内容统计 (实填内容数)
    ocr_valid_parts = [p for p in current_parts if p.strip()]
    ocr_valid_count = len(ocr_valid_parts)
    answer_count = len(answer_bboxes)
    
    print(f"[Judge] Step 2: 维度对齐检查...")
    print(f"       - 结构维度: 逻辑预期={logical_count}, 视觉空位={blank_count}, OCR总项={ocr_total_count}")
    print(f"       - 内容维度: 视觉笔迹={answer_count}, OCR有效作答={ocr_valid_count}")

    # 判定逻辑：
    # 1. 坑位必须对齐：逻辑空位 == 视觉横线 == OCR拆分的总段数
    # 2. 内容必须对齐：视觉笔迹块数 == OCR识别出的非空项数
    struct_consistent = (logical_count == blank_count == ocr_total_count)
    content_consistent = (answer_count == ocr_valid_count)
    
    # 只有双维度完全匹配，才跳过结构重构
    needs_structure_check = not (struct_consistent and content_consistent)
    
    if not needs_structure_check:
        print("[Judge] ✨ 维度数据完美匹配，跳过结构重构环节")
    else:
        # --- 3. 结构重构：利用逻辑锚点纠正视觉/OCR偏差 ---
        print("[Judge] ⚠️ 数据分歧，触发 Gemini 结构重构 (归并碎片)...")
        struct_prompt = COMPLETION_STRUCTURE_CHECK_PROMPT.format(
            question_text=question_text,
            logical_ans=logical_ans_str,
            logical_count=logical_count,
            ocr_raw=ocr_raw,
            blank_count=blank_count,
            answer_count=answer_count
        )
        try:
            v_success, raw_struct = gemini_service.generate_content(struct_prompt, [enhanced_img])
            if v_success:
                match = re.search(r'<answer>(.*?)</answer>', raw_struct, re.IGNORECASE | re.DOTALL)
                if match:
                    struct_res = match.group(1).strip()
                    reconstructed = [p.strip() for p in struct_res.split('||')]
                    # 转为内部空字符串处理
                    current_parts = ["" if "[EMPTY]" in p.upper() else p for p in reconstructed]
                    print(f"[Judge] 结构重构完成: {current_parts}")
        except Exception as e:
            print(f"[Error] 结构重构异常: {e}，回退至原始 OCR")

    # 防御性对齐：确保数组长度严格等于逻辑空数
    if len(current_parts) != logical_count:
        current_parts = (current_parts + [""] * logical_count)[:logical_count]

    # --- 4. 视觉核验：分流处理冲突 ---
    final_verified_parts = list(current_parts)
    need_verify_indices = []
    
    # 统计重构后真实的有效作答数
    final_valid_count = len([p for p in current_parts if p.strip()])

    for i in range(logical_count):
        curr = current_parts[i]
        logic = logical_parts[i] if i < len(logical_parts) else ""
        # 归一化比对
        clean_curr = curr.replace('$', '').replace(' ', '')
        clean_logic = logic.replace('$', '').replace(' ', '')
        
        if curr and logic and clean_curr != clean_logic:
            need_verify_indices.append(i)

    if not need_verify_indices:
        print("[Judge] 识别一致，流程结束")
    
    # 模式 A: 只有在逻辑、内容、冲突完全对齐为 1 时，才允许使用切片小图
    elif len(need_verify_indices) == 1 and logical_count == 1 and answer_count == 1:
        print("[Judge] 满足安全切片条件，启动单空精准核验...")
        idx = need_verify_indices[0]
        visual_src = [enhanced_img]
        if answer_bboxes:
            try:
                # 此时 answer_count == 1 确保了 bbox 列表里只有一个明确的作答区
                visual_src = [CropTool.crop_by_normalized_bbox(enhanced_img, answer_bboxes[0])]
            except: pass
            
        verify_prompt = COMPLETION_VISUAL_VERIFY_PROMPT.format(
            correct_val=logical_parts[idx], current_val=current_parts[idx]
        )
        success, v_res = gemini_service.generate_content(verify_prompt, visual_src)
        if success and "YES" in v_res.upper():
            print(f"[Judge] 修正确认: {logical_parts[idx]}")
            final_verified_parts[idx] = logical_parts[idx]

    # 模式 B: 多空冲突、或者单空但笔迹区不唯一（笔迹碎片化），使用大图批量核验
    else:
        print(f"[Judge] 笔迹区不唯一或多空冲突 (待核验: {len(need_verify_indices)} 项)，执行大图批量核验...")
        verify_list_str = ""
        for i in need_verify_indices:
            v_curr = current_parts[i] if current_parts[i] else "[EMPTY]"
            v_logic = logical_parts[i] if i < len(logical_parts) else ""
            verify_list_str += f"{i+1}: 转录='{v_curr}', 标准='{v_logic}'\n"
        
        batch_prompt = COMPLETION_VISUAL_BATCH_VERIFY_PROMPT.format(verify_list=verify_list_str)
        try:
            success, b_res = gemini_service.generate_content(batch_prompt, [enhanced_img])
            if success:
                for i in need_verify_indices:
                    # 使用正则精准匹配编号结果
                    if re.search(rf"{i+1}:\s*YES", b_res, re.IGNORECASE):
                        print(f"[Judge] 批量核验：空位 #{i+1} 判定一致，修正为标准答案")
                        final_verified_parts[i] = logical_parts[i]
        except Exception as e:
            print(f"[Error] 批量核验异常: {e}")

    # --- 5. 组装并同步最终结果 ---
    final_ans = _safe_join_answers(final_verified_parts)
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
        result = run_step_completion(enhanced_img, result)
    
    print(f"[Final] 判定答案: {result['final_answer']}")
    return f"<st_question>{result['question_text']}</st_question>\n<st_question_id>{result['question_id']}</st_question_id>\n<st_question_type>{result['question_type']}</st_question_type>\n<st_answer>{result['answer_text']}</st_answer>\n<st_final_answer>{result['final_answer']}</st_final_answer>"

if __name__ == "__main__":
    # test_file = "/mnt/afs/tongronglei/code/judge_data/test_ocr/tmp/17-94adb5c7-7c32-469e-9901-57d8010a4eb8.jpeg"
    # final_parsed_result = run_evaluation(test_file)

    images_path = "/mnt/afs/tongronglei/code/judge_data/test_ocr/tmp"
    
    if os.path.exists(images_path):
        sample_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in sample_files:
            full_path = os.path.join(images_path, filename)
            print(f"\n{'='*20} Processing: {filename} {'='*20}")
            
            try:
                final_parsed_result = run_evaluation(full_path)
            except Exception as e:
                print(f"[Fatal] 文件处理崩溃: {e}")
                
            print("-" * 60)