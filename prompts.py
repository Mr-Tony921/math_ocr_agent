# 带作答区域数量
vision_prompt = """
你是视觉模型 OCR 校准专家。

你的任务：
1. 先根据图片内容理解题目与学生书写，生成各字段初步 OCR 内容（仅用于校准，不直接输出）。字段格式参考 block_prompt：
{block_prompt}

2. 再基于以下 full_result 以及作答数量信息对 <st_question>、<st_answer>、<st_final_answer> 三个字段进行校准：
full_result:
{full_result}

作答信息：
- total_slots: {total_slots}  # 题目总作答区域数量（包括空白）
- answered_slots: {answered_slots}  # 学生实际填写数量

校准规则：
1. <st_question>：
   - 不得混入学生答案。
   - 内容以 full_result 为主，仅在明显看到字符/数字/符号 OCR 错误时才修正。
   - 不要重写题干或增加/减少内容。

2. <st_answer> 与 <st_final_answer>：
   - 校准后的答案应总共包含 total_slots 段，每段对应一个作答区域。
   - 已填写的 answered_slots 段，尽量保持图片中可识别的内容，并修正 OCR 错误。
   - 剩余空白段应严格保留，不得填充或编造内容。
   - 不得随意增加或减少作答区域数量。

3. 严格禁止：
   - 改变 tag 结构或新增/删除字段；
   - 将图片中不存在的内容写入 full_result；
   - 根据题意推断正确答案。

最终输出要求：
- 输出格式必须与输入 full_result 完全一致；
- 仅修改 <st_question>、<st_answer>、<st_final_answer> 三个字段内容；
- 不输出任何解释、过程或分析；
- 最终输出仅包含更新后的完整 full_result。
"""

# 优化空白幻觉
# vision_prompt = """
# 你是视觉模型 OCR 校准专家。

# 你的任务：
# 1. 先根据图片内容理解题目与学生书写，生成各字段初步 OCR 内容（仅用于校准，不直接输出）。字段格式参考 block_prompt：
# {block_prompt}

# 2. 再基于以下 full_result 对 <st_question>、<st_answer>、<st_final_answer> 三个字段进行校准：
# full_result:
# {full_result}

# 校准规则：
# 1. <st_question>：
#    - 不得混入学生答案。
#    - 内容以 full_result 为主，仅在明显看到某个字符/数字/符号被错误 OCR 时才修正。
#    - 除非模型非常确定图片内容与 full_result 不一致，否则不要大幅重写。

# 2. <st_answer> 与 <st_final_answer>：
#    - 若某段为空、缺失或被遮挡，必须相信 full_result，不要补齐或编造。
#    - 对于图片中明显可辨认的字符/数字/公式，允许进行纠正。
#    - 不得根据题意推理正确答案，不得填补模型想象内容。

# 3. 严格禁止：
#    - 改变 tag 结构或新增删除字段；
#    - 将图片不存在的内容写入 full_result；
#    - 根据题意自动纠正为“标准答案”。

# 最终输出要求：
# - 输出格式必须与输入 full_result 完全一致；
# - 仅修改 <st_question>、<st_answer>、<st_final_answer> 三个字段内容；
# - 不输出任何解释、过程、分析；
# - 最终输出仅包含更新后的完整 full_result。
# """

# P3 94.24 89.39 97.29 93.23 90.41
# vision_prompt = """
# 你是视觉模型 OCR 校准专家。请先根据图片内容理解题目和学生书写，生成各字段初步 OCR 识别结果，格式参考 block_prompt：

# {block_prompt}

# 然后结合用户提供的完整结果（full_result）进行校准：
# - <st_question>：OCR 识别题干，必须忠实反映图片内容。
# - <st_answer> 和 <st_final_answer>：OCR 识别学生答案，清晰可读的内容优先，必要时参考 full_result 校正。
# - 其他字段（<st_question_title>、<st_question_pure_content>、<st_question_id>、<st_question_type> 等）：如发现明显错误可修正，否则偏向相信 full_result。

# 输入：
# - full_result: {full_result}

# 要求：
# 1. 输出格式必须与 full_result 完全一致；
# 2. 不破坏任何 tag 或结构，只修改其中的内容；
# 3. 忠实还原图片中 OCR 的真实信息，避免盲目纠正为正确答案；
# 4. 最终输出仅包含更新后的完整 full_result。
# """

# 志强测试用例表格使用
# vision_prompt = """
# 你是视觉模型的 OCR 校准专家。你的任务是根据题目图片忠实识别学生书写内容，并判断是否需要更新 full_result 中的 <st_answer> 和 <st_final_answer>。

# 输入：
# - 完整结果（整页 OCR）：{full_result}
# - 图片内容（学生实际书写）

# 要求：
# 1. 输出格式必须与 full_result 完全一致，只能修改 <st_answer> 和 <st_final_answer>，其他字段保持不变。
# 2. 始终以图片上学生实际书写为主证据：
#    - 即使与 full_result 或算术正确性不符，也必须忠实还原学生写法。
# 3. 仅在以下情况参考 full_result：
#    - 学生书写模糊、涂改、删除线或无法识别
# 4. 不参考任何手写小图或其他来源，不受外部噪声干扰。
# 5. 若学生书写清晰可读 → 忠实还原，不自动纠正为正确答案。
# 6. **绝不破坏最终输出的任何 tag 或结构**。
# 7. 最终输出：
#    - 返回一个与 full_result 完全相同格式的结果
#    - 仅 <st_answer> 和 <st_final_answer> 根据判断更新
#    - 不输出推理过程
# """

merge_prompt = """
你是一名资深数学题 OCR 内容融合专家。你收到两类结果：
1. 完整结果（整页 OCR）：{full_result}
2. {num_crop} 个手写框 OCR 结果（局部小图）：{crop_results}

任务：只更新 full_result 中的 <st_answer> 和 <st_final_answer> 字段，其他字段保持完全不变，格式必须一致。

规则：
1. 手写框为空（漏识别）时，**务必相信完整结果**。
2. 手写框有内容且合理 → 优先手写框。
3. 若手写框与完整结果冲突：
   - 内容显然错误 → 选更合理的结果
   - 题型限制可辅助判断
4. 目标始终是还原学生真实书写，而不是正确答案。

请输出仅修改 <st_answer> 和 <st_final_answer> 后的完整 full_result。
"""

block_prompt = """请从图片中提取题目的题干文字、题目大标题、无冗余的题目内容、全局题号、全局题型、答案文字、最终结果，按以下格式输出：

<st_question>题干文字</st_question>
<st_question_title>题目大标题</st_question_title>
<st_question_pure_content>无冗余的题目内容</st_question_pure_content>
<st_question_id>全局题号</st_question_id>
<st_question_type>全局题型</st_question_type>
<st_answer>答案文字</st_answer>
<st_final_answer>最终结果</st_final_answer>

提取规则：
1. 仅提取文字，不解题，不判断正误。
2. 保持原貌，不修改任何文字、数字、符号。
3. 不创造或补全答案。
4. 数学公式使用 LaTeX，表格使用 Markdown。
5. 不提取图形中的文字或数字，不提取水印、页眉页脚等非题目信息。
6. 若图片中有手写答案，保留完整解答过程及最终答案，但忽略涂抹、划线删除或大面积打叉部分。
7. 无答案时，**答案文字**和**最终结果**均置为空 ""。
8. <st_question_title> 提取反映题型或结构的文字，如“二、填空题”“第一部分 选择题”，无则置空。
9. <st_question_pure_content> 去掉与解题无关信息，只保留核心题目文字。
10. <st_question_id> 仅提取全局题号，无题号置空。中文括号保留，中括号取内部内容，特殊符号保留，忽略后续空格或标点。
11. <st_question_type>：单子题选择最合适的类型 ["选择题"、"填空题"、"判断题"、"计算题"、"解答题"、"证明题"、"作图题"、"小学口算题"]；多子题若类型一致返回该类型，否则返回 "综合题"。
12. <st_final_answer> 与图片及答案一致，删除解题过程；多子题用分号 ";" 分隔。
13. 选择题答案只提取选中选项字母，不含解题过程。
14. 证明题、作图题或综合题中对应子题的最终结果置空。
15. 题型提示：
- 选择题：题干含括号，答案为 A/B/C/D。
- 填空题：题干含括号或下划线，答案简短。
- 判断题：题干含括号，答案为 ✓、√、×、X。
- 计算题：数值或运算符构成的竖式、横式、列式或脱式计算，题干汉字少。
- 解答题：应用题，汉字多，无括号。
- 证明题：含“求证”“证明”字样。
- 作图题：含“画出”“画图”字样。
- 小学口算题：多道小学难度的加减乘除混合运算，与计算题的区别主要在数量和难度上。
"""

crop_prompt = """抽取图中的手写体内容，并以如下格式返回结果：<st_handwritten>手写体内容</st_handwritten>"""

