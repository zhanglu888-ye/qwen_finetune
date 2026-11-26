import asyncio
import os
import json
import pandas as pd
from llm_util import LlmUtil
from tqdm import tqdm
import argparse  # 新增：用于命令行参数解析
import re


##提示词10
prompt = """你是一位图像火灾预警技术专家，需要对上传的图像输出结构化描述、短文本描述、10个问答对。
【任务逻辑】
第一步：依据图像的“风险等级”和“风险代码”，结合图像及“结构化和短文本描述要求”，输出结构化描述和短文本描述；
第二步：依据“风险等级”、“风险代码”、第一步输出的“结构化描述”、“短文本描述”，结合图像及“问答对生成要求”，输出10个问答对。

【输入信息】                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
图像数据（一张可见光图像）、图像风险等级和风险代码（如：高风险 HS1）
风险等级和风险代码：

【第一步：结构化和短文本描述要求】
“风险等级定义表”
| 风险等级   | 事件                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 现象                                                                                                                                                               | 序列号 |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| 高风险     | 无法明确事件                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 浓黑烟、黄烟、其他异常颜色烟雾；                                                                                                                                     | HC1      |
| 高风险     | 消防人员或消防设备出动伴随烟雾或火焰                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | /                                                                                                                                                                  | HC2      |
| 高风险     | 1.特定区域的失控明火： <br> ● 山林区域：任何失控明火（如违规祭祀、烧烤、野炊、取暖、丢弃烟头等引发的林火或具有高度蔓延风险的火情）。 <br> ● 工业区：厂房失火、仓储/物料堆场燃烧、化学品泄漏燃烧、违规动火作业失控、区域内违规烧烤/祭祀失控。<br> ● 农田/乡村区域：农事用火失控（如秸秆、荒草、垃圾等焚烧失控或大面积蔓延）。<br> ● 公共场所/非住宅建筑区域：祭祀用火失控、大型或失控烟花爆竹燃放（火势较大或有蔓延风险）、失控的户外烧烤/篝火。<br>2. 普遍高风险事件：<br> ● 车辆自燃或事故引发的燃烧。<br> ● 电线电缆过载、短路或故障引发的燃烧。<br> ● 居民住宅建筑失火（包括因室内外用火不慎如失控的祭祀/烧烤/烟花、电器故障、燃气泄漏、纵火等原因，导致住宅建筑物主体结构或内部可燃物发生燃烧） | /                                                                                                                                                                  | HC3      |
| 疑似高风险 | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 图像中存在烟雾，其来源无法明确判断为烟囱/排口。烟雾特征为：大量白色/灰色烟雾（视觉上烟雾弥漫范围广、密度高，明显区别于少量稀疏烟雾）。                                                 | HS1      |
| 中风险     | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 烟囱/排口排烟（明确看到烟是从烟囱/排口冒出）黑色、深灰色、或持续排放的五颜六色工业废气（非正常水蒸气）                                                                               | MC1      |
| 疑似中风险 | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 图像中存在烟雾，其来源无法明确判断为烟囱/排口。烟雾特征为：非大量、非浓密的白色/灰色烟雾（通常表现为少量、稀薄，或中等程度但明显未达到HS1的规模），且不属于明确的LC2-LC11中定义的其他正常现象。 | MS1      |
| 正常       | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 烟囱/排口排烟/排汽（明确看到烟是从烟囱/排口冒出，且为正常水蒸气或浅色、少量烟雾，非浓黑/黄/异常颜色烟雾）                                                                             | LC1      |
| 正常       | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 居民区：受控的小型烧烤、小型烟花 (有人员看管)、居民区祭祀用火（有人看管）、日常工具排烟（小型烟雾，必须能明确识别工具来源且受控，否则少量不明烟雾应考虑MS1）                                       | LC2      |
| 正常       | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 扬尘 (明确为施工、车辆行驶等引起，形态与燃烧烟雾有区别)                                                                                                                            | LC3      |
| 正常       | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 雾气、霾 (大范围、均匀分布，区别于局部、漂浮的烟雾)                                                                                                                              | LC4      |
| 正常       | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 车辆发动机过热 (未起火，可能伴有少量水蒸气，非燃烧产生的烟雾)                                                                                                                        | LC5      |
| 正常       | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 太阳光照                                                                                                                                                             | LC6      |
| 正常       | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 照明设施                                                                                                                                                             | LC7      |
| 正常       | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 镜头雨珠、镜头脏污                                                                                                                                                           | LC8      |
| 正常       | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 云层漂浮，飞鸟、飞虫群                                                                                                                                                         | LC9      |
| 正常       | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 农药喷洒(必须存在喷洒农药的人或喷洒农药的无人机等设备，仔细观察室否存在人或无人机等设备)                                                                                                      | LC10     |
| 正常       | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 阳光反射：屋顶、管道、玻璃、水面、金属表面、路面等反射现象                                                                                                                            | LC11     |
| 正常       | /                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 图像不符合任何【事件】定义，且不符合上述任何特定【现象】（LC1-LC11, MC1, HS1, MS1, HC3现象部分）。关键是：图像中无火灾迹象，也无任何来源不明或可疑的烟雾/蒸汽。                                    | LC12     |

"结构化描述要求"
1. 严格且仅根据图片中清晰可见的、明确无误的视觉内容进行描述。
2. 依据图像的风险等级和风险代码，结合图像生成两种格式的描述：结构化描述和短文本言描述（可以合理参考“风险等级定义表”的内容）。
3. 参考图像的风险等级和风险代码，但不强制覆盖：
- 风险等级和风险代码提示了原始图像对被关注的原因。尝试在描述中体现这个原因相关的视觉特征，前提是这些特征在当前这张图像中确实可见。
- 如果风险等级和风险代码描述的现象在当前的图像中并不明显或不相关，那么你的描述应侧重于图像本身的视觉内容。
4. 禁止主观臆断、联想或推测图片中未明确展示的内容。
5. 图片中可能存在其他软件叠加的红框、蓝框、圆点、圆圈等标记，请忽略这些标记。
6. 描述要素定义如下：
时间信息：如果图像中可识别出时间，描述为具体的日期和时间，如：2025年5月23日 14:30等。
位置信息：描述图像中的地理环境类型，如：山林区域、农田乡村、工业区域、城镇建筑、道路沿线等。
事件性质：识别主要事件类型，如：失控火灾、正常排烟、消防作业、日常活动、无明显事件等。
火情状态：描述火灾相关情况，如：可见明火、无明火、火势较大、已扑灭、未发现火点等。
烟雾特征：描述烟雾的颜色、浓度、范围，如：浓黑色烟雾、少量白烟、大量灰烟、无烟雾等。
安全评估：总体安全状况评价，如：火情严重、已受控制、安全正常、需要关注、情况不明等。
风险等级和风险代码：结合上述信息，按照“风险等级定义表”中的标准，判别风险等级和风险代码，如：高风险 HC1
7. 输出要求：
- 每个字段内容：8-12字
- 严格按照指定格式输出
- 内容简洁准确，避免冗余
8.输出格式：
时间信息：[具体时间]
位置信息：[描述地理环境]
事件性质：[描述主要事件]
火情状态：[描述火灾情况]
烟雾特征：[描述烟雾状况]
安全评估：[描述安全状况]
风险等级和风险代码：[描述风险等级和风险代码]
9.输出示例：
# 结构化描述
时间信息：2025年5月23日 14:30
位置信息：山林密集区域
事件性质：失控火灾燃烧
火情状态：可见明显橙红色火焰
烟雾特征：大量浓黑色烟雾上升
安全评估：火情严重需紧急处置
风险等级和风险代码：高风险 HC3

“短文本描述要求”
1. 基于风险等级、风险代码和结构化描述，生成40-60字的流畅段落
- 语言自然，逻辑清晰
- 突出重点信息，符合中文表达习惯
- 保持客观准确，不添加主观色彩
2.特殊情况处理原则
- 证据不足时：使用"疑似"、"可能"等词汇
- 无相关内容时：填写"无"或"未发现"
- 多重情况时：选择最主要或最明显的情况描述
- 正常情况时：如实描述为"正常"或"无异常"
3. 输出格式：
[基于上述结构化信息和风险等级生成的40-60字自然段落描述]
4.输出示例：
# 短文本描述
2025年5月23日下午2点30分，山林密集区域发生严重失控火灾，现场可见明显的橙红色火焰正在燃烧，同时产生大量浓黑色烟雾向天空升腾。风险等级为高风险。目前尚未发现应急救援措施，火情严重需要紧急处置。

【第二步：问答对生成要求】
1. 严格根据图像、风险等级、风险代码、图像结构化描述信息、图像短文本描述信息，生成问答对，确保问题和答案与图像直接相关。
2. 每幅图像生成10组问答对，每组问答对包括一个问题和一个答案。
3. 问题应涵盖多种类型，包括具体细节、事件性质、数量、位置、特征、影响等。
4. 答案必须简洁明了，直接回应问题，避免冗长的解释。
5. 禁止主观臆断、联想或推测图片中未明确展示的内容。
6. 问题类型定义
问题1：图像中是否存在火灾迹象？是否能在图像中看到明显烟雾或火焰？
问题2：如果图像中存在火灾迹象，按照“风险等级定义表”的描述，请确认该火灾的风险等级和风险代码，并给出具体原因分析。
问题3：如果图像中存在火灾迹象，按照初起阶段、发展阶段、猛烈阶段和熄灭阶段四个阶段，请确认该火灾当前的发展阶段，并简要描述火情信息。
问题4：事件性质问题：
- 询问图像中发生的主要事件或现象。
- 例：图中描述的主要事件是什么？
问题5：时间信息问题：
- 询问事件发生的时间。
- 例：请问这个事件是在什么时间发生的？
问题6：位置相关问题
- 询问事件发生的具体地点或环境。
- 例：事件发生在什么样的地点？
问题7：数量与特征问题
- 询问图像中可见的物体数量或特征。
- 例：图像中有多少辆消防车？
问题8：烟雾与火焰特征问题
- 询问图像中烟雾或火焰的特征。
- 例：可以描述一下烟雾的颜色和类型吗？
问题9：应急响应问题
- 询问是否有应急响应措施或人员到场。
- 例：现场是否有消防人员在处理火情？
问题10：环境与影响问题
- 询问事件对环境或周围的影响。
- 例：火灾对周围植被有什么影响？

7. 输出格式要求
生成的问答对应遵循以下格式：
1. **问题**：[问题内容]
   **答案**：[答案内容]
   
2. **问题**：[问题内容]
   **答案**：[答案内容]

3. **问题**：[问题内容]
   **答案**：[答案内容]

4. **问题**：[问题内容]
   **答案**：[答案内容]

5. **问题**：[问题内容]
   **答案**：[答案内容]

6. **问题**：[问题内容]
   **答案**：[答案内容]
   
7. **问题**：[问题内容]
   **答案**：[答案内容]

8. **问题**：[问题内容]
   **答案**：[答案内容]

9. **问题**：[问题内容]
   **答案**：[答案内容]

10. **问题**：[问题内容]
   **答案**：[答案内容]

8. 输出示例
# 问答对
1. **问题**：图像中是否存在火灾迹象？是否能在图像中看到明显烟雾或火焰？
   **答案**：能看到明显火焰。
   
2. **问题**：如果图像中存在火灾迹象，按照“风险等级定义表”的描述，请确认该火灾的风险等级，并给出具体原因分析。
   **答案**：高风险。
   
3. **问题**：如果图像中存在火灾迹象，按照初起阶段、发展阶段、猛烈阶段和熄灭阶段四个阶段，请确认该火灾当前的发展阶段，并简要描述火情信息。
   **答案**：发展阶段。火情快速发展蔓延中。
   
4.**问题**：图中描述的主要事件是什么？
   **答案**：山林发生失控火灾。

5. **问题**：请问这个事件是在什么时间发生的？
   **答案**：2025年5月23日下午。

6. **问题**：事件发生在什么样的地点？
   **答案**：事件发生在山林密集区域。

7. **问题**：图像中可以看到多少名消防人员？
   **答案**：图像中没有消防人员。

8. **问题**：可以描述一下烟雾的颜色和类型吗？
   **答案**：烟雾为浓黑色，浓度较高。

9. **问题**：火灾对周围植被有什么影响？
   **答案**：火灾导致周围植被被烧毁。
   
10. **问题**：扑救工作是否已经取得进展？
   **答案**：火势已基本得到控制。

# 示例输入与输出
## 示例 1:
*   输入图像：
*   风险等级：高风险
*   风险代码：HC3
*   输出描述：
【结构化描述】
时间信息：2024 年 05 月 08 日 19:02:03
位置信息：乡村农田区域
事件性质：失控火灾
火情状态：可见明火，有烟雾
烟雾特征：灰白色烟雾
安全评估：火情严重需关注
风险等级和风险代码：高风险 HC3
【短文本描述】
2024年5月8日19时02分许，乡村农田区域发生失控火灾，现场可见明火，伴有灰白色烟雾。风险等级为高风险，情况紧急，需尽快采取灭火措施。
【问答对】
问题：图像中是否存在火灾迹象？是否能在图像中看到明显烟雾或火焰？
答案：能看到明显火焰和灰白色烟雾。
问题：按照 “风险等级定义表”，该情况的风险等级和风险代码是什么？原因是什么？
答案：高风险 HC3。有失控明火，符合高风险特征。
问题：按照火灾发展阶段，当前处于什么阶段？简要描述火情信息。
答案：发展阶段，可见明火和烟雾。
问题：图中描述的主要事件是什么？
答案：乡村农田区域发生火灾。
问题：请问这个事件是在什么时间发生的？
答案：2024 年 05 月 08 日 19:02:03 。
问题：事件发生在什么样的地点？
答案：乡村农田区域。
问题：图像中能看到几个着火点？
答案：至少两个着火点。
问题：可以描述一下烟雾的颜色和类型吗？
答案：烟雾为灰白色，属火灾产生的烟雾。
问题：现场是否有消防人员在处理火情？
答案：图像中未显示消防人员。
问题：火灾可能对周围房屋有什么影响？
答案：可能威胁周围房屋安全。

---

**现在，请根据提供的以下信息分析以下图像，并生成描述：**
*   风险等级：{risk_level}
*   风险代码：{risk_type}
"""

# 全局变量存储结果和已处理文件集合
all_results = []
processed_files_set = set()  # 已处理的 image_path 集合
write_lock = asyncio.Lock()   # 异步锁用于并发安全写入

class BatchProcessUtil:
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    async def process(self, return_exceptions=False):
        results = []
        for i in range(0, len(self.tasks), self.batch_size):
            start = asyncio.get_event_loop().time()
            batch_tasks = self.tasks[i:i + self.batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=return_exceptions)
            results.extend(batch_results)
            print(f"completed: {i + self.batch_size}/{len(self.tasks)}, cost: {int(asyncio.get_event_loop().time() - start)}s")
        return results

    def clear(self):
        self.tasks.clear()

async def save_json(output_json_path):
    global all_results
    async with write_lock:
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"写入 JSON 失败：{str(e)}")

def parse_model_output(output_text):
    """解析模型输出的结构化描述、短文本描述和问答对"""
    # 初始化各部分为空
    structured_desc = {}
    short_desc = ""
    qa_pairs = []
    
    # 尝试解析结构化描述
    structured_match = re.search(r'【结构化描述】\s*(.*?)\s*【短文本描述】', output_text, re.DOTALL)
    if not structured_match:
        # 尝试另一种分隔符
        structured_match = re.search(r'# 结构化描述\s*(.*?)\s*# 短文本描述', output_text, re.DOTALL)
    
    if structured_match:
        structured_text = structured_match.group(1).strip()
        # 提取结构化描述的各个字段
        for line in structured_text.split('\n'):
            if '：' in line:
                key, value = line.split('：', 1)
                structured_desc[key.strip()] = value.strip()
    
    # 尝试解析短文本描述
    short_match = re.search(r'【短文本描述】\s*(.*?)\s*【问答对】', output_text, re.DOTALL)
    if not short_match:
        short_match = re.search(r'# 短文本描述\s*(.*?)\s*# 问答对', output_text, re.DOTALL)
    
    if short_match:
        short_desc = short_match.group(1).strip()
    
    # 尝试解析问答对
    qa_match = re.search(r'【问答对】\s*(.*?)$', output_text, re.DOTALL)
    if not qa_match:
        qa_match = re.search(r'# 问答对\s*(.*?)$', output_text, re.DOTALL)
    
    if qa_match:
        qa_text = qa_match.group(1).strip()
        # 提取问答对
        qa_items = re.findall(r'(\d+)\.\s*\*\*问题\*\*：(.*?)\s*\*\*答案\*\*：(.*?)(?=\n\d+\.|\Z)', qa_text, re.DOTALL)
        for num, question, answer in qa_items:
            qa_pairs.append({
                "question": question.strip(),
                "answer": answer.strip()
            })
    
    return structured_desc, short_desc, qa_pairs

async def process_single_image(llm_util, record, output_json_path):
    global all_results
    try:
        image_path = record["image_path"]
        result = record["result"]
        data_from = record.get("data_from", "")
        
        # 解析风险等级和风险代码
        risk_parts = result.split()
        risk_level = risk_parts[0] if len(risk_parts) > 0 else ""
        risk_type = risk_parts[1] if len(risk_parts) > 1 else ""
        
        # 构建提示词
        prompt_format = prompt.format(
            risk_level=risk_level,
            risk_type=risk_type,
        )
        
        # 调用模型
        model_output = await llm_util.call_doubao(
            prompt_format, 
            [llm_util.path2base64(image_path)]
        )
        
        # 解析模型输出
        structured_desc, short_desc, qa_pairs = parse_model_output(model_output)
        
        # 构建结果记录
        new_record = {
            "image_path": image_path,
            "result": result,
            "data_from": data_from,
            "modelout_description":model_output,
            "structured_description": structured_desc,
            "short_text_description": short_desc,
            "qa_pairs": qa_pairs
        }
        
    except Exception as e:
        print(f"处理失败: {image_path} | 错误: {str(e)}")
        new_record = {
            "image_path": image_path,
            "result": result,
            "data_from": data_from,
            "error": str(e)
        }
    
    async with write_lock:
        all_results.append(new_record)
    await save_json(output_json_path)
    return new_record

async def process_records_concurrently(llm_util, records, output_json_path):
    batch_processor = BatchProcessUtil(batch_size=6)
    global processed_files_set
    
    # 过滤已处理的记录
    todo_records = [r for r in records if r["image_path"] not in processed_files_set]
    print(f"总记录数: {len(records)}, 已处理: {len(records)-len(todo_records)}, 待处理: {len(todo_records)}")
    
    # 添加任务
    for record in todo_records:
        batch_processor.add_task(
            process_single_image(llm_util, record, output_json_path)
        )
        
    # 处理批次
    if batch_processor.tasks:
        await batch_processor.process()
    
    # 最终保存
    await save_json(output_json_path)

async def main():
    parser = argparse.ArgumentParser(description="从JSON读取图像信息并生成风险等级评估结果")
    parser.add_argument("--input_json", default = '/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/标注结果/1w_训练.json', type=str, 
                        help="输入JSON文件路径")
    parser.add_argument("--output_json",  default = '/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/标注结果/1w_训练_短描述.json', type=str,
                        help="输出JSON文件路径")
    args = parser.parse_args()

    global all_results, processed_files_set
    
    # 加载输入JSON
    try:
        with open(args.input_json, 'r', encoding='utf-8') as f:
            input_records = json.load(f)
        print(f"✅ 已加载输入JSON，共 {len(input_records)} 条记录")
    except Exception as e:
        print(f"⚠️ 加载输入JSON失败: {e}")
        return

    # 加载已有输出结果（用于断点续处理）
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            processed_files_set = {item['image_path'] for item in all_results}
            print(f"✅ 已加载已有输出结果 {len(all_results)} 条记录")
        except Exception as e:
            print(f"⚠️ 加载已有输出结果失败: {e}")

    # 初始化模型工具
    llm_util = LlmUtil(sys_content="你是一位图像火灾预警技术专家，需要对上传的图像进行风险等级评判.")

    # 开始处理记录
    await process_records_concurrently(
        llm_util, 
        input_records, 
        args.output_json
    )

    print(f"处理完成，共 {len(all_results)} 条记录，已保存至 {args.output_json}")

if __name__ == '__main__':
    asyncio.run(main())