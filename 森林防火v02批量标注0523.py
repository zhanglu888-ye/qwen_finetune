import asyncio
import os
import json
import pandas as pd
from llm_util import LlmUtil
from tqdm import tqdm
import argparse  # 新增：用于命令行参数解析


##提示词10
prompt = """任务：您将接收一张可见光图像。您的任务是根据提供的图像，严格按照以下“风险等级定义”表格中的标准，判断图像所显示的场景对应的最终风险等级和序列号。

**核心判断逻辑与策略：**

## 步骤A：判断是否存在烟雾，并对其进行分类。

A1.  **检查是否为有明确烟囱/排口来源的烟雾 (LC1, MC1)：**
    *   如果能明确识别出烟雾是从“烟囱”或“排口”冒出的：
        *   根据烟雾颜色和形态判断是 **LC1** (正常) 还是 **MC1** (中风险) 。
        *   如果判定为 **LC1** 或 **MC1**，则以此为准，结束判断。

A2.  **如果存在烟雾，但其来源非明确的烟囱/排口 (即非A1情况)，则判断是否为来源不明的烟雾 (HC1、HS1 或 MS1)：**
    *   仔细观察图像中的烟雾，如果其来源不明确或非烟囱/排口：
        *   首先，判断是否符合 **HC1** (高风险) 的特征： 即烟雾是否为浓黑烟、黄烟、其他异常颜色烟雾。如果符合 **HC1** 特征，则判定为 **HC1**，结束判断。
        *   其次，如果烟雾不符合 **HC1** 的特征，然后判断是否符合 **HS1** (疑似高风险) 的特征： 即烟雾是否为大量且较浓的白色/灰色烟雾（视觉上烟雾弥漫范围广、密度高）。如果符合 **HS1** 特征，则判定为 **HS1**，结束判断。
        *   最后，如果烟雾不符合 **HC1** 和 **HS1** 特征，再判断是否符合 **MS1** (疑似中风险) 的特征： 即烟雾为非大量、非浓密的白色/灰色烟雾，通常表现为少量、稀薄，或中等程度但明显未达到 **HS1** 的规模。如果符合 **MS1** 特征，则判定为 **MS1**，结束判断。
        *   对于 **MS1** 的判断： 如果观察到此类来源不明的烟雾，且它不属于明确的 **LC2-LC11** 中定义的特定非燃烧现象（如清晰的扬尘、大范围雾气等），则应考虑 **MS1**。

## 步骤B：若非步骤A现象，则检查LC2-LC11定义的正常现象。

*   检查图像是否符合表格中 **LC2-LC11** **【现象】** 列描述的其他具体、可识别的现象。
*   如果能明确对应到这些现象中的一个或多个：选择风险等级最高的那个作为结果，结束判断。

## 步骤C：判定为“其他正常情况” (LC12)。

*   仅当图像既不符合任何 **【事件】**，也不符合步骤A（所有烟雾情况均已排除或分类）和步骤B（所有其他特定现象均不符合）的描述时， 才最终判定为 **LC12**。
*   **LC12** 意味着图像中没有火灾迹象，也没有任何无法解释的、可疑的烟雾或蒸汽。

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


输入信息：图像数据（一张可见光图像）
输出要求：请仅输出最终判断的风险等级和对应的序列号，格式为：风险等级 序列号例如：高风险 HC1或者正常 LC4
请严格按照上述定义和逻辑进行判断并输出。
"""

# 全局变量存储结果和已处理文件集合
all_results = []
processed_files_set = set()  # 已处理的 image_path 集合
write_lock = asyncio.Lock()   # 异步锁用于并发安全写入


def collect_images_recursively(img_dir):
    """递归收集目录下的所有图片文件"""
    supported_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = []
    
    for root, _, files in os.walk(img_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_ext:
                full_path = os.path.join(root, file)
                image_paths.append(full_path)
    return image_paths

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
    global all_results, processed_files_set
    async with write_lock:
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
            current_paths = {item['image_path'] for item in all_results}
            processed_files_set.update(current_paths)
        except Exception as e:
            print(f"写入 JSON 失败：{str(e)}")

async def process_single_image(llm_util, image_path, output_json_path, data_from):  # 新增data_from参数
    global all_results
    try:
        result = await llm_util.call_doubao(prompt, [llm_util.path2base64(image_path)])        
        data = {
            'image_path': image_path,
            'result': result,
            'data_from': data_from  # 新增字段
        }
    except Exception as e:
        print(f"处理失败: {image_path} | 错误: {str(e)}")
        data = {
            'image_path': image_path,
            'result': str(e),
            'data_from': data_from  # 异常情况也记录来源
        }
    async with write_lock:
        all_results.append(data)
    await save_json(output_json_path)
    return data

async def process_images_concurrently(llm_util, image_paths, output_json_path, data_from):  # 新增data_from参数
    batch_processor = BatchProcessUtil(batch_size=12)
    global processed_files_set
    
    todo_paths = [p for p in image_paths if p not in processed_files_set]
    print(f"总图片数: {len(image_paths)}, 已处理: {len(image_paths)-len(todo_paths)}, 待处理: {len(todo_paths)}")
    
    for image_path in todo_paths:
        # 传递data_from参数
        batch_processor.add_task(
            process_single_image(llm_util, image_path, output_json_path, data_from)
        )
        
    if batch_processor.tasks:
        await batch_processor.process()
    await save_json(output_json_path)

async def main():
    parser = argparse.ArgumentParser(description="批量处理图像并生成风险等级评估结果")
    parser.add_argument("--img_dir", required=True, type=str, 
                        help="图片所在的目录路径")
    parser.add_argument("--output_json_path", required=True, type=str,
                        help="输出JSON文件的保存路径")
    # 新增data_from参数
    parser.add_argument("--data_from", required=True, type=str,
                        help="数据来源标识（如采集设备ID/项目名称等）")
    args = parser.parse_args()

    global all_results, processed_files_set
    # 加载已有结果
    if os.path.exists(args.output_json_path):
        try:
            with open(args.output_json_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            processed_files_set = {item['image_path'] for item in all_results}
            print(f"✅ 已加载已有 {len(all_results)} 条记录")
        except Exception as e:
            print(f"⚠️ 无法加载已有结果文件: {e}")

    # 初始化模型工具
    llm_util = LlmUtil(sys_content="你是一位图像火灾预警技术专家，需要对上传的图像进行风险等级评判.")

    # 收集图片
    image_paths = collect_images_recursively(args.img_dir)
    print(f"共发现 {len(image_paths)} 张图片，开始并发处理...")

    # 开始处理
    await process_images_concurrently(
        llm_util, 
        image_paths, 
        args.output_json_path,
        args.data_from  # 新增参数
    )

    print(f"最终处理完成，共 {len(all_results)} 条记录，已保存至 {args.output_json_path}")

if __name__ == '__main__':
    asyncio.run(main())