import asyncio
import os
import json
from llm_util import LlmUtil
from tqdm import tqdm

# prompt = """你的主要任务是准确识别并提取图片中包含的特定结构化信息，特别是日期、时间、星期、地理位置和图像类型。同时，你还需要捕获并列出图片中所有其他可见的文本内容，无论其格式或位置如何。
# 如果某些特定信息（如日期）在图片中未找到，请明确指出（例如，使用“未识别”或“N/A”）。
# 图片中通常包含拍摄信息，格式可能类似于：“2023-04-18 18:06:55 星期二、夷陵区雾渡河镇龚家河村2组（热成像）”，但也可能包含其他文字。
# 请提取并分别列出：
# - 日期：
# - 时间：
# - 星期：
# - 地点：
# - 图像类型：
# - 包含上述信息的原始文本条目 (如果存在)：
# - 图片中的所有其他文本：

# 例如，对于一张包含“2023-04-18 18:06:55 星期二、夷陵区雾渡河镇龚家河村2组（热成像）”以及角落里有“设备编号: XYZ123”的图片，提取结果可能为：
# - 日期：2023-04-18
# - 时间：18:06:55
# - 星期：星期二
# - 地点：夷陵区雾渡河镇龚家河村2组
# - 图像类型：热成像
# - 包含上述信息的原始文本条目：2023-04-18 18:06:55 星期二、夷陵区雾渡河镇龚家河村2组（热成像）
# - 图片中的所有其他文本：设备编号: XYZ123"""


# 提示词
prompt = """你的主要任务是准确识别并提取图片中包含的图片类型（可见光or热成像），如果在图片中未找到，请明确指出（例如，使用“未识别”或“N/A”）。
例如，对于一张包含“2023-04-18 18:06:55 星期二、夷陵区雾渡河镇龚家河村2组（热成像）”以及角落里有“设备编号: XYZ123”的图片，提取结果可能为：
- 图像类型：热成像
"""

# 配置路径
img_dir = "/host/algorithm4/zhanglu/dataset/vlm/fire/yichang_60w/按照秒分类_reorganized/2023"
output_json_path = "/host/algorithm4/zhanglu/dataset/vlm/fire/yichang_60w/按照秒分类_reorganized/output_results_2023.json"
progress_file = "/host/algorithm4/zhanglu/dataset/vlm/fire/yichang_60w/按照秒分类_reorganized/progress_2023.json"


# 全局变量存储结果和已处理文件集合
all_results = []
processed_files_set = set()  # 已处理的 image_path 集合
write_lock = asyncio.Lock()   # 异步锁用于并发安全写入


def collect_images(root_dir):
    """
    收集所有图片路径，按文件夹组织
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    folder_dict = {}

    for root, dirs, files in os.walk(root_dir):
        images = []
        for file in files:
            if file.lower().endswith(image_extensions):
                images.append(os.path.join(root, file))
        if images:
            folder_dict[root] = images

    return folder_dict


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


async def save_json():
    global all_results, processed_files_set
    async with write_lock:
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)

            # 更新已处理集合
            current_paths = {item['image_path'] for item in all_results}
            processed_files_set.update(current_paths)

            # 保存进度文件
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(list(processed_files_set), f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"写入 JSON 失败：{str(e)}")


async def process_single_image(llm_util, image_path):
    global all_results
    try:
        result = await llm_util.call_qwen(prompt, [llm_util.path2base64(image_path)])
        data = {'image_path': image_path, 'result': result}
    except Exception as e:
        print(f"处理失败: {image_path} | 错误: {str(e)}")
        data = {'image_path': image_path, 'result': str(e)}

    async with write_lock:
        all_results.append(data)
    await save_json()
    return data


async def process_images_concurrently(llm_util, folder_dict):
    batch_processor = BatchProcessUtil(batch_size=8)
    global processed_files_set

    for folder_path, image_paths in folder_dict.items():
        for image_path in image_paths:
            if image_path in processed_files_set:
                continue
            batch_processor.add_task(process_single_image(llm_util, image_path))

    if batch_processor.tasks:
        await batch_processor.process()

    await save_json()


async def main():
    global all_results, processed_files_set

    # 如果存在旧结果文件，加载进来
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            print(f"✅ 已加载已有 {len(all_results)} 条记录")
        except Exception as e:
            print(f"⚠️ 无法加载已有结果文件: {e}")

    # 构建已处理集合
    processed_files_set = set(item['image_path'] for item in all_results)

    # 加载进度文件（可选）
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                saved_paths = set(json.load(f))
            processed_files_set.update(saved_paths)
            print(f"✅ 已加载进度文件，共 {len(processed_files_set)} 个文件已处理")
        except Exception as e:
            print(f"⚠️ 无法加载进度文件: {e}")

    # 初始化模型工具
    llm_util = LlmUtil(sys_content="你是一个专门用于从图片中提取文本信息的AI助手，具备强大的OCR（光学字符识别）能力。")

    # 收集图片
    folder_dict = collect_images(img_dir)
    total_folders = len(folder_dict)
    print(f"共发现 {total_folders} 个文件夹，开始并发处理...")

    # 开始处理
    await process_images_concurrently(llm_util, folder_dict)

    print(f"最终处理完成，共 {len(all_results)} 条记录，已保存至 {output_json_path}")


if __name__ == '__main__':
    asyncio.run(main())