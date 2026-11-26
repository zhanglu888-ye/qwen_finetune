import asyncio
import os
import json
import pandas as pd
from llm_util import LlmUtil
from tqdm import tqdm

# # 提示词
prompt = """# 任务与核心要求
1.  请**严格且仅**根据图片中**清晰可见的、明确无误的**视觉内容进行分析；仔细观察图中所有存在烟雾的区域。
2.  **仔细阅读并理解**以下“事件”和“烟雾描述”的每一条定义。
3.  **判断事件**：
    *   识别图片中所有**完全符合**“事件”列表中某一项或多项定义的事件。
    *   如果识别出多个事件，请将所有符合的事件编号记录下来。
    *   如果图片内容与事件1-19中的任何一项均**不完全符合**，或者缺乏关键视觉证据，则事件编号为 **0**。
4.  **判断烟雾描述（与事件对应）**：
    *   **为每一个识别出的事件编号，判断其对应的烟雾描述编号。**
    *   如果一个事件本身产生燃烧类烟雾或工业废气（如事件2,3,4,5,7,8,9,19），则根据实际烟雾情况选择烟雾描述1-4。
    *   如果一个事件本身是可见的“类烟雾”现象（如事件10扬尘、事件11雾霾、事件17农药喷洒），请为其选择最合适的烟雾描述编号（例如，扬尘可选3或4，雾霾可选5，农药可选3或4）。
    *   如果一个事件本身不产生烟雾，或图片中该事件区域无烟，或该事件属于非烟雾现象（如事件1消防出动但现场无烟、事件6受控活动无烟、事件12发动机过热无烟、事件13太阳光照、事件14照明设施、事件15镜头干扰、事件16自然景象、事件18阳光反射），则其对应的烟雾描述编号为 **6**。
    *   如果事件编号为 **0** (未观察到明确定义的事件)，则其对应的烟雾描述编号固定为 **6**。
5.  **输出格式要求：**
    *   **你的回答必须严格遵循以下两行格式，且仅包含这两行内容：**
        ```
        事件：[事件编号1],[事件编号2],...
        烟雾描述：[烟雾描述编号1],[烟雾描述编号2],...
        ```
    *   **`事件：`行后是用英文逗号“,”分隔的一个或多个事件编号。如果事件编号为0，则只有0。**
    *   **`烟雾描述：`行后是用英文逗号“,”分隔的一个或多个烟雾描述编号。**
    *   **烟雾描述编号的数量必须与事件编号的数量完全相同，并且按顺序一一对应。例如，第一个烟雾描述编号对应第一个事件编号，第二个烟雾描述编号对应第二个事件编号，以此类推。**
    *   **绝对不要添加任何其他文字、解释、说明、引言或总结。你的整个回复只能是这两行。**
6.  禁止主观臆断、联想或推测图片中未明确展示的内容。如果证据不足或不明确，则倾向于“无匹配”或“来源不明”。
7.  图片中可能存在其他软件叠加的红框、蓝框、圆点、圆圈等非原始图像的标记，请观察一下标记指示的位置是否存在烟雾，该标记可能是误检，请忽略标记本身以避免影响你的判断。

## 事件：
(括号内为该事件典型情况下对应的烟雾描述建议，具体需根据图片判断)
0.  **未观察到明确定义的事件**：图片内容不符合以下1-19中任何一项事件的明确定义，或仅仅是普通场景，没有需要特别关注的烟雾或火情迹象(烟雾描述固定为：6)
1.  消防人员或消防设备出动 (若现场有烟则判断烟雾1-4，若无烟则为6)
2.  山林区域中失控火灾：**必须存在明显可看见的火焰** (烟雾描述为1-4之一)，若不可见火焰，应为19(有疑似火点但不够明确的火焰也应为不可见的火焰)
3.  工业区中失火、仓储/物料堆场燃烧、化学品泄漏燃烧、违规动火作业失控、区域内违规烧烤/祭祀失控 (烟雾描述为1-4之一)
4.  农田/乡村区域中农事用火失控：**必须存在明显可看见的火焰** (烟雾描述为1-4之一)，若不可见火焰，应为19(有疑似火点但不够明确的火焰也应为不可见的火焰)
5.  公共场所/非住宅建筑区域中失控的祭祀用火、失控的烟花爆竹燃放、失控的户外烧烤/篝火 (烟雾描述为1-4之一)
6.  受控的小型烧烤、 有人员看管的小型烟花、有人看管的居民区祭祀用火、日常工具排烟 (若有烟则判断烟雾1-4，若无烟则为6，仅水蒸气则为3或4)
7.  车辆自燃或事故引发的燃烧 (烟雾描述为1-4之一)
8.  电线电缆过载、短路或故障引发的燃烧 (烟雾描述为1-4之一)
9.  **明确的烟囱/排口排烟/排汽**，需要仔细观察图中是否有烟囱，明确看到烟囱且烟雾从烟囱排出才算做该事件，不要对某个位置/结构进行推测，避免推测为烟囱/排烟口 (工业/大型设施/居民区房顶烟囱，烟雾描述为1-4之一，如正常水蒸气可选3或4)
10. 明确为施工、车辆行驶等引起的**扬尘**，必须存在施工设备、车辆且是由其引起的扬尘 (烟雾描述可选3、4，无扬尘可选6)
11. 雾气、霾 (自然天气现象，烟雾描述通常为5)
12. 车辆发动机过热 (若有轻微烟雾则判断烟雾4，若无则为6)
13. 太阳光照 (导致物体反光或区域过曝，烟雾描述为6)
14. 照明设施 (灯光本身或其造成的光晕，烟雾描述为6)
15. 镜头雨珠、镜头脏污 (烟雾描述为6)
16. 云层漂浮、乌云、片云，飞鸟、飞虫群 (烟雾描述为6)
17. 可明确分辨的农药喷洒，必须存在喷洒农药的人或喷洒农药的无人机等设备，仔细观察是否存在人或无人机等设备 (烟雾描述可选3或4)
18. 阳光/灯光反射：屋顶、管道、玻璃、水面、金属表面、路面等 (烟雾描述为6)
19. **烟雾来源不明**：图片中存在烟雾（任意大小，包括轻微烟雾），但无法关联到以上任何已定义事件 (烟雾描述为1-4之一)

## 烟雾描述：
1.  浓黑烟、黄烟、其他异常颜色烟雾(不区分烟雾大小)
2.  黑色、灰色、浓灰色、深灰色烟雾或持续排放的五颜六色工业废气（非正常水蒸气，不区分烟雾大小）
3.  大量白色/浅灰色烟雾（占比大，可包括水蒸气、扬尘、农药）
4.  少量白色或浅灰色烟雾（占比小，可包括水蒸气、扬尘、农药）
5.  大范围弥漫的雾气/霾，能见度降低
6.  **无烟雾** (或对应事件不产生烟雾/类烟雾现象)

## **二次检查项**
1.确认以上事件及烟雾描述后对事件编号为"2.山林区域中失控火灾"和"4.农田/乡村区域中农事用火失控"进行二次检查，确认图中是否有明火，**疑似明火**或**无明火**请将事件调整为"19.烟雾来源不明"
2."19.烟雾来源不明"的请再次仔细观察图片，烟雾是否来自居民区房顶，如果是再检查一下是否可以明确观察到是从烟囱排放的，必须要看到烟囱，不要推测，是的话请调整事件为9

# 输出示例
(请注意，以下示例本身严格遵守了输出格式要求，模型应学习这一点)

## 示例1 (消防出动，现场少量灰烟)
事件：1
烟雾描述：4

## 示例2 (山火，浓黑烟)
事件：2
烟雾描述：1

## 示例3 (普通乡村，无事发生，无烟)
事件：0
烟雾描述：6

## 示例4 (车辆自燃产生浓黑烟 + 消防出动，消防区域无额外烟雾。假设模型识别顺序为消防先，车辆后)
事件：1,7
烟雾描述：6,1

## 示例5 (山火产生浓黑烟 + 远处有少量来源不明的灰色烟雾。假设模型识别顺序为山火先，不明烟雾后)
事件：2,19
烟雾描述：1,4

## 示例6 (工厂烟囱排放大量白色工业蒸汽/烟雾)
事件：9
烟雾描述：3

## 示例7 (工业区厂房失火产生异常颜色烟雾 + 消防出动，消防区域无额外烟雾。假设模型识别顺序为消防先，厂房后)
事件：1,3
烟雾描述：6,1

## 示例8 (消防出动处理已扑灭的小火点，现场无烟)
事件：1
烟雾描述：6

## 示例9 (施工现场大量扬尘)
事件：10
烟雾描述：3

## 示例10 (车辆行驶在土路产生少量扬尘 + 远处山林失火有浓黑烟。假设模型识别顺序为山火先，扬尘后)
事件：2,10
烟雾描述：1,4

## 示例11 (大范围雾霾天气)
事件：11
烟雾描述：5

## 示例12 (阳光强烈反射在金属屋顶，无其他事件)
事件：18
烟雾描述：6

## 示例13 (一辆车发动机过热冒少量白汽，旁边有消防车但无其他烟)
事件：1,12
烟雾描述：6,4

## 示例14 (工业区厂房失火有浓黑烟，同时旁边另一个烟囱在排放大量正常白色水蒸气)
事件：3,9
烟雾描述：1,3
"""









# 配置路径
excel_path = "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级定义测试结果/人工标注结果-all/merged_risk_levels_output_3_0516.xlsx"  # merged_risk_levels_output_1.xlsx
img_dir = "/host/algorithm4/zhanglu/dataset/vlm/fire/fire_数据标注0425"
output_json_path = "/host/algorithm2/zhanglu/code/tmp_0522_ls3_xlsx3.json"


# 全局变量存储结果和已处理文件集合
all_results = []
processed_files_set = set()  # 已处理的 image_path 集合
write_lock = asyncio.Lock()   # 异步锁用于并发安全写入



RISK_LEVEL_ORDER = ["正常", "疑似中风险", "中风险", "疑似高风险", "高风险"]
DEFAULT_RISK = "正常"
RISK_RULES_MAP = {}


def parse_and_assess_risk(llm_output: str) -> str:
    lines = llm_output.strip().split('\n')
    if len(lines) != 2:
        return f"错误：LLM输出应有两行，实际为 {len(lines)} 行. 输出内容: '{llm_output}'"

    event_line = lines[0].strip()
    smoke_line = lines[1].strip()

    if not event_line.startswith("事件：") or not smoke_line.startswith("烟雾描述："):
        return f"错误：LLM输出格式不正确，缺少'事件：'或'烟雾描述：'前缀. 输出内容: '{llm_output}'"

    event_ids_str = event_line[len("事件："):].strip()
    smoke_ids_str = smoke_line[len("烟雾描述："):].strip()

    try:
        if not event_ids_str and not smoke_ids_str:
             event_ids = []
             smoke_ids = []
        elif not event_ids_str:
            return "错误：事件ID为空，但烟雾描述ID不为空"
        elif not smoke_ids_str:
            return "错误：烟雾描述ID为空，但事件ID不为空"
        else:
            event_ids = [int(x.strip()) for x in event_ids_str.split(',')]
            smoke_ids = [int(x.strip()) for x in smoke_ids_str.split(',')]
    except ValueError:
        return f"错误：事件或烟雾描述ID包含非整数值. 事件: '{event_ids_str}', 烟雾: '{smoke_ids_str}'"

    if len(event_ids) != len(smoke_ids):
        return (f"错误：事件ID数量 ({len(event_ids)}) 与烟雾描述ID数量 ({len(smoke_ids)}) 不匹配. "
                f"事件: '{event_ids_str}', 烟雾: '{smoke_ids_str}'")
    if not event_ids:
        return DEFAULT_RISK

    current_max_risk_index = -1
    for event_id, smoke_id in zip(event_ids, smoke_ids):
        risk_for_pair = RISK_RULES_MAP.get((event_id, smoke_id), DEFAULT_RISK)
        try:
            risk_index = RISK_LEVEL_ORDER.index(risk_for_pair)
            if risk_index > current_max_risk_index:
                current_max_risk_index = risk_index
        except ValueError:
            print(f"警告：风险等级 '{risk_for_pair}' (来自事件 {event_id}, 烟雾 {smoke_id}) 未在 RISK_LEVEL_ORDER 中定义。")

    if current_max_risk_index == -1:
        return DEFAULT_RISK
    else:
        return RISK_LEVEL_ORDER[current_max_risk_index]






def collect_images_from_excel(excel_path, img_dir):
    """
    从 xlsx 文件的 '文件名' 列读取图片名称，并组合 img_dir 构建完整路径。
    只保留真实存在的图片路径。
    """
    df = pd.read_excel(excel_path)
    image_paths = []

    for _, row in df.iterrows():
        filename = row['文件名']
        if isinstance(filename, str) and filename.strip() != '':
            full_path = os.path.join(img_dir, filename)
            if os.path.exists(full_path):
                image_paths.append(full_path)

    # 按照文件夹组织（兼容原结构）
    folder_dict = {img_dir: image_paths}
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


        except Exception as e:
            print(f"写入 JSON 失败：{str(e)}")


async def process_single_image(llm_util, image_path):
    global all_results
    try:
        ##不思考
        result = await llm_util.call_doubao(prompt, [llm_util.path2base64(image_path)])        
        data = {'image_path': image_path, 'result': result}
        risk_1 = parse_and_assess_risk(result)
        data = {'image_path': image_path, 'result': risk_1, 'result_model': result}        
        
        ##思考
        # result = await llm_util.call_doubao(prompt, [llm_util.path2base64(image_path)])        
        # data = {'image_path': image_path, 'result': result[0], 'reason': result[1]}  
        # risk_1 = parse_and_assess_risk(result[0])
        # data = {'image_path': image_path, 'result': risk_1, 'result_model': result[0], 'reason': result[1]}
    except Exception as e:
        print(f"处理失败: {image_path} | 错误: {str(e)}")
        data = {'image_path': image_path, 'result': str(e)}

    async with write_lock:
        all_results.append(data)
    await save_json()
    return data


async def process_images_concurrently(llm_util, folder_dict):
    batch_processor = BatchProcessUtil(batch_size=15)
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

    # # 构建已处理集合
    # processed_files_set = set(item['image_path'] for item in all_results)

    # # 加载进度文件（可选）
    # if os.path.exists(progress_file):
    #     try:
    #         with open(progress_file, 'r', encoding='utf-8') as f:
    #             saved_paths = set(json.load(f))
    #         processed_files_set.update(saved_paths)
    #         print(f"✅ 已加载进度文件，共 {len(processed_files_set)} 个文件已处理")
    #     except Exception as e:
    #         print(f"⚠️ 无法加载进度文件: {e}")

    # 初始化模型工具
    llm_util = LlmUtil(sys_content="你是一名专业的图像分析师，专门负责识别可见光图片中的火灾事件和烟雾现象。你的判断必须严格基于图片中清晰可见的视觉证据。")

    # 收集图片（从Excel读取）
    folder_dict = collect_images_from_excel(excel_path, img_dir)

    total_images = len(folder_dict.get(img_dir, []))
    print(f"共发现 {total_images} 张图片，开始并发处理...")

    # 开始处理
    await process_images_concurrently(llm_util, folder_dict)

    print(f"最终处理完成，共 {len(all_results)} 条记录，已保存至 {output_json_path}")


if __name__ == '__main__':
    asyncio.run(main())
