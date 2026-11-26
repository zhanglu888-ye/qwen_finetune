import asyncio
import os
import json
import pandas as pd
from llm_util import LlmUtil
from tqdm import tqdm
import argparse  # 新增：用于命令行参数解析
import re
from PIL import Image


prompt_bigimg_mask = """**## 角色与目标 (Role and Goal)**

你是一位专业的图像数据标注专家和资深新闻记者。你的任务是为一张包含红色轮廓线标注人物的复杂场景图片，生成一段结构化、详细且客观的描述文本。你的描述需要像一篇精确的新闻现场报道，既有宏观概览，也有微观细节。

**## 输入 (Input)**

你将收到一张图片。图片特点如下：
1.  是一张信息量丰富的场景大图。
2.  图中一个或多位重要人物的身体轮廓被**红色线条**清晰地勾勒出来。这些是你需要重点描述的对象。

**## 输出要求 (Output Requirements)**

请严格按照以下三部分结构和格式输出你的描述。使用Markdown格式，确保最终输出内容清晰、易读。

---

**### 第一部分：简介概述 (TLDR)**

*   **风格**：新闻报道的导语（Lead paragraph）风格。
*   **内容**：用一到两句高度凝练的话，总结整张图片最核心的事件或场景。这部分应作为全文的“内容提要”（TLDR）。
*   **要求**：客观、简洁、信息量大，让读者能迅速了解图片高层次的内容。

**### 第二部分：整体图像描述**

*   **风格**：客观、中立的事实陈述。
*   **内容**：详细描述图片的整体环境和背景。请包含但不限于以下元素：
    *   **环境类型**：是在城市街道、室内空间、自然景观、事故现场还是其他特定场景？
    *   **关键背景元素**：描述除重点人物外的建筑物、其他车辆、植被、天空、道路状况等。
    *   **光线与氛围**：描述当时的光照条件（如白天、黄昏、夜晚、晴天、阴天）和整体氛围（如紧张、平和、繁忙、萧条）。
    *   **其他非重点人物或物体**：简要提及图片中未被红色轮廓线标注的其他人物或值得注意的物体。

**### 第三部分：重点人物详细描述**

*   **核心原则**：这是描述的重点。你需要对所有被**红色轮廓线**勾勒出的人物进行详尽的描述。
*   **描述顺序**：请遵循**显著性原则**。从画面中最重要或最引人注目的人物开始描述，然后依次描述其他人物。显著性可以依据以下几点判断：
    1.  位于画面视觉中心或黄金分割点。
    2.  处于关键互动或动作的焦点。
    3.  在画面中占比最大或最靠近镜头。

*   **对每个重点人物的描述，请务必包含以下信息：**
    1.  **个体信息**：
        *   **编号与位置**：为了清晰，请在描述第一个人物时称其为“人物A”，第二个为“人物B”，以此类推。精确描述其在画面中的绝对位置（例如，“人物A位于画面中央偏左的前景位置”）和相对位置（例如，“人物B站在人物A身后约两米处”）。
        *   **外貌与着装**：描述人物的性别、大致年龄段、发型、面部特征（如是否佩戴眼镜、胡须），以及身上穿着衣物的颜色、款式和特点。
        *   **姿态与动作**：描述人物的身体姿态（站立、坐着、弯腰、奔跑）、手势、视线方向以及正在进行的具体动作（如交谈、打电话、指向某物、操作设备）。
        *   **情绪与表情**：基于可观察到的面部表情，客观描述其可能的情绪状态（如表情严肃、面带微笑、看起来很惊恐或专注）。
    2.  **关系描述（至关重要）**：
        *   **人物之间**：清晰阐述重点人物之间的空间关系和互动关系。他们是在交谈、协作、对峙，还是彼此独立？例如：“人物A与人物B正面对面交谈，两者距离很近，目光有交流。”或“人物C背对人物D，两者无明显互动。”
        *   **人物与车辆/关键物体**：描述人物与图中车辆或其他关键物体（如工具、行李箱、设备）的互动。例如：“人物A正倚靠在一辆蓝色的轿车车门上。”或“人物B手中提着一个黑色的工具箱。”
        *   **人物与环境**：描述人物如何融入或影响周围环境。例如：“人物们正站在建筑物的阴影下避雨。”或“人物C正在横穿斑马线。”

---

**## 输出格式模板 (Output Format Template)**

请严格使用以下Markdown模板生成最终描述文本：

```markdown
### 简介概述 (TLDR)
[在这里填写1-2句的新闻导语式总结]

### 整体图像描述
[在这里填写对整个场景、环境、氛围和背景元素的详细描述]

### 重点人物详细描述

**人物A**:
*   **个体描述**: [位于画面...位置。该人物...（外貌着装描述）。其姿态为...，正在进行...（动作描述）。面部表情...]
*   **关系描述**: [与人物B...（人物间关系）。该人物与...车辆/物体...（人与物关系）。他/她正处于...环境中（人与环境关系）。]

**人物B**:
*   **个体描述**: [位于人物A的...方向。该人物...（外貌着装描述）。其姿态为...，正在进行...（动作描述）。面部表情...]
*   **关系描述**: [与人物A...（人物间关系）。该人物与...车辆/物体...（人与物关系）。他/她正处于...环境中（人与环境关系）。]

(如果还有其他被框选的人物，请继续添加人物C, D...)
```

---

**## 示例参考 (Reference Examples)**

以下是三个高质量的描述示例，请仔细研究并模仿其风格、结构和详细程度。

### **示例一：居民区巷道场景**

### 简介概述 (TLDR)
监控画面显示，在一条停满车辆的潮湿居民区巷道内，三名行人正各自朝不同方向行进，展现了典型的社区日间生活场景。

### 整体图像描述
该图像由高角度监控摄像头拍摄，记录了一个狭窄的居民区巷道。巷道两旁紧密停放着多辆私家车，包括一辆白色SUV、一辆黑色轿车以及一辆被银色车衣覆盖的车辆。地面为水泥地，有多处积水，表明可能刚刚下过雨或地面潮湿。左侧是居民楼，右侧是一堵砖墙。光线为白天自然光，整体氛围平静，呈现日常生活的景象。远景处可以看到巷道延伸，并有其他模糊的人影。

### 重点人物详细描述

**人物A**:
*   **个体描述**: 位于画面右侧前景，是一位中年女性。她身穿一件带有蝴蝶图案的黑色短袖T恤和黑色裤子，斜挎着一个蓝棕色的小包。她的视线正向下注视着手中的手机，表情专注。她正朝着摄像头的方向走来。
*   **关系描述**: 人物A在三位被标注者中离镜头最近。她与其他人没有明显的互动，完全沉浸在自己的手机世界中。她正行走在巷道右侧的潮湿路面上。

**人物B**:
*   **个体描述**: 位于画面中央，走在人物A的左后方。这是一位年轻女性，看似学生，上身穿着一件蓝白相间的运动校服外套，内搭白色上衣，下穿深色长裤和白色运动鞋。她的视线略微偏向自己的左前方，表情自然，正在稳步前行。
*   **关系描述**: 人物B与人物A和人物C在空间上接近，但没有发生直接交流。她正沿着巷道中央的路径向镜头方向走来，与人物C的行进方向相反。

**人物C**:
*   **个体描述**: 位于人物B的后方偏左位置，是一位戴着眼镜的中年男性。他身穿一件灰色拉链外套和深色裤子。他正用右手拉着一个绿色的可折叠购物手拉车，车内似乎装有物品。他正背对镜头，向巷道深处走去。
*   **关系描述**: 人物C与人物A和B的行进方向相反，彼此没有互动。他正在停放的车辆之间留出的通道中行走，其携带的购物手拉车表明他可能刚购物归来或正要去购物。

---
### **示例二：小区出入口场景**

### 简介概述 (TLDR)
监控画面捕捉到一名年轻女性在查看手机的同时，正从一处设有道闸和岗亭的小区门口走出，背景中一名工作人员正在站岗。

### 整体图像描述
图像展示了一个小区或停车场的出入口场景。地面为大片水泥地，部分区域有裂缝。左侧设有一个保安岗亭和一处红白相间的自动道闸，道闸目前处于开启状态，允许通行。右侧是一栋建筑的墙体和一个棕色大门，门上贴有“值班室闲人免进”的告示。远景处可以看到一条马路，有车辆正在行驶。整体光线充足，为白天的户外环境。

### 重点人物详细描述

**人物A**:
*   **个体描述**: 位于画面中心前景位置，是一位正在行走的年轻女性。她留着长发，佩戴眼镜，身穿一件白色外套，内搭深色上衣和深色长裤，脚穿白色鞋子。她的左臂下夹着一个蓝绿色的袋子或靠垫。她双手捧着一部手机，正低头专注地看着屏幕。
*   **关系描述**: 人物A是画面的视觉焦点，正从门禁区域向外走。她与背景中的人物B没有互动，注意力完全集中在手机上。她正走在出入口的中央通道上。

**人物B**:
*   **个体描述**: 位于画面背景处，站在道闸附近。这是一位男性，身穿一件黄色的反光背心，表明其可能是保安或现场工作人员。他站在道闸控制杆的后方，身体略微侧向左边，似乎也在查看手中的设备或保持警戒。
*   **关系描述**: 人物B处于工作岗位上，其位置与道闸和岗亭直接相关，构成了该出入口的管理元素。他与正在走出的人物A没有发生交流，两人处于各自独立的状态，共同构成了“行人通过岗哨”这一场景。

---
### **示例三：居民院落遛狗场景**

### 简介概述 (TLDR)
监控画面记录了在一个居民区院落内，一对男女正带着一条宠物狗散步，而另一名女子则从背景中走过，构成一幅生活化的场景。

### 整体图像描述
该图像从高处俯拍了一个居民楼前的院落或通道。地面是由大块方形水泥砖铺成，上面有明显的裂缝。左上角背景处可见一栋居民楼的入口和一些杂物，包括一个大型蓝色垃圾箱。右侧是一道黑色的金属栅栏门，门后停放着一辆白色汽车。右前景是建筑物的墙壁，上面贴有社区公告。整体光线为白天，场景氛围日常而闲适。

### 重点人物详细描述

**人物A**:
*   **个体描述**: 位于画面左侧前景，是一位戴眼镜的年轻女性，头发盘起。她身穿一件黑色连帽外套和浅色（米白或浅灰）阔腿裤。她正用双手紧握牵引绳，身体前倾，正在遛一条中等体型的、黑白灰三色花斑的狗。
*   **关系描述**: 人物A是遛狗活动的主导者，与人物B并排行走，两者关系亲密，似乎是同伴。她和她的狗是画面的主要动态元素，正从画面中央向左下方移动。

**人物B**:
*   **个体描述**: 位于人物A的右后方，是一位年轻男性。他穿着一件灰色与黑色拼接的长袖上衣和深色裤子。他身体侧向人物A，视线似乎落在人物A或狗身上，左手提着一个黑色袋状物。
*   **关系描述**: 人物B与人物A构成一个同行的组合，他们之间的近距离和身体朝向表明二人正在互动或交谈。他作为陪伴者，与人物A和狗共同组成了场景的焦点。

**人物C**:
*   **个体描述**: 位于画面左上方的背景区域，是一位留着长发的女性。她身穿一件浅粉色或米色的上衣和黑色长裤，左手提着一个白色手提袋。她正背对镜头，沿着一条斜坡向上行走。
*   **关系描述**: 人物C与前景的人物A和B没有互动，是一个独立的路人。她的存在丰富了画面的生活气息，显示出这是一个有其他居民活动的公共空间。她正从院落深处走向居民楼方向。

---

**## 最终检查 (Final Check)**

在生成最终文本前，请确认：
- 所有被红色轮廓线标注的人物都已被描述。
- 描述顺序遵循了显著性原则。
- 事实陈述保持客观中立，避免主观臆断。
- 格式完全符合上述模板要求，并且质量能达到示例水平。
"""

# 全局变量存储结果
all_results = []
processed_images = set()
write_lock = asyncio.Lock()  # 异步锁用于并发安全写入

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
            print(f"Completed: {i + self.batch_size}/{len(self.tasks)}, Cost: {int(asyncio.get_event_loop().time() - start)}s")
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
            print(f"Failed to write JSON: {str(e)}")

async def process_single_image(llm_util, image_path, output_json_path):
    """处理单张图片"""
    global all_results
    
    try:
        # 调用模型处理图片
        model_output = await llm_util.call_gemini(
            prompt_bigimg_mask, 
            [llm_util.path2base64(image_path)]
        )
        
        # 构建结果记录
        new_record = {
            "image_path": image_path,
            "model_output": model_output
        }
        
    except Exception as e:
        print(f"Processing failed: {image_path} | Error: {str(e)}")
        new_record = {
            "image_path": image_path,
            "error": str(e)
        }
    
    async with write_lock:
        all_results.append(new_record)
    await save_json(output_json_path)
    return new_record

async def process_images_concurrently(llm_util, image_folder, output_json_path):
    """并发处理图片文件夹中的所有图片"""
    batch_processor = BatchProcessUtil(batch_size=8)
    global processed_images
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(image_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 为每张图片创建处理任务
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, image_file)
        
        if image_path not in processed_images:
            batch_processor.add_task(
                process_single_image(llm_util, image_path, output_json_path))
            processed_images.add(image_path)
    
    # 处理批次
    if batch_processor.tasks:
        await batch_processor.process()
    
    # 最终保存
    await save_json(output_json_path)

async def main():
    parser = argparse.ArgumentParser(description="处理图片并生成分析结果")
    parser.add_argument("--image_folder", default="/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/sam1_bigimg_mulbox",
                       help="包含图片的文件夹路径")
    parser.add_argument("--output_json", default="/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/大图描述_0627.json",
                       help="输出JSON文件路径")
    args = parser.parse_args()

    global all_results, processed_images
    
    # 检查文件夹是否存在
    if not os.path.isdir(args.image_folder):
        print(f"Error: Image folder not found {args.image_folder}")
        return
    
    # 加载已有输出结果（用于断点续处理）
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            processed_images = {item['image_path'] for item in all_results if 'image_path' in item}
            print(f"✅ Loaded existing results: {len(all_results)} records")
        except Exception as e:
            print(f"⚠️ Failed to load existing results: {e}")

    # 初始化模型工具
    llm_util = LlmUtil(sys_content="You are a multimodal AI analysis expert.")

    # 处理图片
    await process_images_concurrently(
        llm_util, 
        args.image_folder,
        args.output_json
    )
    print(f"Processing completed. Total {len(all_results)} records saved to {args.output_json}")

if __name__ == '__main__':
    asyncio.run(main())