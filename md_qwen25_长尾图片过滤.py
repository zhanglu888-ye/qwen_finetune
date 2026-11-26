import os
import shutil
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
# qwen_vl_utils.py 应该和你的主脚本在同一个目录下
from qwen_vl_utils import process_vision_info 

# --- 1. 配置区域 ---
# 请根据你的实际情况修改这些路径
INPUT_DATA_DIR = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/test_data_quchong_tmp"  # 包含31个类别文件夹的根目录
OUTPUT_KEPT_DIR = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/test_data_quchong_过滤_v2"   # 保留的图片存放处
OUTPUT_FILTERED_DIR = "/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/test_data_quchong_filtered_out_v2" # 过滤掉的图片存放处
# LOCAL_MODEL_PATH = "/host/ssddata/disk2/zhanglu/md_models/models/Qwen/Qwen2___5-VL-72B-Instruct"
LOCAL_MODEL_PATH = "/host/ssddata/disk1/wyd/model/qwen/qwen2.5-vl-3b-instruct"

# --- 2. 提示词模板 ---
PROMPT_TEMPLATE = """
你是一位极其严谨、注重细节的图像内容审核专家。你的任务是基于“真实性”和“内容相关性”两个核心标准，对图片进行分类。

**## 核心标准**
1.  **真实性**: 图片必须是真实世界的场景。可以是普通照片、监控截图、新闻图片或视频截图。**绝对不能是手绘、卡通、漫画、3D渲染或任何形式的电脑合成艺术图**。
2.  **内容相关性**: 图片必须清晰、明确地展示“**{category}**”这一事件或场景。如果图片内容模糊、有歧义或与类别定义不符，则视为不相关。

**## 判断逻辑示例**
下面是一些判断示例，请学习我的判断逻辑：

---
**示例1**
*   **任务**: 判断一张关于“城市火灾”的图片。
*   **假设图片内容是**: 一张从远处拍摄的、城市高楼冒出滚滚浓烟的照片，能看到消防车在楼下。
*   **分析过程**:
    1.  真实性检查：这是真实照片，通过。
    2.  内容相关性检查：清晰地展示了城市背景下的火灾，通过。
*   **结论**: 是

---
**示例2**
*   **任务**: 判断一张关于“城市火灾”的图片。
*   **假设图片内容是**: 一张森林大火的图片，背景是山和树木，没有城市建筑。
*   **分析过程**:
    1.  真实性检查：这是真实照片，通过。
    2.  内容相关性检查：这是森林火灾，不是“城市火灾”，不通过。
*   **结论**: 否

---
**示例3**
*   **任务**: 判断一张关于“拉横幅”的图片。
*   **假设图片内容是**: 一张卡通人物拉着庆祝横幅的漫画。
*   **分析过程**:
    1.  真实性检查：这是卡通漫画，不通过。
*   **结论**: 否

---

**## 你的任务**
现在，请严格按照上述标准和示例逻辑，判断下面这张关于“**{category}**”的真实图片。

请只回答“是”或“否”，不要输出任何其他内容。
"""

# --- 3. 模型加载 ---
print(f"正在从本地路径加载模型: {LOCAL_MODEL_PATH}")
if not os.path.isdir(LOCAL_MODEL_PATH):
    raise FileNotFoundError(f"模型目录不存在，请检查路径: {LOCAL_MODEL_PATH}")

model = AutoModelForVision2Seq.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True
)
print("模型加载完成。")

# --- 4. 核心处理函数 ---
def judge_image(image_path: str, category: str) -> bool:
    """
    使用Qwen-VL模型判断一张图片是否符合要求。
    返回 True 表示保留，False 表示过滤。
    """
    prompt_text = PROMPT_TEMPLATE.format(category=category)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    try:
        # 准备推理输入
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # 推理
        generated_ids = model.generate(**inputs, max_new_tokens=10) # 只需要生成"是"或"否"，10个token足够
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 解析模型输出
        # 使用 .strip() 去除前后空格，使用 startswith 判断，更健壮
        if output_text.strip().startswith("是"):
            return True
        else:
            return False

    except Exception as e:
        print(f"\n处理图片 {image_path} 时发生错误: {e}")
        return False # 发生错误时，默认也过滤掉

# --- 5. 主执行流程 ---
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_KEPT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FILTERED_DIR, exist_ok=True)

    # 获取所有类别文件夹
    try:
        category_folders = [d for d in os.listdir(INPUT_DATA_DIR) if os.path.isdir(os.path.join(INPUT_DATA_DIR, d))]
        if not category_folders:
            print(f"错误：在 '{INPUT_DATA_DIR}' 目录下没有找到任何类别子文件夹。")
            return
    except FileNotFoundError:
        print(f"错误：输入目录 '{INPUT_DATA_DIR}' 不存在。")
        return

    print(f"找到 {len(category_folders)} 个类别，开始处理...")

    # 遍历每个类别文件夹
    for category in tqdm(category_folders, desc="总进度"):
        source_category_path = os.path.join(INPUT_DATA_DIR, category)
        kept_category_path = os.path.join(OUTPUT_KEPT_DIR, category)
        filtered_category_path = os.path.join(OUTPUT_FILTERED_DIR, category)
        
        os.makedirs(kept_category_path, exist_ok=True)
        os.makedirs(filtered_category_path, exist_ok=True)

        image_files = [f for f in os.listdir(source_category_path) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif'))]
        
        # 遍历类别下的每张图片
        for image_name in tqdm(image_files, desc=f"处理中: {category}", leave=False):
            image_path = os.path.join(source_category_path, image_name)
            
            # 检查图片是否可读
            try:
                img = Image.open(image_path)
                img.close()
            except (IOError, Image.UnidentifiedImageError):
                print(f"\n图片损坏或无法打开，直接过滤: {image_path}")
                shutil.move(image_path, os.path.join(filtered_category_path, image_name))
                continue

            # 获取模型判断
            should_keep = judge_image(image_path, category)

            # 移动文件
            if should_keep:
                destination_path = os.path.join(kept_category_path, image_name)
            else:
                destination_path = os.path.join(filtered_category_path, image_name)
            
            shutil.move(image_path, destination_path)

    print("\n所有图片处理完成！")
    print(f"保留的图片位于: {OUTPUT_KEPT_DIR}")
    print(f"过滤的图片位于: {OUTPUT_FILTERED_DIR}")

if __name__ == "__main__":
    main()
