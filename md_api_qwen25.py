import os
import json
import base64
import time
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# --- 1. 配置区域 ---

# 在这里填入你从 ModelScope 个人中心获取的真实 API Key
# 获取地址: https://modelscope.cn/my/myaccesstoken
MS_API_KEY = "ms-5acdd5f7-23ad-49eb-9bee-842863de3578"

# API 和模型配置
BASE_URL = 'https://api-inference.modelscope.cn/v1'
MODEL_ID = 'qwen/Qwen2.5-VL-72B-Instruct' # 这里使用Chat模型ID，它同样支持Instruct指令

# 数据和输出目录配置
DATA_DIR = Path("/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/test_data_quchong_过滤_v2")
OUTPUT_DIR = Path("/host/ssddata/disk2/zhanglu/clip-data/datasets/changwei/test_data_quchong_过滤_v2_result_qwen")
MAX_IMAGES_PER_CLASS = 80 # 每个类别最多测试的图片数量

# --- 2. 专家级提示词 ---
# 这是我们之前设计的最佳提示词
EXPERT_PROMPT = """
# 角色
你是一位精通图像识别的“城市综合治理AI分析专家”。

# 任务
你的核心任务是分析我提供的图片，并将其精确地归类到以下预定义的类别中。

# 类别列表及关键定义
- 井盖缺失: 街道或路面上井盖丢失或严重破损，露出洞口。
- 人员踩踏: 指人踩踏草坪、花坛等禁止进入的绿地之外的公共设施或区域。
- 城市火灾: 发生在城市建筑、居民区或公共设施的火灾。
- 摊贩占道经营: 小商贩在人行道、车行道等非指定区域摆摊设点。
- 树木倾倒: 树木因天气或自身原因倒下，可能阻碍交通或造成危险。
- 水位过高: 河流、湖泊、城市排水系统水位异常上涨，有漫溢风险。
- 消防通道占用: 车辆、货物或任何障碍物堵塞了标有“消防通道”的区域。
- 电动车入楼: 电动车或其电池被带入电梯、楼道等建筑内部空间。
- 翻墙越界: 有人正在或已经翻越围墙、栅栏等屏障。
- 遛狗不牵绳: 在公共场所遛狗但未使用牵引绳。
- 非机动车乱停乱放: 自行车、电动车等随意停放，妨碍通行。
- 黑臭水体: 水体呈现黑色或深色，明显浑浊，可能伴有垃圾，暗示水质严重污染。
- 人员聚集: 大量人群在非正常情况下密集聚集，可能存在安全隐患。
- 地面坍塌: 道路或地面出现明显的塌陷或坑洞。
- 山体滑坡: 山坡上的土石整体向下滑动。
- 机动车乱停乱放: 汽车等机动车未按规定停放在停车位内。
- 森林火灾: 发生在森林或大面积林地的火灾。
- 河道垃圾漂浮物: 河流、沟渠水面上有明显的垃圾、废弃物漂浮。
- 游行: 人群有组织地在街道上行进，通常带有标语或口号。
- 秸秆焚烧: 在农田或开阔地带燃烧农作物秸秆，产生大量烟雾。
- 草坪踩踏: 专指有人在草坪上行走或活动，对草地造成破坏。
- 醉酒呕吐: 有人表现出醉酒状态并在公共场所呕吐。
- 非法捕鱼: 使用违禁工具（如电鱼、网鱼）或在禁渔区进行捕捞。

# 输出格式
请严格按照以下 JSON 格式提供你的最终结论，不要添加任何额外的解释或代码块标记。
{
  "category": "最终选择的类别名称",
  "confidence": "（高/中/低）",
  "reason": "（简要说明你为什么选择这个类别）"
}

---
现在，请分析这张图片。
"""

# --- 3. 辅助函数 ---

def image_to_base64(image_path: Path) -> str:
    """将图片文件转换为 Base64 编码的字符串"""
    try:
        with image_path.open("rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return ""

def call_vision_api(client: OpenAI, image_path: Path) -> dict | None:
    """调用大模型 API 并返回解析后的 JSON 结果"""
    base64_image = image_to_base64(image_path)
    if not base64_image:
        return None

    image_url = f"data:image/jpeg;base64,{base64_image}"

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': EXPERT_PROMPT},
                    {'type': 'image_url', 'image_url': {'url': image_url}},
                ],
            }],
            # stream=False is default, no need to set
        )
        
        # 提取模型返回的 JSON 字符串
        model_output_str = response.choices[0].message.content
        
        # 解析 JSON 字符串
        # 模型有时会返回被 markdown 包裹的json，例如 ```json\n{...}\n```，需要清理
        if "```json" in model_output_str:
            model_output_str = model_output_str.split("```json\n")[1].split("\n```")[0]
            
        return json.loads(model_output_str)

    except json.JSONDecodeError:
        print(f"\n[错误] 模型返回的不是有效的 JSON 格式: {model_output_str}")
        return {"category": "json_parse_error", "reason": model_output_str}
    except Exception as e:
        print(f"\n[错误] API 调用失败: {e}")
        return None


# --- 4. 主逻辑 ---

def main():
    """主执行函数"""
    # 初始化 OpenAI 客户端
    client = OpenAI(base_url=BASE_URL, api_key=MS_API_KEY)

    # 准备输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 获取所有类别文件夹
    if not DATA_DIR.exists():
        print(f"错误: 数据目录 '{DATA_DIR}' 不存在。")
        return
    
    categories = [d for d in DATA_DIR.iterdir() if d.is_dir()]
    if not categories:
        print(f"错误: 在 '{DATA_DIR}' 中没有找到任何类别文件夹。")
        return

    print(f"找到 {len(categories)} 个类别，将开始测试...")
    
    all_results = []
    total_api_calls = 0

    # 使用 tqdm 创建总进度条
    for category_dir in tqdm(categories, desc="总进度"):
        true_label = category_dir.name
        
        # 为每个类别创建输出子目录
        (OUTPUT_DIR / true_label).mkdir(exist_ok=True)
        
        # 获取该类别下的所有图片文件
        image_files = sorted([f for f in category_dir.glob('*') if f.suffix.lower() in ('.jpg', '.jpeg', '.png')])
        
        # 限制每个类别的图片数量
        images_to_test = image_files[:MAX_IMAGES_PER_CLASS]
        total_api_calls += len(images_to_test)

        for image_path in tqdm(images_to_test, desc=f"处理类别 [{true_label}]", leave=False):
            # 调用 API
            result_json = call_vision_api(client, image_path)
            
            # 如果调用成功
            if result_json:
                # 保存 JSON 结果
                
                result_json["source_image_path"] = str(image_path.resolve())
                
                output_json_path = OUTPUT_DIR / true_label / f"{image_path.stem}.json"
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_json, f, ensure_ascii=False, indent=4)
                
                # 记录结果用于最终统计
                predicted_label = result_json.get("category", "预测失败")
                all_results.append({
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "image": str(image_path)
                })
            else:
                 # 记录失败的结果
                 all_results.append({
                    "true_label": true_label,
                    "predicted_label": "api_call_failed",
                    "image": str(image_path)
                })
            
            # API 调用之间短暂休息，防止过于频繁触发限流
            time.sleep(1) 

    print(f"\n所有图片处理完毕，总共进行了 {len(all_results)} 次 API 调用。")
    generate_accuracy_report(all_results)


def generate_accuracy_report(results: list):
    """根据结果生成并打印准确率报告"""
    if not results:
        print("没有可供分析的结果。")
        return

    total_count = len(results)
    correct_count = 0
    
    class_stats = {} # { "类别名": {"total": 0, "correct": 0} }

    for res in results:
        true_label = res["true_label"]
        predicted_label = res["predicted_label"]

        # 初始化类别统计
        if true_label not in class_stats:
            class_stats[true_label] = {"total": 0, "correct": 0}
        
        class_stats[true_label]["total"] += 1

        if true_label == predicted_label:
            correct_count += 1
            class_stats[true_label]["correct"] += 1

    # --- 打印报告 ---
    print("\n" + "="*50)
    print(" " * 18 + "分类准确率报告")
    print("="*50)

    # 1. 总体准确率
    overall_accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
    print(f"[*] 总体准确率: {overall_accuracy:.2f}% ({correct_count} / {total_count})")
    print("-" * 50)

    # 2. 各类别准确率
    print("[*] 各类别详细准确率:")
    sorted_classes = sorted(class_stats.keys())
    for class_name in sorted_classes:
        stats = class_stats[class_name]
        class_total = stats["total"]
        class_correct = stats["correct"]
        class_accuracy = (class_correct / class_total) * 100 if class_total > 0 else 0
        print(f"  - {class_name:<15}: {class_accuracy:>6.2f}% ({class_correct:>3} / {class_total:<3})")

    print("="*50)

# --- 脚本入口 ---
if __name__ == "__main__":
    main()