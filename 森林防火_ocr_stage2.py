import os
import json
import shutil
from tqdm import tqdm
from datetime import datetime

# 配置参数
json_file = '/host/algorithm4/zhanglu/dataset/vlm/fire/yichang_60w/按照秒分类_reorganized/output_results_2023.json'  # 替换为你的 JSON 文件路径
output_dir = '/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/yichang/split_images_2023'  # 输出目录
unknown_log_file = os.path.join(output_dir, '/host/algorithm4/zhanglu/dataset/vlm/fire/yichang_60w/split_images_2023_unknown_images.jsonl')  # 未知图像记录文件

visible_dir = os.path.join(output_dir, 'visible_light')
thermal_dir = os.path.join(output_dir, 'thermal_imaging')

visible_day_dir = os.path.join(visible_dir, 'day')
visible_night_dir = os.path.join(visible_dir, 'night')


# 创建目标文件夹（如果不存在）
os.makedirs(visible_day_dir, exist_ok=True)
os.makedirs(visible_night_dir, exist_ok=True)
os.makedirs(thermal_dir, exist_ok=True)


# 定义每月的白昼时间范围 (start_hour, start_minute), (end_hour, end_minute)
MONTH_DAYTIME_RANGES = {
    1: ((7, 0), (18, 30)),
    2: ((6, 40), (18, 50)),
    3: ((6, 10), (19, 5)),
    4: ((5, 45), (19, 20)),
    5: ((5, 30), (19, 30)),
    6: ((5, 25), (19, 35)),
    7: ((5, 35), (19, 30)),
    8: ((5, 50), (19, 5)),
    9: ((6, 10), (18, 40)),
    10: ((6, 5), (18, 20)),
    11: ((6, 0), (18, 0)),
    12: ((6, 50), (18, 20)),
}


# 判断是否是白天
def is_daytime(dt: datetime) -> bool:
    month = dt.month
    hour, minute = dt.hour, dt.minute
    current_time = hour * 60 + minute

    if month not in MONTH_DAYTIME_RANGES:
        return False

    start_time_tuple, end_time_tuple = MONTH_DAYTIME_RANGES[month]
    start_time = start_time_tuple[0] * 60 + start_time_tuple[1]
    end_time = end_time_tuple[0] * 60 + end_time_tuple[1]

    return start_time <= current_time <= end_time


# 解析 image_path 获取时间信息
def parse_datetime_from_path(image_path):
    folder_path = os.path.dirname(image_path)
    parts = os.path.normpath(folder_path).split(os.sep)

    for part in parts:
        try:
            dt = datetime.strptime(part, "%Y-%m-%d %H:%M:%S")
            return dt
        except ValueError:
            continue
    return None


# 主函数
def main():
    # 加载结果
    with open(json_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    # 初始化未知图像记录文件
    with open(unknown_log_file, 'w', encoding='utf-8') as f:
        pass  # 清空或创建文件

    for item in tqdm(results, desc="正在分类图片"):
        image_path = item['image_path']
        result = item['result']

        if "热成像" in result:
            target_dir = thermal_dir
        elif "可见光" in result:
            dt = parse_datetime_from_path(image_path)
            if not dt:
                print(f"无法解析时间信息：{image_path}")
                continue

            if is_daytime(dt):
                target_dir = visible_day_dir
            else:
                target_dir = visible_night_dir
        else:
            # 记录未知图像类型到JSONL文件
            with open(unknown_log_file, 'a', encoding='utf-8') as f:
                json.dump({"image_path": image_path, "result": result}, f, ensure_ascii=False)
                f.write('\n')
            continue

        # 构建目标路径
        image_name = os.path.basename(image_path)
        dst_path = os.path.join(target_dir, image_name)

        if os.path.exists(dst_path):
            # 已存在则跳过
            continue

        try:
            shutil.copy(image_path, dst_path)  # 使用 copy 或 move
        except Exception as e:
            print(f"处理失败：{image_path} | 错误：{str(e)}")

    print("图片分类完成！")
    print(f"未知图像类型已记录到: {unknown_log_file}")


if __name__ == '__main__':
    main()