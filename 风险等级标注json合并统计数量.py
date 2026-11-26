import json
import glob
import os

def merge_and_statistics(input_paths, merged_output):
    """
    参数说明：
    input_paths - 可以是单个目录路径或包含多个路径的列表
    merged_output - 合并后的输出文件路径
    """
    merged_data = []
    categories = {
        '高风险': 0,
        '疑似高风险': 0,
        '中风险': 0,
        '疑似中风险': 0,
        '正常': 0
    }

    # 处理输入路径（支持目录和文件混合）
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    for path in input_paths:
        # 如果是目录，递归查找所有json文件
        if os.path.isdir(path):
            json_files = glob.glob(os.path.join(path, "**/*.json"), recursive=True)
        # 如果是文件直接加入列表
        elif os.path.isfile(path):
            json_files = [path]
        else:
            continue

        # 合并数据并统计
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    merged_data.extend(data)
                    
                    # 实时统计（避免二次遍历）
                    for item in data:
                        result = item.get('result', '')
                        if result:
                            category = result.split()[0]
                            if category in categories:
                                categories[category] += 1
            except Exception as e:
                print(f"处理文件 {file_path} 时出错：{str(e)}")

    # 保存合并文件
    with open(merged_output, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)

    return categories

if __name__ == "__main__":
    # 配置参数（支持三种格式）
    input_paths = [
        "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/fire-dataset/lannotated_images_small_output.json", 
        "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/fire-dataset/lannotated_images_big_output.json", 
        "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/D-Fire/lannotated_images_small/白天_output.json",
        "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/D-Fire/lannotated_images_big_output.json"
        
        
        
    ]
    

/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/shandong/fire_数据标注0425_可见光山东_白天_output.json --data_from 'shandong'
python 森林防火v02批量标注0523.py --img_dir /host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/yichang/split_images_2024/visible_light/day --output_json_path /host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/yichang/split_images_2024/visible_light/day_output.json --data_from 'yichang'
python 森林防火v02批量标注0523.py --img_dir /host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/yichang/split_images_2025/visible_light/day --output_json_path /host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/yichang/split_images_2025/visible_light/day_output.json --data_from 'yichang'
    
    
    output_file = "merged.json"

    # 执行操作
    statistics = merge_and_statistics(input_paths, output_file)

    # 打印统计结果
    print("\n类别统计结果：")
    for category, count in statistics.items():
        print(f"{category.ljust(8)}: {count}")