import os
import json
from tqdm import tqdm
from PIL import Image

def find_matching_txt_line(large_image_path, box, txt_folder, img_width, img_height):
    """在对应的txt文件中查找匹配的坐标行"""
    # 从large_image路径提取文件名（不含扩展名）
    base_name = os.path.splitext(os.path.basename(large_image_path))[0]
    txt_path = os.path.join(txt_folder, f"{base_name}.txt")
    
    if not os.path.exists(txt_path):
        return None
    
    # 将json中的box转换为整数列表
    target_box = list(map(int, box))
    
    with open(txt_path, 'r') as f:
        for line_num, line in enumerate(f, 1):  # 从1开始计数
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0] == '0':  # 只处理类别0的行
                # 将txt中的坐标转换为整数                
                x1, y1, x2, y2 = map(lambda x: int(float(x)), parts[1:5])
                    
                    # 坐标归一化计算 (防止除零错误)
                width_scale = img_width if img_width > 0 else 1
                height_scale = img_height if img_height > 0 else 1
                    
                x1_new = int(x1 / width_scale * 1000)
                y1_new = int(y1 / height_scale * 1000)
                x2_new = int(x2 / width_scale * 1000)
                y2_new = int(y2 / height_scale * 1000)
                
                txt_box=[x1_new, y1_new, x2_new, y2_new]
                
                # 比较两个box是否完全一致
                if txt_box == target_box:
                    return line_num - 1  # 返回行号(从0开始)
    
    return None

def process_json(input_json_path, txt_folder, output_json_path):
    """处理JSON文件并添加small_image字段"""
    # 读取输入JSON
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理每个条目
    for item in tqdm(data, desc="Processing items"):
        large_image_path = item["large_image"]
        box = item["box"]
        
        
        with Image.open(large_image_path) as img:
            img_width, img_height = img.size
        
        # 查找匹配的txt行号
        matched_line = find_matching_txt_line(large_image_path, box, txt_folder, img_width, img_height)
        
        if matched_line is not None:
            # 构建small_image路径
            base_name = os.path.splitext(os.path.basename(large_image_path))[0]
            small_image_name = f"{base_name}_obj{matched_line}.jpg"
            
            # 构建完整路径（根据您的目录结构调整）
            small_image_dir = os.path.join(
                os.path.dirname(os.path.dirname(large_image_path)),
                "筛选小图_depth_v3/persons_cut",
                base_name
            )
            small_image_path = os.path.join(small_image_dir, small_image_name)
            
            # 添加small_image字段
            item["small_image"] = small_image_path
    
    # 保存输出JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="匹配JSON和TXT文件并添加small_image字段")
    parser.add_argument("--input_json", default="/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/persons_normolized_labels_result_gemini_tmp.json", help="输入JSON文件路径")
    parser.add_argument("--txt_folder", default="/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/annotated_txt_quality", help="TXT文件夹路径")
    parser.add_argument("--output_json", default="/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/persons_normolized_labels_result_gemini_tmp_pro.json", help="输出JSON文件路径")
    
    args = parser.parse_args()
    
    process_json(args.input_json, args.txt_folder, args.output_json)
    print(f"处理完成，结果已保存到 {args.output_json}")