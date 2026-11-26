import os
import json
import math
from openpyxl import Workbook
from openpyxl.drawing.image import Image as OpenpyxlImage
from openpyxl.styles import Font, Alignment
from PIL import Image as PillowImage

# 配置参数
IMAGES_PER_EXCEL = 500                      # 每个Excel文件最多图片数
TARGET_IMAGE_HEIGHT = 600                   # 图片显示高度（像素）
COLUMN_WIDTHS = {'A': 65, 'B': 35, 'C': 20, 'D': 15}  # 列宽配置（字符）
ROW_HEIGHT_PADDING = 20                     # 行高附加间距（磅）
OUTPUT_PREFIX = "output_batch_test"              # 输出文件名前缀

def load_json_files(json_paths):
    """加载并合并所有JSON文件中的标注数据"""
    all_entries = []
    for json_path in json_paths:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_entries.extend(data)
                else:
                    print(f"[Warning] {json_path} 格式不符（需数组格式），已跳过")
        except Exception as e:
            print(f"[Error] 加载 {json_path} 失败: {str(e)}")
    return all_entries

def generate_excel(entries, output_path):
    """生成Excel文件"""
    wb = Workbook()
    ws = wb.active
    ws.title = "标注结果"

    # 设置列宽（交换A、B列宽度）
    COLUMN_WIDTHS = {'A': 35, 'B': 65, 'C': 20, 'D': 15}  # 修改后的列宽
    for col, width in COLUMN_WIDTHS.items():
        ws.column_dimensions[col].width = width

    # 创建表头（交换前两列）
    headers = ['图片', '图片路径', '标注结果', '数据来源']  # 修改表头顺序
    for col_num, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col_num, value=header)
        cell.font = Font(bold=True, size=12)
        cell.alignment = Alignment(wrap_text=True, vertical='center', horizontal='center')
    ws.row_dimensions[1].height = 35

    # 填充数据
    for row_num, entry in enumerate(entries, 2):
        # 写入文本数据（交换前两列）
        ws.cell(row=row_num, column=2, value=entry['image_path'])  # B列：原图片路径
        ws.cell(row=row_num, column=3, value=entry['result'])
        ws.cell(row=row_num, column=4, value=entry['data_from'])
        
        # 插入图片到A列
        try:
            with PillowImage.open(entry['original_path']) as img:
                width, height = img.size
                aspect = width / height
                
            xl_img = OpenpyxlImage(entry['original_path'])
            xl_img.height = TARGET_IMAGE_HEIGHT
            xl_img.width = TARGET_IMAGE_HEIGHT * aspect
            ws.add_image(xl_img, f'A{row_num}')  # 修改插入位置为A列
        except Exception as e:
            print(f"[Error] 加载图片失败: {entry['original_path']} - {str(e)}")
            continue

        # 设置行高保持不变
        ws.row_dimensions[row_num].height = (TARGET_IMAGE_HEIGHT * 0.75) + ROW_HEIGHT_PADDING

    # 调整文本样式（跳过A列图片）
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            if cell.column_letter != 'A':  # 跳过图片列
                cell.alignment = Alignment(wrap_text=True, vertical='top')

    try:
        wb.save(output_path)
        print(f"成功生成文件: {output_path}")
    except Exception as e:
        print(f"保存失败: {output_path} - {str(e)}")

def main(json_paths, output_dir):
    # 加载数据
    entries = load_json_files(json_paths)
    print(f"共加载 {len(entries)} 条标注记录")
    
    # 匹配有效条目
    valid_entries = []
    for entry in entries:
        original_path = entry.get('image_path', '')
        result = entry.get('result', '')
        data_from = entry.get('data_from', '')
        
        if os.path.exists(original_path):
            if 'opendata/开源小火苗/D-Fire/lannotated_images_big' in original_path:
                save_path = original_path.replace('/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/D-Fire/lannotated_images_big', 'D-Fire_big')
            if 'opendata/开源小火苗/D-Fire/lannotated_images_small/白天' in original_path:
                save_path = original_path.replace('/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/D-Fire/lannotated_images_small/白天', 'D-Fire_small')   
            if 'opendata/开源小火苗/fire-dataset/lannotated_images_big' in original_path:
                save_path = original_path.replace('/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/fire-dataset/lannotated_images_big', 'fire-dataset_big')
            if 'opendata/开源小火苗/fire-dataset/lannotated_images_small' in original_path:
                save_path = original_path.replace('/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/fire-dataset/lannotated_images_small', 'fire-dataset_small')             
            if 'shandong/fire_数据标注0425_可见光山东_白天' in original_path:
                save_path = original_path.replace('/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/shandong/fire_数据标注0425_可见光山东_白天', 'fire_数据标注0425_可见光山东')   
            if 'yichang/split_images_2024/visible_light/day' in original_path:
                save_path = original_path.replace('/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/yichang/split_images_2024/visible_light/day', 'yichang2024_day')
            if 'yichang/split_images_2025/visible_light/day' in original_path:
                save_path = original_path.replace('/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/yichang/split_images_2025/visible_light/day', 'yichang2025_day') 
            if 'yichang/split_images_2023/visible_light/day' in original_path:
                save_path = original_path.replace('/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/yichang/split_images_2023/visible_light/day', 'yichang2023_day') 
            
            valid_entries.append({
                'original_path': original_path,
                'image_path': save_path,
                'result': result,
                'data_from': data_from
            })
        else:
            print(f"[Warning] 图片未找到: {original_path}")

    # 分批次生成Excel
    total = len(valid_entries)
    batches = math.ceil(total / IMAGES_PER_EXCEL)
    print(f"发现 {total} 个有效条目，将分 {batches} 批导出")

    for i in range(batches):
        start = i * IMAGES_PER_EXCEL
        end = start + IMAGES_PER_EXCEL
        batch = valid_entries[start:end]
        
        output_path = os.path.join(
            output_dir, 
            f"{OUTPUT_PREFIX}_{i+1}.xlsx"
        )
        
        print(f"正在生成第 {i+1}/{batches} 批（{len(batch)} 条）...")
        generate_excel(batch, output_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='标注数据导出工具')
    parser.add_argument('--jsons', 
                      default=[
                          '/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/标注结果/第二批_1w_测试.json',
                        #   '/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/fire-dataset/lannotated_images_big_output.json'
                      ], 
                      nargs='+', 
                      help='JSON文件路径（支持多个）')
    parser.add_argument('--output-dir', 
                      default='/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/标注结果_第二批',
                      help='输出目录（默认当前目录）')
    
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args.jsons, args.output_dir)