import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse
import textwrap

def create_combined_image(image_path, text, output_path, font_path=None):
    """创建左右分栏的图片，左侧是原图，右侧是文本描述"""
    try:
        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"无法读取图片: {image_path}")
            return
        
        # 获取图片尺寸
        img_height, img_width = img.shape[:2]
        
        # 右侧文本区域宽度与图片相同
        text_width = img_width
        
        # 创建空白右侧区域 (白色背景)
        text_img = np.ones((img_height, text_width, 3), dtype=np.uint8) * 255
        
        # 将OpenCV图像转换为PIL图像以便添加中文文本
        pil_text_img = Image.fromarray(text_img)
        draw = ImageDraw.Draw(pil_text_img)
        
        # 设置字体 (使用系统字体或指定字体文件)
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, 20)
            else:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # 设置文本区域边距和行间距
        margin = 20
        line_spacing = 5
        x, y = margin, margin
        max_width = text_width - 2 * margin
        max_height = img_height - 2 * margin
        
        # 改进的文本换行处理
        paragraphs = text.split('\n')
        y_text = y
        
        for para in paragraphs:
            if not para.strip():
                y_text += line_spacing * 2  # 段落间距
                continue
                
            # 使用textwrap进行自动换行
            wrapped_lines = textwrap.wrap(para, width=75)  # 每行约40个字符
            
            for line in wrapped_lines:
                if y_text > max_height - 25:  # 预留最后一行空间
                    break
                
                # 绘制文本行
                draw.text((x, y_text), line, font=font, fill=(0, 0, 0))
                y_text += 25 + line_spacing  # 行高+行间距
        
        # 转换回OpenCV格式
        text_img = np.array(pil_text_img)
        
        # 合并左右图像
        combined_img = np.hstack((img, text_img))
        
        # 保存结果
        cv2.imwrite(output_path, combined_img)
        
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")

def process_json(input_json, output_folder, font_path=None):
    """处理JSON文件并生成合并图片"""
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 读取JSON文件
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理每个条目
    for item in data:
        image_path = item.get('image_path')
        model_output = item.get('model_output', '')
        
        if not image_path or not os.path.exists(image_path):
            print(f"图片路径不存在: {image_path}")
            continue
        
        # 生成输出文件名
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, base_name)
        
        # 创建合并图片
        create_combined_image(image_path, model_output, output_path, font_path)
        print(f"已生成: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='将图片和描述合并为左右分栏图片')
    parser.add_argument('--input_json', default="/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/大图描述_0627.json", help='输入JSON文件路径')
    parser.add_argument('--output_folder', default="/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/大图描述_show", help='输出文件夹路径')
    parser.add_argument('--font_path', default="/host/algorithm2/zhanglu/code/SimHei.ttf", help='可选的中文字体文件路径(如.ttf)')
    args = parser.parse_args()
    
    process_json(args.input_json, args.output_folder, args.font_path)
    print("处理完成!")

if __name__ == '__main__':
    main()