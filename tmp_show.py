import os
import json
from PIL import Image, ImageDraw, ImageFont

def visualize_boxes(image_path, boxes_data, output_path=None, show_labels=True):
    """
    可视化归一化到0-1000的边界框
    
    参数:
        image_path: 输入图片路径
        boxes_data: 边界框数据列表，格式如 [{"box_2d": [ymin,xmin,ymax,xmax], "label": "class"}, ...]
        output_path: 输出图片路径(可选)
        show_labels: 是否显示标签(默认为True)
    """
    # 打开图片
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"错误: 找不到图片文件 {image_path}")
        return
    except Exception as e:
        print(f"错误: 无法打开图片 {image_path}: {e}")
        return
    
    # 获取图片原始尺寸
    img_width, img_height = img.size
    
    # 创建绘图对象
    draw = ImageDraw.Draw(img)
    
    # 定义颜色映射 (可以根据需要修改)
    color_map = {
        "car": "red",
        "person": "blue",
        "man": "green",
        "woman": "purple",
        # 默认颜色
        "default": "yellow"
    }
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # 绘制每个边界框
    for box_info in boxes_data:
        # 获取归一化坐标
        ymin_norm, xmin_norm, ymax_norm, xmax_norm = box_info["box_2d"]
        # xmin_norm, ymin_norm, xmax_norm, ymax_norm = box_info["box_2d"]        
        
        
        label = box_info.get("label", "")
        
        # 将归一化坐标转换为实际像素坐标
        ymin = int(ymin_norm * img_height / 1000)
        xmin = int(xmin_norm * img_width / 1000)
        ymax = int(ymax_norm * img_height / 1000)
        xmax = int(xmax_norm * img_width / 1000)
        
        # 获取颜色
        box_color = color_map.get(label.lower(), color_map["default"])
        
        # 绘制边界框
        draw.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=3)
        
        # 绘制标签
        if show_labels and label:
            # 计算文本位置 (放在框的左上角)
            text_position = (xmin + 5, ymin + 5)
            # 绘制文本背景
            text_bbox = draw.textbbox(text_position, label, font=font)
            draw.rectangle(text_bbox, fill="black")
            # 绘制文本
            draw.text(text_position, label, fill="white", font=font)
    
    # 保存或显示结果
    if output_path:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        print(f"可视化结果已保存到: {output_path}")
    else:
        img.show()
    
    return img

# 示例使用
if __name__ == "__main__":
    # 输入图片路径
    input_image = "/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图/2633_0_1749465087_758_NM-1040-576-176-336.jpg"  # 替换为你的图片路径
    
    # 边界框数据 (可以直接使用你提供的JSON)
    boxes_data = [
  {
    "box_2d": [457, 281, 545, 667],
    "category": "人体"
  },
  {
    "box_2d": [583, 325, 676, 764],
    "category": "人体"
  },
  {
    "box_2d": [629, 510, 782, 998],
    "category": "人体"
  },
  {
    "box_2d": [418, 480, 482, 639],
    "category": "人体手拿物"
  },
  {
    "box_2d": [646, 714, 717, 865],
    "category": "背包"
  },
  {
    "box_2d": [632, 816, 692, 931],
    "category": "手提物"
  },
  {
    "box_2d": [728, 716, 753, 760],
    "category": "手机"
  }
]
    
    
    
    
    # 输出路径 (可选)
    output_image = "/host/algorithm4/zhanglu/dataset/tmp/0609/2633_visualized_output.jpg"
    
    # 调用可视化函数
    visualize_boxes(input_image, boxes_data, output_image)