import os
import cv2
from tqdm import tqdm

def extract_yes_objects(image_folder, txt_folder, output_folder):
    """
    提取类别为0且标记为yes的目标区域并保存
    
    :param image_folder: 图片文件夹路径
    :param txt_folder: 标签txt文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]
    
    for txt_file in tqdm(txt_files, desc="Processing files"):
        try:
            # 构建文件路径
            txt_path = os.path.join(txt_folder, txt_file)
            base_name = os.path.splitext(txt_file)[0]
            image_path = os.path.join(image_folder, f"{base_name}.jpg")  # 假设图片是jpg格式
            
            # 检查图片是否存在
            if not os.path.exists(image_path):
                print(f"Warning: Image not found for {txt_file}")
                continue
                
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                continue
                
            # 创建当前图片的输出子文件夹
            image_output_folder = os.path.join(output_folder, base_name)
            os.makedirs(image_output_folder, exist_ok=True)
            
            # 读取txt内容
            with open(txt_path, 'r') as f:
                lines = f.readlines()
                
            # 处理每一行
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 2:  # 不是有效的目标行
                    continue
                    
                # 检查类别和yes标记
                try:
                    label = int(float(parts[0]))
                    if label != 0:
                        continue
                        
                    # 检查最后是否是yes
                    if parts[-1].lower() != 'yes':
                        continue
                        
                    # 获取坐标
                    x1, y1, x2, y2 = map(float, parts[1:5])
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 确保坐标在图像范围内
                    h, w = img.shape[:2]
                    x1 = max(0, min(x1, w-1))
                    y1 = max(0, min(y1, h-1))
                    x2 = max(0, min(x2, w-1))
                    y2 = max(0, min(y2, h-1))
                    
                    # 确保区域有效
                    if x1 >= x2 or y1 >= y2:
                        continue
                        
                    # 裁剪图像区域
                    cropped_img = img[y1:y2, x1:x2]
                    if cropped_img.size == 0:
                        continue
                        
                    # 保存抠图到当前图片的子文件夹
                    output_filename = f"{base_name}_obj{i}.jpg"
                    output_path = os.path.join(image_output_folder, output_filename)
                    cv2.imwrite(output_path, cropped_img)
                    
                except Exception as e:
                    print(f"Error processing line in {txt_file}: {line.strip()} - {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error processing file {txt_file}: {str(e)}")
            continue

if __name__ == '__main__':
    # 设置路径
    # image_folder = "/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图"  # 替换为你的图片文件夹路径
    # txt_folder = "/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/annotated_txt_quality"    # 替换为你的标签txt文件夹路径
    # output_folder = "/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/persons_cut"      # 替换为输出文件夹路径
    
    image_folder = "/host/algorithm4/zhanglu/dataset/tmp/0609/2k张大图_codetr/images"  # 替换为你的图片文件夹路径
    txt_folder = "/host/algorithm4/zhanglu/dataset/tmp/0609/2k张大图_codetr/depth_v4_pqa/annotated_txt_quality"    # 替换为你的标签txt文件夹路径
    output_folder = "/host/algorithm4/zhanglu/dataset/tmp/0609/2k张大图_codetr/depth_v4_pqa/persons_cut"      # 替换为输出文件夹路径



    # 提取目标区域
    extract_yes_objects(image_folder, txt_folder, output_folder)
    print("\n所有文件处理完成！")