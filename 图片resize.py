import os
from PIL import Image

def resize_images(input_dir, output_dir, target_size=(1000, 1000)):
    """
    将输入目录中的所有图片调整为指定大小并保存到输出目录
    
    参数:
        input_dir: 输入图片目录路径
        output_dir: 输出目录路径
        target_size: 目标尺寸，默认为(1000, 1000)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    
    # 遍历输入目录
    for filename in os.listdir(input_dir):
        # 检查文件是否为支持的图片格式
        if filename.lower().endswith(supported_formats):
            try:
                # 构建完整文件路径
                input_path = os.path.join(input_dir, filename)
                output_path = os.path.join(output_dir, filename)
                
                # 打开图片
                with Image.open(input_path) as img:
                    # 调整大小
                    resized_img = img.resize(target_size, Image.LANCZOS)
                    
                    # 保存图片，保持原始格式和质量
                    if filename.lower().endswith(('.jpg', '.jpeg')):
                        resized_img.save(output_path, 'JPEG', quality=95)
                    elif filename.lower().endswith('.png'):
                        resized_img.save(output_path, 'PNG', compress_level=6)
                    else:
                        resized_img.save(output_path)
                    
                    print(f"已处理: {filename} -> {output_path}")
                    
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    # 输入和输出目录
    input_directory = "/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_codetr/cut_imgs_no/expand_1.1_all/images"
    output_directory = "/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_codetr/cut_imgs_no/expand_1.1_all/resized_images"
    
    # 验证输入目录是否存在
    if not os.path.isdir(input_directory):
        print(f"错误: 输入目录 '{input_directory}' 不存在!")
        exit()
    
    # 执行调整大小操作
    resize_images(input_directory, output_directory)
    
    print("\n所有图片处理完成!")