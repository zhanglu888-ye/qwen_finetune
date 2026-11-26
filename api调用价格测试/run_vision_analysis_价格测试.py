"""
单次执行版本：
每次针对一张大图和一张小图，运行一次分析并保存结果。
请在运行前，修改 main 函数中的文件路径和 API 密钥配置。
"""

import os
import json
import base64
import logging
from pathlib import Path
from openai import OpenAI
from typing import Dict, Any

# --- 基本配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def encode_image_to_base64(image_path: str) -> str:
    """读取图片文件并将其编码为base64字符串。"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except IOError as e:
        logging.error(f"无法读取或编码图片 {image_path}: {e}")
        raise

def get_prompt(prompt_path: str) -> str:
    """从文件加载Prompt模板。"""
    try:
        return Path(prompt_path).read_text(encoding='utf-8')
    except FileNotFoundError:
        logging.error(f"Prompt模板文件未找到: {prompt_path}")
        raise
    except Exception as e:
        logging.error(f"读取Prompt模板时发生错误 {prompt_path}: {e}")
        raise

def call_openai_vision_api(
    client: OpenAI,
    model: str,
    prompt: str,
    base64_large_image: str,
    base64_small_image: str
) -> Dict[str, Any]:
    """
    调用OpenAI Vision API并返回JSON格式的响应。
    此次调用会发送一张大图（上下文）和一张小图（焦点）。
    """
    pre_prompt = """您将收到两张图片：
- **第一张图片** 是原始的、广角的视图，为您提供完整的环境背景。
- **第二张图片** 是聚焦于我们要分析的核心人物的、经过裁剪的特写图。

您的核心任务是：**以第二张特写图中的人物为绝对焦点，进行极致详细的分析。**
同时，您必须利用第一张广角图来理解该人物的周边环境、与他人的互动，以及任何在特写图中可能被截断的身体部分或物品。

现在，请严格遵循下面的详细指令来完成任务。
---
"""
    final_prompt = pre_prompt + prompt

    try:
        logging.info("正在调用视觉API (大图+小图模式)...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_large_image}",
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_small_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.0,
        )
        response_text = response.choices[0].message.content
        
        # 清理并解析JSON响应
        if response_text.startswith("```json"):
            json_str = response_text.strip()[7:-3].strip()
        elif response_text.startswith("{"):
            json_str = response_text
        else:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx+1]
            else:
                raise json.JSONDecodeError("无法在响应中找到有效的JSON对象。", response_text, 0)
        
        return json.loads(json_str)

    except json.JSONDecodeError as e:
        logging.error(f"API返回的不是有效的JSON格式: {e}")
        logging.error(f"原始响应: {response_text}")
        return {"error": "Invalid JSON response", "raw_response": response_text}
    except Exception as e:
        logging.error(f"调用API时发生错误: {e}")
        return {"error": str(e)}

def main():
    """
    主执行函数。
    每次运行前，请在此处配置本次分析所需的所有路径和参数。
    """
    # =========================================================================
    #  每次运行时请在此处修改配置
    #  路径示例:
    #  - Windows: "C:\\Users\\YourUser\\Documents\\Project\\image.jpg"
    #  - Linux / macOS: "/home/youruser/project/image.jpg"
    # =========================================================================
    
    # --- 1. 文件路径配置 (必须修改为您的绝对路径) ---
    
    # Prompt模板文件的路径
    prompt_path = "/host/ssddata/disk2/zhanglu/qwen_eval/api调用价格测试/prompt.zh.md"
    
    # 大图 (上下文背景图) 的完整路径
    large_image_path = "/host/ssddata/disk2/zhanglu/qwen_eval/api调用价格测试/2633_0_1749465087_758_NM-1040-576-176-336.jpg"
    
    # 小图 (要分析的人物特写图) 的完整路径
    small_image_path = "/host/ssddata/disk2/zhanglu/qwen_eval/api调用价格测试/person_1.jpg"
    
    # 保存分析结果的文件夹路径 (文件名会自动生成)
    output_dir = "/host/ssddata/disk2/zhanglu/qwen_eval/api调用价格测试/yourouter"

    # --- 2. API 配置 ---
    api_key = "2k4SZWsk3UDmkdWUHVqQr9XgwKpri8U6wS5wrfBrU5y4HvZb"
    api_base_url = "https://api.yourouter.ai/v1" # "https://one-api.sensoro.com/v1/"
    
    # --- 3. 模型配置 ---
    model_name = "gemini-2.5-pro"

    # =========================================================================
    #  配置修改结束
    # =========================================================================

    # --- 检查配置 ---
    if "YOUR_API_KEY" in api_key:
        logging.critical("致命错误: API密钥未配置。请在脚本中修改 'api_key' 变量或设置 OPENAI_API_KEY 环境变量。")
        return
        
    for path_str in [prompt_path, large_image_path, small_image_path]:
        if "/path/to/your/" in path_str:
            logging.critical(f"致命错误: 路径 '{path_str}' 看起来像一个未修改的占位符。请将其更新为实际的绝对路径。")
            return
        if not Path(path_str).exists():
            logging.critical(f"致命错误: 路径 '{path_str}' 不存在。请检查路径。")
            return

    # --- 开始执行任务 ---
    try:
        logging.info(f"开始处理: \n  - 大图: {large_image_path}\n  - 小图: {small_image_path}")

        # 初始化客户端
        client = OpenAI(api_key=api_key, base_url=api_base_url)
        
        # 加载Prompt
        prompt_template = get_prompt(prompt_path)

        # 编码图片
        base64_large_image = encode_image_to_base64(large_image_path)
        base64_small_image = encode_image_to_base64(small_image_path)
        
        # 调用API
        analysis_result = call_openai_vision_api(
            client=client,
            model=model_name,
            prompt=prompt_template,
            base64_large_image=base64_large_image,
            base64_small_image=base64_small_image
        )

        # 准备输出路径
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 使用小图的文件名（不含扩展名）来命名结果文件
        small_image_stem = Path(small_image_path).stem
        output_json_path = output_path / f"{small_image_stem}_analysis.json"

        # 保存结果
        with open(output_json_path, 'w', encoding='utf-8') as out_f:
            json.dump(analysis_result, out_f, ensure_ascii=False, indent=2)
        
        logging.info(f"任务完成！分析结果已保存至: {output_json_path}")

    except Exception as e:
        logging.critical(f"在主流程中发生无法恢复的错误: {e}")

if __name__ == "__main__":
    main()