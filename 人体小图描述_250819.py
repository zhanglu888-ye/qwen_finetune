'''配置hosts
/etc/hosts
180.184.58.111 one-api.sensoro.com

恢复-rw-r--r--  644
'''

"""
脚本说明：99调用API进行分析.py
本脚本用于调用API对小图进行分析，并保存结果。
输入: 大图  +  小图  +  prompt（prompt.zh.md）
输出: 分析结果

python 99调用API进行分析.py \
  --prompt_path "../prompt/2-3/prompt.zh.md" \
  --large_image_dir "../data/1原始大图" \
  --data_dir "../data/99提取的小图" \
  --output_dir "../data/100API解析结果" \
  --api_key "sk-DpnzsrnqfZTjmicf48011f2b6e1e4bBf9d22A5Ca7301505b" \
  --model "gemini-2.5-pro-preview-06-05"
  
作者：liangsa
日期：2025-07
"""

 
import os
import json
import base64
import argparse
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
                                # "detail": "low" # 上下文图使用低分辨率以节省成本
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_small_image}",
                                # "detail": "high" # 焦点图使用高分辨率以保证细节
                            },
                        },
                    ],
                }
            ],
            # max_tokens=4096,
            temperature=0.0,
        )
        response_text = response.choices[0].message.content
        
        # 清理并解析JSON响应
        if response_text.startswith("```json"):
            json_str = response_text.strip()[7:-3].strip()
        elif response_text.startswith("{"):
            json_str = response_text
        else:
            # 尝试在响应中找到被包裹的JSON
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

def process_data(
    client: OpenAI,
    model: str,
    prompt_template: str,
    large_image_dir: str,
    data_dir: str,
    output_dir: str
):
    """
    遍历数据目录，为每张小图匹配大图，调用API并保存结果。
    """
    data_path = Path(data_dir)
    large_image_path_dir = Path(large_image_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"开始处理目录: {data_path}")

    # 遍历数据目录中的每个子文件夹（案例）
    for case_dir in data_path.iterdir():
        if not case_dir.is_dir():
            continue

        base_name = case_dir.name
        jsonl_path = case_dir / f"{base_name}.jsonl"

        # 查找对应的大图
        large_image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            potential_path = large_image_path_dir / f"{base_name}{ext}"
            if potential_path.exists():
                large_image_path = potential_path
                logging.info(f"找到案例 '{base_name}' 的大图: {large_image_path}")
                break
        
        if not large_image_path:
            logging.warning(f"找不到与案例 '{base_name}' 对应的大图，跳过整个案例。")
            continue

        logging.info(f"--- 正在处理案例: {base_name} ---")


        for small_img in os.listdir(os.path.join(data_path, base_name)):
                # 准备API调用
                small_image_path = os.path.join(data_path, base_name) + "/" + small_img
                try:
                    base64_large_image = encode_image_to_base64(str(large_image_path))
                    base64_small_image = encode_image_to_base64(str(small_image_path))
                except Exception:
                    continue # 错误已在函数内记录


                person_base_name = Path(small_image_path).stem


                output_json_base = output_dir + '/' + base_name 


                os.makedirs(output_json_base, exist_ok=True)
                
                output_json_path = output_json_base + '/' + person_base_name +'.json'

                

                # 调用API
                analysis_result = call_openai_vision_api(
                    client=client,
                    model=model,
                    prompt=prompt_template,
                    base64_large_image=base64_large_image,
                    base64_small_image=base64_small_image
                )

                with open(output_json_path, 'w', encoding='utf-8') as out_f:
                    json.dump(analysis_result, out_f, ensure_ascii=False, indent=2)
                
                logging.info(f"    [+] 分析结果已保存至: {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="使用OpenAI Vision API对人物小图进行批量分析和描述（同时使用大图作为上下文）。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--prompt_path", type=str, default="prompt_small_v2.zh.md",
        help="包含Prompt模板的 .md 文件路径, e.g., './人体描述规划/2-1/prompt.zh.md'"
    )
    # parser.add_argument(
    #     "--large_image_dir", type=str, default="/host/ssddata/disk2/zhanglu/dataset/vlm/person_desc/250806/tmpResults_250812/downloaded_images_tmp_",
    #     help="包含大图(原图)的文件夹路径, e.g., './筛选小图'"
    # )
    # parser.add_argument(
    #     "--data_dir", type=str, default="/host/ssddata/disk2/zhanglu/dataset/vlm/person_desc/250806/tmpResults_250812/cropped_humans_tmp_",
    #     help="包含(小图+JSONL)子文件夹的主数据路径, e.g., './裁剪后小图_带附属物'"
    # )
    # parser.add_argument(
    #     "--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/dataset/vlm/person_desc/250806/tmpResults_250812/xiaotu_result_tmp2",
    #     help="保存生成分析结果的根文件夹路径, e.g., './analysis_results'"
    # )

    parser.add_argument(
        "--large_image_dir", type=str, default="/host/ssddata/disk2/zhanglu/dataset/codetr/tmp_250909/特殊人群图片",
        help="包含大图(原图)的文件夹路径, e.g., './筛选小图'"
    )
    parser.add_argument(
        "--data_dir", type=str, default="/host/ssddata/disk2/zhanglu/dataset/codetr/tmp_250909/特殊人群图片_codetr_results_tmp_cut",
        help="包含(小图+JSONL)子文件夹的主数据路径, e.g., './裁剪后小图_带附属物'"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/host/ssddata/disk2/zhanglu/dataset/codetr/tmp_250909/特殊人群图片_codetr_results_tmp_cut_gemini_xiaotu",
        help="保存生成分析结果的根文件夹路径, e.g., './analysis_results'"
    )

    parser.add_argument(
        "--api_key", type=str, default='1h1K6lRCB4M2s7n94y14Yr7C9i9D2C8ilr4J3wSq2H4q8T2C',
        help="OpenAI API密钥。如果未提供，将尝试从环境变量 'OPENAI_API_KEY' 读取。"
    )
    parser.add_argument(
        "--api_base_url", type=str, default="https://api.yourouter.ai/v1/",
        help="OpenAI API的基础URL。可用于指向兼容的第三方服务。"
    )
    parser.add_argument(
        "--model", type=str, 
        # default="doubao-pro-vision-2",
        default="gemini-2.5-pro",
        help="要使用的模型名称。"
    )

    args = parser.parse_args()

    # if not args.api_key:
    #     logging.critical("致命错误: 未提供API密钥。请使用 --api_key 参数或设置 OPENAI_API_KEY 环境变量。")
    #     return
        
    for path_arg in [args.prompt_path, args.large_image_dir, args.data_dir]:
        if not Path(path_arg).exists():
            logging.critical(f"致命错误: 路径 '{path_arg}' 不存在。请检查路径。")
            return

    try:
        # 初始化OpenAI客户端
        client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)
        
        # client = OpenAI(
        #     base_url='https://one-api.sensoro.com/v1/',
        #     api_key='sk-DpnzsrnqfZTjmicf48011f2b6e1e4bBf9d22A5Ca7301505b', 
        # )
        
        # 加载Prompt
        prompt_template = get_prompt(args.prompt_path)

        # 开始处理
        process_data(
            client=client,
            model=args.model,
            prompt_template=prompt_template,
            large_image_dir=args.large_image_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        logging.info("所有任务处理完成！")

    except Exception as e:
        logging.critical(f"在主流程中发生无法恢复的错误: {e}")

if __name__ == "__main__":
    main()
