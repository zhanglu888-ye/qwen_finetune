import json
import re

def process_json(input_file, output_file):
    # 读取输入JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_data = []
    
    for item in data:
        # 保留large_image和small_image
        new_item = {
            "large_image": item.get("large_image", ""),
            "small_image": item.get("small_image", ""),
            "model_output": {}
        }
        
        # 处理model_output字段
        model_output_str = item.get("model_output", "")
        if model_output_str:
            # 从字符串中提取JSON部分
            json_str = re.search(r'```json\n(.*?)\n```', model_output_str, re.DOTALL)
            if json_str:
                try:
                    model_output = json.loads(json_str.group(1))
                    
                    # 只保留observedAttributes中的accessories, carriedItems, heldObjects
                    observed_attrs = model_output.get("observedAttributes", {})
                    filtered_observed = {
                        "accessories": observed_attrs.get("accessories", []),
                        "carriedItems": observed_attrs.get("carriedItems", []),
                        "heldObjects": observed_attrs.get("heldObjects", [])
                    }
                    
                    # 拼接color和type
                    things = []
                    
                    # 处理accessories
                    for acc in filtered_observed["accessories"]:
                        color = acc.get("color", ["无颜色"])[0] if acc.get("color") else "无颜色"
                        item_type = acc.get("type", "无类型")
                        things.append(f"{color}{item_type}")
                    
                    # 处理carriedItems
                    for item in filtered_observed["carriedItems"]:
                        color = item.get("color", ["无颜色"])[0] if item.get("color") else "无颜色"
                        item_type = item.get("type", "无类型")
                        things.append(f"{color}{item_type}")
                    
                    # 处理heldObjects
                    for obj in filtered_observed["heldObjects"]:
                        color = obj.get("color", ["无颜色"])[0] if obj.get("color") else "无颜色"
                        item_type = obj.get("type", "无类型")
                        things.append(f"{color}{item_type}")
                    
                    # 添加things字段
                    filtered_observed["things"] = " ".join(things) if things else "无"
                    
                    new_item["model_output"] = {
                        "observedAttributes": filtered_observed
                    }
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing model_output JSON: {e}")
                    new_item["model_output"] = {}
        
        processed_data.append(new_item)
    
    # 写入输出JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    
    input_file = "/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/persons_result_gemini_0623_1.json"
    output_file = "/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/persons_result_gemini_0623_1_附属物.json"
    
    process_json(input_file, output_file)
    print(f"Processing complete. Output saved to {output_file}")
