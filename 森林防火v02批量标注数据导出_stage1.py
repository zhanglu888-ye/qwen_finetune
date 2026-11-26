import json
import random
from collections import defaultdict
import os

# 配置规则（每个文件单独设置抽取规则）
'''
##第一批数据集
RULES = [
    {   # 示例配置（请根据实际需求修改）
        "path": "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/fire-dataset/lannotated_images_small_output_v11.json",
        "select": {
            "高风险": 0.22,   
            "疑似高风险": 1.0,    
            "中风险": 1.0,   
            "疑似中风险": 0.45,    
            "正常": 0.16
        }
    },
    {
        "path": "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/fire-dataset/lannotated_images_big_output_v11.json",
        "select": {
            "高风险": 0.22,   
            "疑似高风险": 1.0,    
            "中风险": 1.0,   
            "疑似中风险": 0.45,    
            "正常": 0.16
        }
    },
    {       
        "path": "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/D-Fire/lannotated_images_small/白天_output_v11.json",
        "select": {
            "高风险": 0.22,   
            "疑似高风险": 1.0,    
            "中风险": 1.0,   
            "疑似中风险": 0.45,    
            "正常": 0.16
            } 
    },
    {
        "path": "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/opendata/开源小火苗/D-Fire/lannotated_images_big_output_v11.json",
        "select": {
            "高风险": 0.22,   
            "疑似高风险": 1.0,    
            "中风险": 1.0,   
            "疑似中风险": 0.45,    
            "正常": 0.16
            }
    },      
    { 
        "path": "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/shandong/fire_数据标注0425_可见光山东_白天_output_v11.json",
        "select": {
            "高风险": 1.0,   
            "疑似高风险": 1.0,    
            "中风险": 1.0,   
            "疑似中风险": 0.45,    
            "正常": 0.16
            }
    },  
    {
        "path": "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/yichang/split_images_2024/visible_light/day_output_v11.json",
        "select": {
            "高风险": 0.22,   
            "疑似高风险": 1.0,    
            "中风险": 1.0,   
            "疑似中风险": 0.45,    
            "正常": 0.16
            } 
    },
    {
        "path": "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/yichang/split_images_2025/visible_light/day_output_v11.json",
        "select": {
            "高风险": 0.22,   
            "疑似高风险": 1.0,    
            "中风险": 1.0,   
            "疑似中风险": 0.45,    
            "正常": 0.16
            }
    },                
]
'''
##第二批数据集
RULES = [
    {   # 示例配置（请根据实际需求修改）
        "path": "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/yichang/split_images_2023/visible_light/day_output_v11.json",
        "select": {
            "高风险": 1.0,   
            "疑似高风险": 1.0,    
            "中风险": 1.0,   
            "疑似中风险": 1.0,    
            "正常": 0.36
        }
    }            
]



def load_single_file(json_path):
    """加载单个JSON文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[Error] 加载 {json_path} 失败: {str(e)}")
        return []

def process_file(data, select_rules):
    """处理单个文件数据"""
    selected = []
    remaining = []
    category_pool = defaultdict(list)
    category_stats = defaultdict(int)  # 新增：用于统计每个类别的选中数量
    
    # 第一遍：分类数据
    for item in data:
        result = item.get("result", "未标注").strip()
        normalized = ' '.join(result.split())
        
        matched = False
        for prefix in ['正常', '疑似中风险', '疑似高风险', '高风险', '中风险']:
            if normalized.startswith(prefix):
                category = prefix
                if category in select_rules:
                    ratio = select_rules[category]
                    if ratio == 1.0:
                        selected.append(item)
                        category_stats[category] += 1  # 记录全量抽取
                    else:
                        category_pool[category].append(item)
                    matched = True
                    break
        if not matched:
            remaining.append(item)
    
    # 第二遍：处理需要抽样的类别
    for category, items in category_pool.items():
        ratio = select_rules[category]
        total = len(items)
        select_num = int(round(total * ratio))
        
        random.shuffle(items)
        selected.extend(items[:select_num])
        remaining.extend(items[select_num:])
        category_stats[category] += select_num  # 记录抽样数量
    
    return selected, remaining, category_stats  # 修改返回参数

def split_data():
    """执行数据分割主逻辑"""
    all_selected = []
    all_remaining = []
    
    for rule in RULES:
        path = rule["path"]
        select_rules = rule["select"]
        
        data = load_single_file(path)
        if not data:
            continue
            
        selected, remaining, stats = process_file(data, select_rules)  # 获取统计信息
        all_selected.extend(selected)
        all_remaining.extend(remaining)
        
        # 新增详细统计输出
        print(f"\n处理文件：{os.path.basename(path)}")
        print("├─ 选中类别明细：")
        for category, count in stats.items():
            print(f"│   ├─ {category}: {count}")
        print(f"│   └─ 总计选中：{len(selected)}")
        print(f"└─ 剩余数量：{len(remaining)}")
    
    return all_selected, all_remaining

# 其他代码保持不变...

def save_json(data, filename):
    """保存JSON文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n保存成功：{filename} ({len(data)}条)")
def load_json_files(json_paths):
    """加载多个JSON文件并合并数据"""
    all_data = []
    for path in json_paths:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    print(f"[Warning] {path} 格式不符，已跳过")
        except Exception as e:
            print(f"[Error] 加载 {path} 失败: {str(e)}")
    return all_data

def count_categories(data):
    """统计详细分类和大类"""
    detail_counter = defaultdict(int)
    major_counter = defaultdict(int)
    
    # 定义大类映射规则
    major_mapping = {
        '正常': '正常',
        '疑似中风险': '疑似中风险',
        '疑似高风险': '疑似高风险', 
        '高风险': '高风险',
        '中风险': '中风险'
    }
    
    for item in data:
        result = item.get("result", "未标注").strip()
        normalized = ' '.join(result.split())
        
        # 详细统计
        detail_counter[normalized] += 1
        
        # 大类统计
        found = False
        for prefix, category in major_mapping.items():
            if normalized.startswith(prefix):
                major_counter[category] += 1
                found = True
                break
        if not found and normalized != "未标注":
            major_counter["其他"] += 1
    
    return detail_counter, major_counter

def print_statistics(detail_counter, major_counter):
    """打印统计结果"""
    total = sum(detail_counter.values())
    
    # 详细统计
    print("\n详细类别统计：")
    print("-" * 60)
    print(f"{'风险类别':<20} | {'数量':<8} | {'占比':<6} | 累计占比")
    print("-" * 60)
    
    sorted_details = sorted(detail_counter.items(), 
                          key=lambda x: x[1], 
                          reverse=True)
    
    cumulative = 0
    for category, count in sorted_details:
        percentage = count / total * 100
        cumulative += percentage
        print(f"{category:<20} | {count:<8} | {percentage:5.1f}% | {cumulative:5.1f}%")
    
    # 大类统计
    print("\n大类汇总统计：")
    print("-" * 40)
    print(f"{'风险大类':<10} | {'数量':<8} | {'占比':<6}")
    print("-" * 40)
    
    sorted_majors = sorted(major_counter.items(),
                         key=lambda x: x[1],
                         reverse=True)
    
    for category, count in sorted_majors:
        percentage = count / total * 100
        print(f"{category:<10} | {count:<8} | {percentage:5.1f}%")
    
    print("-" * 40)
    print(f"总数据量: {total}")




def split_secondary(data, ratio=0.5):
    """二级分层抽样"""
    # 按类别分组
    category_groups = defaultdict(list)
    for item in data:
        result = item.get("result", "未标注").strip()
        normalized = ' '.join(result.split())
        for prefix in ['正常', '疑似中风险', '疑似高风险', '高风险', '中风险']:
            if normalized.startswith(prefix):
                category = prefix
                break
        else:
            category = "其他"
        category_groups[category].append(item)
    
    # 分层抽样
    selected_1 = []
    selected_2 = []
    stats = defaultdict(lambda: {'total':0, 'part1':0, 'part2':0})
    
    for category, items in category_groups.items():
        random.shuffle(items)
        total = len(items)
        split_idx = int(round(total * ratio))
        
        stats[category]['total'] = total
        stats[category]['part1'] = split_idx
        stats[category]['part2'] = total - split_idx
        
        selected_1.extend(items[:split_idx])
        selected_2.extend(items[split_idx:])
    
    return selected_1, selected_2, stats

def print_secondary_stats(stats):
    """打印二级抽样统计"""
    print("\n二级抽样统计：")
    print("-" * 60)
    print(f"{'类别':<15} | {'总数':<6} | Part1(90%) | Part2(10%)")
    print("-" * 60)
    for cat, data in stats.items():
        print(f"{cat:<15} | {data['total']:<6} | {data['part1']:<9} | {data['part2']}")
    print("-" * 60)





if __name__ == "__main__":
    # 原有统计功能
    all_data = load_json_files([rule["path"] for rule in RULES])
    if all_data:
        detail, major = count_categories(all_data)
        print("\n原始数据统计：")
        print_statistics(detail, major)
    
    #新增数据分割功能
    selected_data, remaining_data = split_data()
    
    # 保存结果
    save_json(selected_data, "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/标注结果/第二批_2w.json")
    save_json(remaining_data, "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/标注结果/第二批_其余.json")
    
    # 新增统计结果
    print("\n选中数据统计：")
    detail_sel, major_sel = count_categories(selected_data)
    print_statistics(detail_sel, major_sel)
    
    print("\n剩余数据统计：")
    detail_rem, major_rem = count_categories(remaining_data)
    print_statistics(detail_rem, major_rem)
    
    
    # 新增二级抽样
    if selected_data:
        selected_1, selected_2, stats = split_secondary(selected_data)
        save_json(selected_1, "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/标注结果/第二批_1w_训练.json")
        save_json(selected_2, "/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级v02批量标注数据集/标注结果/第二批_1w_测试.json")
        print_secondary_stats(stats)
        
        # 新增统计验证
        print("\n验证统计：")
        print(f"原始选中数据量：{len(selected_data)}")
        print(f"分割后Part1数量：{len(selected_1)} (应有：{int(len(selected_data)*0.5)})")
        print(f"分割后Part2数量：{len(selected_2)} (应有：{len(selected_data)-int(len(selected_data)*0.5)})")
