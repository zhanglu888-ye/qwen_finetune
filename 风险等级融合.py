import pandas as pd
from collections import Counter

def determine_fusion_level(row):
    """
    根据投票规则确定融合风险等级
    规则：
    1. 三个中有两个相同则取多数
    2. 三个都不同时取gpt的结果
    """
    # 获取三个模型的预测值
    doubao = row['风险等级-豆包']
    gpt = row['风险等级-gpt']
    qwen = row['风险等级-qwen']
    
    # 构造值列表
    values = [doubao, gpt, qwen]
    
    # 统计频率
    count = Counter(values)
    
    # 情况1：存在多数票（出现次数>=2）
    for value, cnt in count.items():
        if cnt >= 2:
            return value
    
    # 情况2：三个都不相同，返回gpt的结果
    return doubao

def process_risk_levels(input_file, output_file):
    """
    处理风险等级融合逻辑
    :param input_file: 输入CSV文件路径
    :param output_file: 输出CSV文件路径
    """
    # 读取CSV文件
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"错误：文件 {input_file} 不存在")
        return
    
    # 验证必要列是否存在
    required_columns = ['风险等级-豆包', '风险等级-gpt', '风险等级-qwen']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"错误：文件缺少必要列 {missing_cols}")
        return
    
    # 应用融合逻辑
    df['风险等级-融合'] = df.apply(determine_fusion_level, axis=1)
    
    # 保存结果
    df.to_csv(output_file, index=False, encoding='utf_8_sig')
    print(f"处理完成，结果已保存至：{output_file}")

if __name__ == "__main__":
    # 配置输入输出路径
    input_csv = "/host/algorithm4/zhanglu/dataset/vlm/eval_fire/标注风险等级.csv"    # 替换为实际输入文件路径
    output_csv = "/host/algorithm4/zhanglu/dataset/vlm/eval_fire/标注风险等级_new.csv"  # 替换为实际输出文件路径
    
    # 执行处理
    process_risk_levels(input_csv, output_csv)