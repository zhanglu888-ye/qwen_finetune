import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# 类别列表
CLASSES = ['高风险', '疑似高风险', '中风险', '疑似中风险', '正常']

def extract_class(result_str):
    try:
        result_json = json.loads(result_str)
        label_code = result_json.get("标识码", "")
        
        # 根据标识码映射风险等级
        if label_code.startswith("HC"):
            return "高风险"
        elif label_code.startswith("HS"):
            return "疑似高风险"
        elif label_code.startswith("MC"):
            return "中风险"
        elif label_code.startswith("MS"):
            return "疑似中风险"
        elif label_code.startswith("LC"):
            return "正常"
        else:
            return "未知"
    except json.JSONDecodeError:
        return "未知"

def compute_metrics(gt_dict, pred_dict, classes):
    # 初始化统计字典
    stats = {cls: {'标注数量': 0, '模型识别数量': 0, '正确数量': 0} for cls in classes}
    
    # 统计每个类别的数据
    for img_name in gt_dict:
        gt_cls = gt_dict[img_name]
        pred_cls = pred_dict.get(img_name, None)

        if gt_cls not in classes:
            continue

        # 增加标注数量
        stats[gt_cls]['标注数量'] += 1

        if pred_cls is not None:
            # 增加模型识别数量
            if pred_cls in classes:
                stats[pred_cls]['模型识别数量'] += 1
                # 判断是否正确
                if pred_cls == gt_cls:
                    stats[gt_cls]['正确数量'] += 1

    # 计算指标
    total_correct = 0
    total_gt = 0
    results = []

    for cls in classes:
        tp = stats[cls]['正确数量']
        fp = stats[cls]['模型识别数量']
        fn = stats[cls]['标注数量']

        precision = tp / fp if fp > 0 else 0
        recall = tp / fn if fn > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            '火灾风险类型': cls,
            '标注数量': fn,
            '模型识别数量': fp,
            '正确数量': tp,
            '精准率': round(precision, 4),
            '召回率': round(recall, 4),
            'F1-score': round(f1, 4),
        })

        total_correct += tp
        total_gt += fn

    avg_acc = total_correct / total_gt if total_gt > 0 else 0

    # 添加平均准确率到每一行
    for i in range(len(results)):
        results[i]['平均准确率'] = round(avg_acc, 4) if i == 0 else ''

    return pd.DataFrame(results)[['火灾风险类型', '标注数量', '模型识别数量', '正确数量', '精准率', '召回率', 'F1-score', '平均准确率']]

def generate_confusion_matrix(gt_dict, pred_dict, classes, output_path):
    # 提取真实标签和预测标签
    y_true = []
    y_pred = []
    for img_name in gt_dict:
        gt_cls = gt_dict[img_name]
        pred_cls = pred_dict.get(img_name, None)
        
        if gt_cls in classes and pred_cls in classes:
            y_true.append(gt_cls)
            y_pred.append(pred_cls)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # 设置中文字体
    import matplotlib.font_manager as fm

    font_path = '/host/algorithm2/zhanglu/code/SimHei.ttf'  # 替换为你自己的字体路径
    font_prop = fm.FontProperties(fname=font_path, size=12)

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)

    # 绘制热力图
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
        cbar=False,
        square=True,
        annot_kws={"fontproperties": font_prop}  # 注释文字使用字体
    )

    # 设置坐标轴标签字体
    ax.set_xticklabels(ax.get_xticklabels(), fontproperties=font_prop)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=font_prop)

    plt.xlabel("预测标签", fontproperties=font_prop)
    plt.ylabel("真实标签", fontproperties=font_prop)
    plt.title("混淆矩阵", fontproperties=font_prop)

    plt.tight_layout()

    # 保存图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_single_pair(excel_file, json_file):
    # 1. 读取Excel文件，筛选"是否存疑"为"否"的样本
    df_excel = pd.read_excel(excel_file)
    df_filtered = df_excel[df_excel['是否存疑'] == '否']

    # 提取真实标签：{filename -> class}
    gt_dict = dict(zip(df_filtered['文件名'], df_filtered['最终类别']))

    # 2. 读取JSON文件中的预测结果
    with open(json_file, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    pred_dict = {}
    for item in preds:
        image_path = item['image_path']
        filename = os.path.basename(image_path)
        pred_class = extract_class(item['result'])
        pred_dict[filename] = pred_class

    return gt_dict, pred_dict

if __name__ == '__main__':
    # 定义多组输入文件
    file_pairs = [
        {
            'excel': '/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级定义测试结果/人工标注结果-all/merged_risk_levels_output_1_0516.xlsx',
            'json': '/host/algorithm2/zhanglu/code/tmp_0521_fr3_xlsx1.json'
        },
        {
            'excel': '/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级定义测试结果/人工标注结果-all/merged_risk_levels_output_2_0516.xlsx',
            'json': '/host/algorithm2/zhanglu/code/tmp_0521_fr3_xlsx2.json'
        },
        {
            'excel': '/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级定义测试结果/人工标注结果-all/merged_risk_levels_output_3_0516.xlsx',
            'json': '/host/algorithm2/zhanglu/code/tmp_0521_fr3_xlsx3.json'
        },
    ]

    # 合并所有数据
    combined_gt_dict = {}
    combined_pred_dict = {}

    # 处理每组文件并合并结果
    for pair in file_pairs:
        gt_dict, pred_dict = process_single_pair(pair['excel'], pair['json'])
        
        # 合并到总体字典中
        combined_gt_dict.update(gt_dict)
        combined_pred_dict.update(pred_dict)

    # 计算合并后的指标并输出表格
    combined_metrics_df = compute_metrics(combined_gt_dict, combined_pred_dict, CLASSES)
    print("\n合并后总体结果:")
    print(combined_metrics_df.to_string(index=False))

    combined_metrics_df.to_csv('/host/algorithm2/zhanglu/code/tmp_0515_metrics_result_fangrui1_combined.csv', 
                              index=False, encoding='utf-8-sig')

    # 生成合并后的混淆矩阵
    generate_confusion_matrix(
        combined_gt_dict, combined_pred_dict, CLASSES,
        "/host/algorithm2/zhanglu/code/tmp_0515confusion_matrix_fangrui1_combined.png"
    )