import io
import os
import re
import json
import base64
import argparse
from collections import defaultdict
import copy

import ast
from tqdm import tqdm
import numpy as np
from PIL import Image
# import openai
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from matplotlib.font_manager import FontProperties
import torch

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_not_exception_type
)
import traceback

import csv

classes = ["高风险", "中风险", "正常", "无法判断"]

def normalize_text(text):
    # 修复不完整的引号
    text = re.sub(r'‘(.*?)\.{0,1}$', r'‘\1’', text)  # 如果引号不完整，补全右引号
    text = re.sub(r'“(.*?)\.{0,1}$', r'“\1”', text)  # 同理处理双引号
    # 去除多余的空格
    text = re.sub(r'\s+', ' ', text).strip()
    # 确保句子以句号结尾
    if not text.endswith('。'):
        text += '。'
    return text

def extract_from_dict_list(dict_list, target_fields):
    """从字典列表中提取指定字段"""
    extracted_data = []
    for item in dict_list:
        extracted = {field: item.get(field, "") for field in target_fields}
        extracted_data.append(extracted)
    return extracted_data


import re
from collections import defaultdict

def compute_precision_recall(predicted_atomic_facts, oracle_atomic_facts, verification_results, mappings):
    # --- 新增步骤：标准化 mappings 的键 ---
    normalized_mappings = {
        normalize_text(k): v for k, v in mappings.items()
    }
    
    mapped_prediction = verification_results.copy()
    
    # --- 修改集合生成逻辑：使用标准化后的文本 ---
    a = set(normalized_mappings.keys())
    # 对预测事实进行标准化
    b = set([normalize_text(x['事实']) for x in mapped_prediction])
    
    # --- 修复对称差集逻辑 ---
    # 有区别时忽略，不计算。这里后续待优化
    difference = a.symmetric_difference(b)
    for item in list(difference):  
        if item in b:
            b.remove(item)
        if item in a:
            a.remove(item)
    
    # --- 更新断言和检查逻辑 ---
    print('\n标准化后 Mapping Num == Prediction Num:', a == b)
    assert a == b, f"集合不一致: {a.symmetric_difference(b)}"
    
    # --- 更新预测事实的匹配逻辑 ---
    predict_ids = defaultdict(int)
    for d in mapped_prediction:
        # 标准化当前事实
        normalized_fact = normalize_text(d['事实'])
        d['matched_oracle_id'] = normalized_mappings.get(normalized_fact)   
        
        if d['matched_oracle_id'] not in (None, 'None'):
            predict_ids[d['matched_oracle_id']] += 1          
    
    # --- 后续计算保持不变，但使用标准化后的匹配结果 ---
    predict_weights = {k: 1/v for k, v in predict_ids.items()}     
    correct = 0
    total_atomic_facts = len(oracle_atomic_facts)
    print('无效匹配数量:', len([d for d in mapped_prediction if d['matched_oracle_id'] in (None, 'None')]))
    correct_prec = 0
    for d in mapped_prediction:    
        if d['verification'] == 1:
            if d['matched_oracle_id'] in (None, 'None'):    
                correct += 1
                correct_prec += 1
                total_atomic_facts += 1
            else:        
                correct += (1 * predict_weights[d['matched_oracle_id']])
                correct_prec += 1
                
    precision = correct_prec / len(mapped_prediction)
    recall = correct / total_atomic_facts
    
    print('\n最终结果:')
    print(f'总原子事实数: {total_atomic_facts}')
    print(f'精确率: {precision:.4f}')
    print(f'召回率: {recall:.4f}')
    
    ### Compute relevance
    predicted_relevance = set([x['事实'] for x in predicted_atomic_facts if x['相关性'] == 1])
    mapped_prediction_relevant = [x for x in mapped_prediction if x['事实'] in predicted_relevance]
    predict_ids_relevant = defaultdict(int)
    for d in mapped_prediction_relevant:
        d['matched_oracle_id'] = mappings[d['事实']]
        if d['matched_oracle_id'] is not None and d['matched_oracle_id'] != 'None':
            predict_ids_relevant[d['matched_oracle_id']] += 1
    
    predict_weights_relevant = {k: 1/v for k, v in predict_ids_relevant.items()}
    correct_relevant = 0
    total_atomic_facts_relevant = len([x for x in oracle_atomic_facts if x['relevance'] == 1])

    correct_prec_relevant = 0
    for d in mapped_prediction_relevant:    
        if d['verification'] == 1 and (d['matched_oracle_id'] is None or d['matched_oracle_id'] == 'None'):
            correct_relevant += 1
            correct_prec_relevant += 1
            total_atomic_facts_relevant += 1
        
        elif d['verification'] == 1:
            assert predict_weights[d['matched_oracle_id']] > 0
            correct_relevant += (1 * predict_weights[d['matched_oracle_id']])
            correct_prec_relevant += 1
            
    precision_relevant = correct_prec_relevant / len(mapped_prediction_relevant)
    recall_relevant = correct_relevant / total_atomic_facts_relevant    
    
    print('Total Number of Relevant Atomic Facts:', total_atomic_facts_relevant)
    print('Precision (Relevant):', correct_prec_relevant / len(mapped_prediction_relevant))
    print('Recall (Relevant):', correct_relevant / total_atomic_facts_relevant)
    
    res = {
        "precision": precision,
        "recall": recall,
        "f1_score": 2 * precision * recall / (precision + recall + 1e-8),
        "total_atomic_facts": total_atomic_facts,
        
        "precision_relevant": precision_relevant,
        "recall_relevant": recall_relevant,
        "f1_score_relevant": 2 * precision_relevant * recall_relevant / (precision_relevant + recall_relevant + 1e-8),
        "total_atomic_facts_relevant": total_atomic_facts_relevant,
    }
    return res


def data2eval(idx_cap):
    unevals = []
    for idx, cap in idx_cap:
        unevals.append([idx, cap])

    return unevals


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_file', type=str, default="/host/algorithm4/zhanglu/dataset/vlm/fire/eval_fire/caption_0416/fusion0_tmp_result2_stage3.csv")
    parser.add_argument('--save_folder', type=str, default="/host/algorithm2/zhanglu/code/vlmeval/DeCapBench-main/eval/风险等级结果.png")

    args = parser.parse_args()
    return args


def main():
    args = argparser()
    caption_file = args.caption_file
    save_path = args.save_folder
    jsonl_data = []
    with open(caption_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # 遍历每一行
        for row in reader:
            # 直接将每行字典添加到列表（可选）
            jsonl_data.append(row)
            
            # 转换为 JSON 字符串并打印（模仿 JSON Lines）
            json_line = json.dumps(row, ensure_ascii=False)

    oracle_facts_map = {x['\ufeff图片']: x for x in jsonl_data}
    
    total_size = len(oracle_facts_map)
    
    required_fields = ["\ufeff图片", "人工拆分结果", "qwen拆分结果", "qwen-人工匹配结果", "qwen-人工验证结果", "风险等级-qwen", "风险等级-人工", "风险等级-qwen"]  # 需要提取的字段
    
    # 执行提取
    annotations = extract_from_dict_list(jsonl_data, required_fields)  


    captions_part = annotations[:]
    idx_part = list(range(0, total_size))
    idx_captions_part = list(zip(idx_part, captions_part))
    
    idx_captions_part_to_eval = data2eval(idx_captions_part)
    print('Loading finished')
    
    all_results = []

    MAX_TRIES = 1
    print('Start collecting')
    final_result = {'trials': []}

    #####计算风险等级混淆矩阵
    # 提取人工标注和预测结果
    y_true = [cap[1]['风险等级-人工'] for cap in idx_captions_part_to_eval]
    y_pred = [cap[1]['风险等级-qwen'] for cap in idx_captions_part_to_eval]

    # 标签编码
    le = LabelEncoder()
    le.fit(classes)
    y_true_encoded = le.transform(y_true)
    y_pred_encoded = le.transform(y_pred)

    # 计算评估指标
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted')
    recall = recall_score(y_true_encoded, y_pred_encoded, average='weighted')
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    
    # 打印指标
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"加权精确率 (Weighted Precision): {precision:.4f}")
    print(f"加权召回率 (Weighted Recall): {recall:.4f}")
    
    # 打印每个类别的指标
    print("\n各类别详细指标:")
    for i, class_name in enumerate(classes):
        class_precision = precision_score(y_true_encoded, y_pred_encoded, labels=[i], average='micro')
        class_recall = recall_score(y_true_encoded, y_pred_encoded, labels=[i], average='micro')
        print(f"{class_name}: 精确率={class_precision:.4f}, 召回率={class_recall:.4f}")

    # 绘制混淆矩阵
    font = FontProperties(fname='/host/algorithm2/zhanglu/code/SimHei.ttf')  # 替换为字体文件路径
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontproperties=font)
    plt.yticks(tick_marks, classes, fontproperties=font)

    # 添加数值标签
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")



    plt.title('混淆矩阵', fontproperties=font)
    plt.xlabel('预测标签', fontproperties=font)
    plt.ylabel('真实标签', fontproperties=font)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n混淆矩阵已保存至: {save_path}")



    for idx, caption in tqdm(idx_captions_part_to_eval, desc="Part {}/{}".format(1, 1)):
        for _ in range(MAX_TRIES):
            cap_dict = caption.copy()
            if caption['\ufeff图片']=='' or caption['人工拆分结果']=='' or caption['qwen拆分结果']=='' or caption['qwen-人工匹配结果']=='' or caption['qwen-人工验证结果']=='':
                continue

            d = json.loads(oracle_facts_map[caption['\ufeff图片']]['人工拆分结果'])
            cap_dict["human_annotated_oracle_atomic_facts"] = [
                {"id": fact_idx, "fact": fact}
                for fact_idx, fact in enumerate(d)
            ]
            
            eval_dict = {'consume_token': []}
            
            parsed_pred_atomic_facts = json.loads(caption['qwen拆分结果'])

            try:
                match_results_with_human = json.loads(caption['qwen-人工匹配结果'])
            except:
                continue
            mappings_with_human = {d['事实']: d['匹配ID'] for d in match_results_with_human}   ##相关性还是标识符？,应该是匹配的id号
            
            verified_results = json.loads(caption['qwen-人工验证结果'])


            #######################
            human_annotated_oracle_atomic_facts = cap_dict["human_annotated_oracle_atomic_facts"]
            for d in human_annotated_oracle_atomic_facts:
                d['relevance'] = 1
            
            res = compute_precision_recall(parsed_pred_atomic_facts, human_annotated_oracle_atomic_facts, verified_results, mappings_with_human)
            final_result['trials'].append(res)

                    
    final_result['precision_avg'] = sum([x['precision'] for x in final_result['trials']]) / len(final_result['trials'])
    final_result['recall_avg'] = sum([x['recall'] for x in final_result['trials']]) / len(final_result['trials'])
    final_result['f1_score_avg'] = sum([x['f1_score'] for x in final_result['trials']]) / len(final_result['trials'])
    
    final_result['precision_relevant_avg'] = sum([x['precision_relevant'] for x in final_result['trials']]) / len(final_result['trials'])
    final_result['recall_relevant_avg'] = sum([x['recall_relevant'] for x in final_result['trials']]) / len(final_result['trials'])
    final_result['f1_score_relevant_avg'] = sum([x['f1_score_relevant'] for x in final_result['trials']]) / len(final_result['trials'])
    eval_dict['scores'] = final_result
    
    print('Precision:', final_result['precision_avg'])
    print('Recall:', final_result['recall_avg'])
    print('F1 Score:', final_result['f1_score_avg'])
    print('Precision (Relevant):', final_result['precision_relevant_avg'])
    print('Recall (Relevant):', final_result['recall_relevant_avg'])
    print('F1 Score (Relevant):', final_result['f1_score_relevant_avg'])
    
    eval_dict['precision'] = final_result['precision_avg']
    eval_dict['recall'] = final_result['recall_avg']
    eval_dict['f1_score'] = final_result['f1_score_avg']
    
    eval_dict['precision_relevant'] = final_result['precision_relevant_avg']
    eval_dict['recall_relevant'] = final_result['recall_relevant_avg']
    eval_dict['f1_score_relevant'] = final_result['f1_score_relevant_avg']

    
    cap_dict['eval_details'] = eval_dict
    cap_dict['valid'] = True

    all_results.append(cap_dict)

if __name__ == '__main__':
    main()