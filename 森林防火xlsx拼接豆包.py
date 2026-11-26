import pandas as pd
import json
import os
from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.drawing.image import Image

def extract_result_parts(result_str):
    parts = result_str.strip().split()
    if len(parts) >= 2:
        return parts[0], parts[1]
    else:
        return result_str, None  # 如果没有第二部分，只保留第一部分，第二部分为 None

def main():
    # 输入路径
    excel_file = '/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级定义测试结果/人工标注结果-all/merged_risk_levels_output_1_0516.xlsx'
    json_file = '/host/algorithm2/zhanglu/code/tmp_0515_8.json'
    output_file = '/host/algorithm4/zhanglu/dataset/vlm/fire/风险等级定义测试结果/人工标注结果-all/merged_risk_levels_output_1_0516_doubao-8.xlsx'


    # 1. 使用 openpyxl 加载原始 Excel 文件，并保留图片
    wb = load_workbook(excel_file)
    ws = wb.active  # 获取当前活动工作表

    # 2. 提取原始 Excel 数据为 DataFrame
    df_excel = pd.DataFrame(ws.values)
    df_excel.columns = df_excel.iloc[0]  # 设置表头
    df_excel = df_excel[1:].reset_index(drop=True)  # 去掉表头行
    df_excel['文件名'] = df_excel['文件名'].astype(str).str.strip()

    # 3. 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        preds = json.load(f)

    # 构建字典：filename -> (risk_level, risk_code)
    pred_dict = {}
    for item in preds:
        image_path = item['image_path']
        filename = os.path.basename(image_path)
        result = item['result']
        level, code = extract_result_parts(result)
        pred_dict[filename] = (level, code)

    # 4. 新增两列
    df_excel['豆包-风险等级'] = ''
    df_excel['豆包-风险代码'] = ''

    # 填充新列
    for idx, row in df_excel.iterrows():
        filename = row['文件名']
        if filename in pred_dict:
            df_excel.at[idx, '豆包-风险等级'] = pred_dict[filename][0]
            df_excel.at[idx, '豆包-风险代码'] = pred_dict[filename][1]

    # 5. 将更新后的 DataFrame 写回 Excel 文件
    # 清空原工作表内容，但保留图片
    ws.delete_rows(1, ws.max_row)

    # 将 DataFrame 转换为行数据并写入工作表
    for r_idx, row in enumerate(dataframe_to_rows(df_excel, index=False, header=True), 1):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx, column=c_idx, value=value)

    # 6. 保存结果到新 Excel 文件
    wb.save(output_file)
    print(f"输出文件已保存至：{output_file}")

if __name__ == '__main__':
    main()
