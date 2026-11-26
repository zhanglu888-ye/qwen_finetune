import csv
import re

class CustomDialect(csv.excel):
    quoting = csv.QUOTE_MINIMAL
    escapechar = '\\'

# 定义要去掉的标题模式
SECTION_HEADERS = [
    r"\*\*图像标识与基本情况:\*\*",
    r"\*\*场景环境概览:\*\*",
    r"\*\*核心要素分析：火情/烟雾/热源详述:\*\*",
    r"\*\*潜在误报因素分析:\*\*",
    r"\*\*对比分析发现：\*\*",
    r"\*\*综合风险判断：\*\*",
    r"\*\*综合评估与结论:\*\*",
    r"\*\*对比分析与关键发现：\*\*",
    r"\*\*综合风险判断与结论：\*\*"
]

def clean_section(text):
    """
    清理文本中的特殊标题标签，保留其后的内容
    """
    pattern = '|'.join(SECTION_HEADERS)
    cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r"^[:\s]+", '', cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()

def remove_empty_lines(text):
    if not text:
        return ""
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)

def process_row(row):
    result = {
        "图片名称": row["图片名称"],
        "图片类型-可见光": row["图片类型-可见光"],
        "图片描述-可见光": row["图片描述-可见光"],
        "图片类型-热成像": row["图片类型-热成像"],
        "图片描述-热成像": row["图片描述-热成像"],
        "一组描述": row["一组描述"],
        "结构化描述（一组）": "",
        "段落描述（一组）": ""
    }

    structured_parts = []

    # 判断真实可见光和热成像描述来源
    visible_desc = None
    thermal_desc = None

    if row["图片类型-可见光"] == "可见光":
        visible_desc = row["图片描述-可见光"]
    elif row["图片类型-热成像"] == "可见光":
        visible_desc = row["图片描述-热成像"]

    if row["图片类型-可见光"] == "热成像":
        thermal_desc = row["图片描述-可见光"]
    elif row["图片类型-热成像"] == "热成像":
        thermal_desc = row["图片描述-热成像"]

    # 处理可见光描述（截断综合评估）
    if visible_desc:
        parts = re.split(r"\*\*综合评估与结论:\*\*", visible_desc, flags=re.DOTALL)
        visible_desc_cleaned = parts[0].strip()
        structured_parts.append("可见光：\n" + visible_desc_cleaned)

    # 处理热成像描述（截断综合评估）
    if thermal_desc:
        parts = re.split(r"\*\*综合评估与结论:\*\*", thermal_desc, flags=re.DOTALL)
        thermal_desc_cleaned = parts[0].strip()
        if visible_desc:  # 如果同时有可见光，加换行
            structured_parts.append("\n\n热成像：\n" + thermal_desc_cleaned)
        else:
            structured_parts.append("热成像：\n" + thermal_desc_cleaned)

    # 一组描述处理（始终保留完整）
    group_desc = row["一组描述"].strip() if row["一组描述"] else ""
    if group_desc:
        structured_parts.append("\n\n综合分析：\n" + group_desc)

    result["结构化描述（一组）"] = "".join(structured_parts)

    # ========== 新增逻辑：生成段落描述（一组） ==========
    structured_text = result["结构化描述（一组）"]

    if not structured_text:
        result["段落描述（一组）"] = ""
    else:
        # 分割为“可见光”、“热成像”、“一组描述”三个部分
        sections = re.split(r"(可见光：|热成像：|综合分析：)", structured_text, flags=re.IGNORECASE)

        final_paragraphs = []
        i = 1
        while i < len(sections):
            section_title = sections[i].strip()
            section_content = sections[i + 1] if i + 1 < len(sections) else ""

            # 清理该部分内容
            cleaned_content = clean_section(section_content)

            if cleaned_content:
                final_paragraphs.append(f"{section_title} {cleaned_content}")

            i += 2

        result["段落描述（一组）"] = "\n".join(final_paragraphs)

    return result


def process_csv(input_file, output_file):
    csv.register_dialect('custom', CustomDialect)

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:

        reader = csv.DictReader(infile)

        fieldnames = [
            "图片名称",
            "图片类型-可见光",
            "图片描述-可见光",
            "图片类型-热成像",
            "图片描述-热成像",
            "一组描述",
            "结构化描述（一组）",
            "段落描述（一组）"
        ]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames, dialect='custom')
        writer.writeheader()

        for row in reader:
            processed_row = process_row(row)

            # 去除空行（可选）
            # for field in fieldnames:
            #     if field in processed_row and isinstance(processed_row[field], str):
            #         processed_row[field] = remove_empty_lines(processed_row[field])

            writer.writerow(processed_row)


if __name__ == '__main__':
    input_csv = '/host/algorithm4/zhanglu/dataset/vlm/fire/output_合并描述.csv'
    output_csv = '/host/algorithm4/zhanglu/dataset/vlm/fire/output_合并描述_desc.csv'
    process_csv(input_csv, output_csv)