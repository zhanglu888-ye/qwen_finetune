import os
import re
import json

# --- Your Risk Parsing Logic (from previous solution) ---
RISK_LEVEL_ORDER = ["正常", "疑似中风险", "中风险", "疑似高风险", "高风险"]
DEFAULT_RISK = "正常"
RISK_RULES_MAP = {}

def _parse_ids_from_rule_part(part_str):
    ids = set()
    if not part_str:
        return []
    sub_parts = part_str.split('、')
    for sub_part in sub_parts:
        sub_part = sub_part.strip()
        if not sub_part:
            continue
        if '-' in sub_part:
            start, end = map(int, sub_part.split('-'))
            for i in range(start, end + 1):
                ids.add(i)
        else:
            ids.add(int(sub_part))
    return sorted(list(ids))

def _populate_risk_rules():
    for event_id in range(1, 20):
        RISK_RULES_MAP[(event_id, 1)] = "高风险"

    high_risk_rules_specific = [
        "1+1-4", "2+1-4", "3+1-4", "4+1-4", "5+1-4",
        "6+1-2", "7+1-4", "8+1-4", "9+1"
    ]
    for rule_str in high_risk_rules_specific:
        event_part, smoke_part = rule_str.split('+', 1)
        events = _parse_ids_from_rule_part(event_part)
        smokes = _parse_ids_from_rule_part(smoke_part)
        for e_id in events:
            for s_id in smokes:
                RISK_RULES_MAP[(e_id, s_id)] = "高风险"

    event_part, smoke_part = "19+2、3".split('+', 1)
    events = _parse_ids_from_rule_part(event_part)
    smokes = _parse_ids_from_rule_part(smoke_part)
    for e_id in events:
        for s_id in smokes:
            if (e_id, s_id) not in RISK_RULES_MAP:
                RISK_RULES_MAP[(e_id, s_id)] = "疑似高风险"

    event_part, smoke_part = "9+2".split('+', 1)
    events = _parse_ids_from_rule_part(event_part)
    smokes = _parse_ids_from_rule_part(smoke_part)
    for e_id in events:
        for s_id in smokes:
            if (e_id, s_id) not in RISK_RULES_MAP:
                RISK_RULES_MAP[(e_id, s_id)] = "中风险"

    event_part, smoke_part = "19+4".split('+', 1)
    events = _parse_ids_from_rule_part(event_part)
    smokes = _parse_ids_from_rule_part(smoke_part)
    for e_id in events:
        for s_id in smokes:
            if (e_id, s_id) not in RISK_RULES_MAP:
                RISK_RULES_MAP[(e_id, s_id)] = "疑似中风险"

    normal_rules = [
        "0+6", "1+6", "6+3、4、6", "9+3-4", "10+3-4", "11+5",
        "12+4、6", "13-16+6", "17+3-4", "18+6"
    ]
    for rule_str in normal_rules:
        event_part, smoke_part = rule_str.split('+', 1)
        events = _parse_ids_from_rule_part(event_part)
        smokes = _parse_ids_from_rule_part(smoke_part)
        for e_id in events:
            for s_id in smokes:
                if (e_id, s_id) not in RISK_RULES_MAP:
                     RISK_RULES_MAP[(e_id, s_id)] = "正常"

_populate_risk_rules() # Populate the rules map

def parse_and_assess_risk(llm_output: str) -> str:
    lines = llm_output.strip().split('\n')
    if len(lines) != 2:
        return f"错误：LLM输出应有两行，实际为 {len(lines)} 行. 输出内容: '{llm_output}'"

    event_line = lines[0].strip()
    smoke_line = lines[1].strip()

    if not event_line.startswith("事件：") or not smoke_line.startswith("烟雾描述："):
        return f"错误：LLM输出格式不正确，缺少'事件：'或'烟雾描述：'前缀. 输出内容: '{llm_output}'"

    event_ids_str = event_line[len("事件："):].strip()
    smoke_ids_str = smoke_line[len("烟雾描述："):].strip()

    try:
        if not event_ids_str and not smoke_ids_str:
             event_ids = []
             smoke_ids = []
        elif not event_ids_str:
            return "错误：事件ID为空，但烟雾描述ID不为空"
        elif not smoke_ids_str:
            return "错误：烟雾描述ID为空，但事件ID不为空"
        else:
            event_ids = [int(x.strip()) for x in event_ids_str.split(',')]
            smoke_ids = [int(x.strip()) for x in smoke_ids_str.split(',')]
    except ValueError:
        return f"错误：事件或烟雾描述ID包含非整数值. 事件: '{event_ids_str}', 烟雾: '{smoke_ids_str}'"

    if len(event_ids) != len(smoke_ids):
        return (f"错误：事件ID数量 ({len(event_ids)}) 与烟雾描述ID数量 ({len(smoke_ids)}) 不匹配. "
                f"事件: '{event_ids_str}', 烟雾: '{smoke_ids_str}'")

    if not event_ids:
        return DEFAULT_RISK

    current_max_risk_index = -1
    for event_id, smoke_id in zip(event_ids, smoke_ids):
        risk_for_pair = RISK_RULES_MAP.get((event_id, smoke_id), DEFAULT_RISK)
        try:
            risk_index = RISK_LEVEL_ORDER.index(risk_for_pair)
            if risk_index > current_max_risk_index:
                current_max_risk_index = risk_index
        except ValueError:
            print(f"警告：风险等级 '{risk_for_pair}' (来自事件 {event_id}, 烟雾 {smoke_id}) 未在 RISK_LEVEL_ORDER 中定义。")

    if current_max_risk_index == -1:
        return DEFAULT_RISK
    else:
        return RISK_LEVEL_ORDER[current_max_risk_index]

# --- OpenAI API Interaction ---

# The detailed prompt you provided
SYSTEM_PROMPT = """
# 角色
你是一名专业的图像分析师，专门负责识别可见光图片中的火灾事件和烟雾现象。你的判断必须严格基于图片中清晰可见的视觉证据。

""" # Truncated for brevity in this script, but use your full prompt

def get_llm_analysis(api_key: str, image_description: str, model_name: str = "gpt-3.5-turbo"):
    """
    Calls the OpenAI API to get fire/smoke analysis based on an image description.

    Args:
        api_key: Your OpenAI API key.
        image_description: A textual description of the image content.
        model_name: The OpenAI model to use.

    Returns:
        The raw response string from the LLM, or None if an error occurs.
    """
    try:
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"请分析以下描述的图像场景：{image_description}"}
            ],
            temperature=0.1,  # Lower temperature for more deterministic output
            max_tokens=100     # Enough for the two lines of output
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"OpenAI API调用错误: {e}")
        return None

if __name__ == "__main__":

    input_json = '/host/algorithm2/zhanglu/code/tmp_0522_ls3_xlsx1.json'  # 替换为你的输入JSON文件路径
    output_json = '/host/algorithm2/zhanglu/code/tmp_0522_ls3_xlsx1_new.json'  # 替换为你的输出JSON文件路径
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理每个条目
    processed_data = []
    for item in data:
        # 创建新条目，保留原始数据
        new_item = item.copy()
        
        # 处理result_model字段
        if 'result_model' in item:
            print(item['result_model'])
            risk_1 = parse_and_assess_risk(item['result_model'])
            print(risk_1)
            new_item['result'] = risk_1
        processed_data.append(new_item)
        # 保存处理后的JSON文件
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)