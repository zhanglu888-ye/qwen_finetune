import asyncio
import csv
import json
import os
from io import StringIO
from llm_util import LlmUtil
from tqdm import tqdm


prompt_cut = f"""“原始信息单元”指的是最小的独立信息片段，它们在不丢失任何细节的情况下表示句子的整个含义，通常描述图像中视觉元素的各种属性。原始信息单元应该是一个简单的语句。事实必须代表最小的信息，不能进一步分解。抽象概念或广义解释应简化为更基本的、组成性的观察。基本信息单元应该只包含一个基本元素。 
当从图像标题中提取原始信息单元时，要为主要对象或实体分配唯一标识符。这有助于保持清晰度和防止混淆，特别是当有多个类似的对象或实体时。例如，如果标题提到两只猫，您可以分配唯一标识符，如“猫1”和“猫2”来区分它们。此外，对于每个属性，还应该将标识符分配给它所属的对象。同时，对于空间关系，你可以将标识符分配给原信息单元中的主体对象。 
对于每个原始信息单元，您还需要证明原始信息单元是否直接描述图像。 
**重要**：请提取图像标题中的所有原始信息单元。不要遗漏任何信息，也不要重复！ 
输出应该是字典 [{{"事实": 原始信息单元, "标识符": 唯一标识, "相关性": 1/0}},...]的JSON格式。如果事实中的项已经用id标识过，那么“标识符”可以没有。如果原始信息单元直接描述图像的内容，则“相关性”为1。否则，如果原语信息单元是对描述的推理或扩展，而不是对图像内容的直接描述，则为0。 
请理解以上内容，并等待输入。"""


prompt_match = f"""
你将接收到预测原子事实集和基准原子事实集（真实标注）。原子事实集以JSON格式的字典列表表示[{{"事实": 原子事实, "标识符": id}}, ...]。

执行原子事实匹配的流程如下：
1. 初步审查：对两组原子事实进行全面审阅，理解每个原子事实的细节和上下文
2. 推断标识符映射：需要判断每个预测原子事实集，是否能匹配上基准原子事实集，匹配成功标记基准原子事实集的标识符id，匹配不成功标记None。
   例如预测集的"cat_0"可能对应基准集的"cat_1"。需考虑：
   - 属性特征匹配（颜色/材质/状态）
   - 空间关系一致性（位置/方向）
   - 允许近似映射（无需完全精确匹配）
   - 无匹配时"匹配ID"标记为"None",匹配成功时"匹配ID"标记为"id_n",n为基准集中的原子事实序列，例如匹配成功基准集中的第1个原子事实，"匹配ID": "id_0"，匹配上基准集中的第5个原子事实，"匹配ID": "id_4"

**重要要求**：
- 预测原子事实集的每个事实都要判断是否匹配，不能遗漏
- 输出格式严格遵循JSON字典列表
- 输出的的"事实"、"标识符"和"相关性"要与预测原子事实集的“事实”完全一致，不能有任何变化，只增加"匹配ID"。

输出示例：
[
    {{
        "事实": "图像1的标识信息为‘西梁格北 - 热成像’。",
        "标识符": "图像1",
        "相关性": "1",
        "匹配ID": "id_0"
    }},
    ...
]
"""


prompt_verify = f"""
预测原子事实以 JSON 格式的列表呈现，即 [{{"事实": 原子事实, "标识符": 唯一标识}}, ...]。标识符具有唯一性，用于识别所讨论的主要对象或实体，这有助于保持内容清晰，避免混淆（尤其是当说明文字提及多个相似对象时）。例如，若说明文字提到两只猫，我们会分配“cat_1”和“cat_2”这样的唯一标识符加以区分。  
你需先梳理所有预测原子事实，理解每个原子事实的细节与上下文，再通过自问验证其正确性：  
陈述：“原子事实”  
根据图像和参考字幕，该陈述是否正确？如果正确verification就是1，否则为0

**重要要求**：
- 输出的“事实”应与预测集原子事实的“事实”保持一致。

输出需格式化为 JSON 列表，即 [{{"事实": 原子事实, "标识符": 唯一标识, "verification": 1/0}}, ...]，其中 1 表示事实正确，0 表示事实错误。字典中的其他键与输入一致。
"""

img_dir = "/host/algorithm4/zhanglu/dataset/vlm/eval_fire/3000image"


record = {"图片": "", "豆包段落描述": "","豆包拆分结果": "",  "人工段落描述": "","人工拆分结果":"", "qwen段落描述": "", "qwen拆分结果":"", "豆包-人工匹配结果":"", "qwen-人工匹配结果":"", "豆包-人工验证结果":"", "qwen-人工验证结果":"", "风险等级-豆包":"", "风险等级-人工":"", "风险等级-qwen":""}

# 初始化CSV文件
def init_csv_header(csv_path_save):
    if not os.path.exists(csv_path_save):
        with open(csv_path_save, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=["图片", "豆包段落描述", "豆包拆分结果", "人工段落描述", "人工拆分结果", "qwen段落描述", "qwen拆分结果", "豆包-人工匹配结果", "qwen-人工匹配结果", "豆包-人工验证结果", "qwen-人工验证结果", "风险等级-豆包", "风险等级-人工", "风险等级-qwen"],
                extrasaction='ignore'
            )
            writer.writeheader()

async def Process_splitting(llm_util):
    csv_file = "/host/algorithm4/zhanglu/dataset/vlm/eval_fire/caption_0416/fusion0.csv"
    csv_path_save = "/host/algorithm4/zhanglu/dataset/vlm/eval_fire/caption_0416/fusion0_tmp_result.csv"
    init_csv_header(csv_path_save)  # 确保表头存在

    
    jsonl_data = []

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # 遍历每一行
        for row in reader:
            jsonl_data.append(row)



        # 使用追加模式打开输出文件
        with open(csv_path_save, 'a', encoding='utf-8-sig', newline='') as output_f:
            writer = csv.DictWriter(
                output_f,
                fieldnames=["图片", "豆包段落描述", "豆包拆分结果", "人工段落描述", "人工拆分结果", "qwen段落描述", "qwen拆分结果", "豆包-人工匹配结果", "qwen-人工匹配结果", "豆包-人工验证结果", "qwen-人工验证结果", "风险等级-豆包", "风险等级-人工", "风险等级-qwen"],
                extrasaction='ignore'
            )
            for index, x in tqdm(enumerate(jsonl_data)):
                ##段落拆分
                print("正在处理第{}个\n".format(index))
                doubao_desc = x['段落描述-豆包'].split('第二张图像')[0].strip('第一张图像')
                person_desc = x['段落描述-人工'].split('第二张图像')[0].strip('第一张图像')
                qwen_desc = x['段落描述-Qwen'].split('第二张图像')[0].strip('第一张图像')
                if '可见光' in doubao_desc:
                    doubao_desc_light = doubao_desc
                else:
                    doubao_desc_light = x['段落描述-豆包'].split('第二张图像')[1]
                if '可见光' in person_desc:
                    person_desc_light = person_desc
                else:
                    person_desc_light = x['段落描述-人工'].split('第二张图像')[1]
                if '可见光' in qwen_desc:
                    qwen_desc_light = qwen_desc
                else:
                    qwen_desc_light = x['段落描述-Qwen'].split('第二张图像')[1]                               


                record["图片"] = x['图片']
                record["豆包段落描述"] = x['段落描述-豆包']
                record["人工段落描述"] = x['段落描述-人工']
                record["qwen段落描述"] = x['段落描述-Qwen']
                record["风险等级-豆包"] = x['风险等级-豆包']
                record["风险等级-人工"] = x['风险等级-人工']
                record["风险等级-qwen"] = x['风险等级-Qwen']



                # 豆包处理
                if doubao_desc != '':
                    try:
                        full_prompt = f"{prompt_cut}\n输入：【{doubao_desc_light}】"
                        response = await llm_util.call_doubao_noimg(full_prompt)
                        if response == '```':
                            print("kong")
                        record["豆包拆分结果"] = response.strip("```json\n")
                    except Exception as e:
                        record["豆包拆分结果"] = f"豆包处理错误: {str(e)}"
                        print(f"豆包处理失败: {str(e)}")

                # 人工处理
                if person_desc != '':
                    try:
                        full_prompt = f"{prompt_cut}\n输入：【{person_desc_light}】"
                        response = await llm_util.call_doubao_noimg(full_prompt)
                        if response == '```':
                            print("kong")
                        record["人工拆分结果"] = response.strip("```json\n")
                    except Exception as e:
                        record["人工拆分结果"] = f"人工处理错误: {str(e)}"
                        print(f"人工处理失败: {str(e)}")

                # Qwen处理
                if qwen_desc != '':
                    try:
                        full_prompt = f"{prompt_cut}\n输入：【{qwen_desc_light}】"
                        response = await llm_util.call_doubao_noimg(full_prompt)
                        if response == '```':
                            print("kong")
                        record["qwen拆分结果"] = response.strip("```json\n")
                    except Exception as e:
                        record["qwen拆分结果"] = f"Qwen处理错误: {str(e)}"
                        print(f"Qwen处理失败: {str(e)}")

                # 立即写入当前记录
                writer.writerow(record)
                output_f.flush()  # 强制刷新缓冲区
                os.fsync(output_f.fileno())  # 确保写入磁盘
    return
async def Process_matching(llm_util):
    csv_file = "/host/algorithm4/zhanglu/dataset/vlm/eval_fire/caption_0416/fusion0_tmp_result.csv"
    csv_path_save = "/host/algorithm4/zhanglu/dataset/vlm/eval_fire/caption_0416/fusion0_tmp_result_stage2.csv"
    init_csv_header(csv_path_save)  # 确保表头存在
    jsonl_data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # 遍历每一行
        for row in reader:
            jsonl_data.append(row)
        # 使用追加模式打开输出文件
        with open(csv_path_save, 'a', encoding='utf-8-sig', newline='') as output_f:
            writer = csv.DictWriter(
                output_f,
                fieldnames=["图片", "豆包段落描述", "豆包拆分结果", "人工段落描述", "人工拆分结果", "qwen段落描述", "qwen拆分结果", "豆包-人工匹配结果", "qwen-人工匹配结果", "豆包-人工验证结果", "qwen-人工验证结果", "风险等级-豆包", "风险等级-人工", "风险等级-qwen"],
                extrasaction='ignore'
            )
            for index, x in tqdm(enumerate(jsonl_data)):
                ##段落拆分
                print("正在处理第{}个\n".format(index))
                doubao_desc_split = x['豆包拆分结果']
                person_desc_split = x['人工拆分结果']
                qwen_desc_split = x['qwen拆分结果']
                record["图片"] = x['\ufeff图片']
                record["豆包段落描述"] = x['豆包段落描述']
                record["豆包拆分结果"] = x['豆包拆分结果']
                record["人工段落描述"] = x['人工段落描述']
                record["人工拆分结果"] = x['人工拆分结果']
                record["qwen段落描述"] = x['qwen段落描述']
                record["qwen拆分结果"] = x['qwen拆分结果']
                record["风险等级-豆包"] = x['风险等级-豆包']
                record["风险等级-人工"] = x['风险等级-人工']
                record["风险等级-qwen"] = x['风险等级-qwen']

                # 豆包-人工匹配结果
                if doubao_desc_split !='' and person_desc_split != '':
                    # try:
                    full_prompt = f"{prompt_match}\n预测原子事实集：【{doubao_desc_split}】\n基准原子事实集：【{person_desc_split}】"
                    # print(full_prompt)
                    response = await llm_util.call_doubao_noimg(full_prompt)
                    if response == '```':
                        print("kong")
                    record["豆包-人工匹配结果"] = response.strip("```json\n")
                    # except Exception as e:
                    #     record["豆包-人工匹配结果"] = f"豆包-人工匹配结果错误: {str(e)}"
                    #     print(f"豆包-人工匹配结果: {str(e)}")

                # qwen-人工匹配结果
                if qwen_desc_split !='' and person_desc_split !='' :
                    try:
                        full_prompt = f"{prompt_match}\n预测原子事实集：【{qwen_desc_split}】\n基准原子事实集：【{person_desc_split}】"
                        response = await llm_util.call_doubao_noimg(full_prompt)
                        if response == '```':
                            print("kong")
                        record["qwen-人工匹配结果"] = response.strip("```json\n")
                    except Exception as e:
                        record["qwen-人工匹配结果"] = f"qwen-人工匹配结果错误: {str(e)}"
                        print(f"qwen-人工匹配结果: {str(e)}")
                # 立即写入当前记录
                writer.writerow(record)
                output_f.flush()  # 强制刷新缓冲区
                os.fsync(output_f.fileno())  # 确保写入磁盘






async def Process_verifing(llm_util):
    csv_file = "/host/algorithm4/zhanglu/dataset/vlm/eval_fire/caption_0416/fusion0_tmp_result_stage2.csv"
    csv_path_save = "/host/algorithm4/zhanglu/dataset/vlm/eval_fire/caption_0416/fusion0_tmp_result_stage3.csv"
    init_csv_header(csv_path_save)  # 确保表头存在
    jsonl_data = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        
        # 遍历每一行
        for row in reader:
            jsonl_data.append(row)
        # 使用追加模式打开输出文件
        with open(csv_path_save, 'a', encoding='utf-8-sig', newline='') as output_f:
            writer = csv.DictWriter(
                output_f,
                fieldnames=["图片", "豆包段落描述", "豆包拆分结果", "人工段落描述", "人工拆分结果", "qwen段落描述", "qwen拆分结果", "豆包-人工匹配结果", "qwen-人工匹配结果", "豆包-人工验证结果", "qwen-人工验证结果", "风险等级-豆包", "风险等级-人工", "风险等级-qwen"],
                extrasaction='ignore'
            )
            for index, x in tqdm(enumerate(jsonl_data)):
                ##段落拆分
                print("正在处理第{}个\n".format(index))
                doubao_desc_split = x['豆包拆分结果']
                person_desc = x['人工段落描述'].split('第二张图像')[0].strip('第一张图像')

                if '可见光' in person_desc:
                    person_desc_light = person_desc
                else:
                    person_desc_light = x['人工段落描述'].split('第二张图像')[1]



                qwen_desc_split = x['qwen拆分结果']
                record["图片"] = x['\ufeff图片']
                record["豆包段落描述"] = x['豆包段落描述']
                record["豆包拆分结果"] = x['豆包拆分结果']
                record["人工段落描述"] = x['人工段落描述']
                record["人工拆分结果"] = x['人工拆分结果']
                record["qwen段落描述"] = x['qwen段落描述']
                record["qwen拆分结果"] = x['qwen拆分结果']
                record["豆包-人工匹配结果"] = x['豆包-人工匹配结果']
                record["qwen-人工匹配结果"] = x['qwen-人工匹配结果']
                record["风险等级-豆包"] = x['风险等级-豆包']
                record["风险等级-人工"] = x['风险等级-人工']
                record["风险等级-qwen"] = x['风险等级-qwen']



                img_name = x['\ufeff图片'].split('/')[1]
                img_path_ir = os.path.join(img_dir,img_name+'0_first.jpg')
                img_path_light = os.path.join(img_dir,img_name+'1_second.jpg')
                images = [img_path_light]
                # if '可见光' in person_desc:
                #     images = [img_path_light]
                # else:
                #     images = [img_path_ir]
                # 豆包-人工验证结果
                if doubao_desc_split !='':
                    try:
                        full_prompt = f"{prompt_verify}\n参考字幕：【{person_desc_light}】\n预测原子事实集：【{doubao_desc_split}】"
                        # print(full_prompt)
                        response = await llm_util.call_doubao(full_prompt, [llm_util.path2base64(img) for img in images])
                        if response == '```':
                            print("kong")
                        record["豆包-人工验证结果"] = response.strip("```json\n")
                    except Exception as e:
                        record["豆包-人工验证结果"] = f"豆包-人工验证结果错误: {str(e)}"
                        print(f"豆包-人工验证结果: {str(e)}")

                # qwen-人工匹配结果
                if qwen_desc_split !='':
                    try:
                        full_prompt = f"{prompt_verify}\n参考字幕：【{person_desc_light}】\n预测原子事实集：【{qwen_desc_split}】"
                        response = await llm_util.call_doubao(full_prompt, [llm_util.path2base64(img) for img in images])
                        if response == '```':
                            print("kong")
                        record["qwen-人工验证结果"] = response.strip("```json\n")
                    except Exception as e:
                        record["qwen-人工验证结果"] = f"qwen-人工验证结果错误: {str(e)}"
                        print(f"qwen-人工验证结果: {str(e)}")
                # 立即写入当前记录
                writer.writerow(record)
                output_f.flush()  # 强制刷新缓冲区
                os.fsync(output_f.fileno())  # 确保写入磁盘

async def main():
    ##段落拆分
    # llm_util = LlmUtil(sys_content="你是在图像标题中提取原始信息单位的语言学专家。")
    # await Process_splitting(llm_util)
    llm_util = LlmUtil(sys_content="你现在是视觉-语言领域的专家，负责匹配两组从不同描述生成的原子事实集合。")
    await Process_matching(llm_util)
    llm_util = LlmUtil(sys_content="你是一位卓越的视觉语言专家，负责依据图像及对应的参考说明文字，验证一组原子事实的正确性。这组原子事实提取自一段机器生成的图像说明文字。")
    await Process_verifing(llm_util)

    '''
    ##示例
    images = ["/host/algorithm2/zhanglu/copy_1kwmog34x3uxr_infer.jpg"]
    result = await llm_util.call_doubao("描述图片", [llm_util.path2base64(img) for img in images])
    print(result)
    '''






if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
