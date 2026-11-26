import asyncio
import os
import json
import pandas as pd
from llm_util import LlmUtil
from tqdm import tqdm
import argparse  # 新增：用于命令行参数解析
import re
from PIL import Image


##提示词1
prompt_double = """**角色：** 你是一个专业的人体与场景分析AI，具备敏锐的观察力和严谨的逻辑推理能力。你的任务是分析图像，并以高度结构化的JSON格式输出对特定人物的详细描述。

**输入：**
1.  **[全景图]**: 一张包含完整场景和多个元素的宽幅或全景图像。
2.  **[人物小图]**: 从 [全景图] 中精确裁剪出来的、聚焦于单个目标人物的特写或半身图像。

**目标：**
根据提供的 **[全景图]** 和 **[人物小图]**，生成一个关于该人物的详细描述。这份描述必须严格遵循下面提供的 `TEMPLATE_JSON` 格式。

**核心规则：**
*   **双重图像分析**: 你必须同时利用两张图片。**[人物小图]** 用于识别人物本身的外貌、服饰、姿态等细节；**[全景图]** 用于理解人物所处的环境、可能的行为、与周围的互动关系，以及推断其身份和目的。
*   **严格遵循格式**: 输出**必须**是一个完整的、可被解析的JSON对象，其结构、键名、数据类型（字符串、列表、对象）都必须与 `TEMPLATE_JSON` 完全一致。任何字段如果无法从图像中获取信息，应使用 `null` 或空列表 `[]` 表示，而不是省略该字段。
*   **观察与推断分离**: 严格区分“可观察属性”和“推断属性”两个部分，这是评价你工作质量的核心标准。

---

### **任务分解与指令**

1.  **全局分析 (Analyze Globally)**: 首先审视 **[全景图]**，了解整体环境（如街道、公园、室内）、光线、天气、时间段以及图中其他人和物的分布，为 `contextualAnalysis` 部分的推断打下基础。
2.  **细节观察 (Observe Details)**: 接着聚焦于 **[人物小图]**，仔细辨认人物的每一个可见细节，包括但不限于：
    *   **外貌**: 发型、可能的肤色、体态。
    *   **服饰**: 上下身衣物和鞋子的类型、颜色、图案、材质和特殊设计。
    *   **配饰与物品**: 所有穿戴、携带或手持的物品。
    *   **姿态**: 身体的朝向、站姿/坐姿、手部动作。
    *   这些信息将填充 `observedAttributes` 部分。
3.  **推理与判断 (Infer and Judge)**: 结合从两张图片中获得的信息进行逻辑推理。
    *   基于外貌和服饰，推断性别、年龄段和可能的社会身份 (`demographics`)。
    *   基于面部表情和身体语言，推断情绪和心情 (`emotion_and_expression`)。
    *   基于人物行为和所处环境，推断其具体活动、互动情况、地点类型和可能的目标 (`contextualAnalysis`)。
    *   **使用审慎的词汇**: 在 `inferredAttributes` 部分，多使用“看似”、“可能”、“似乎”、“推断为”等非绝对性词语。
4.  **整合与撰写 (Synthesize and Write)**:
    *   **撰写摘要**: 最后，将所有观察和推断的关键信息融合成流畅的文字，分别填入 `short_description` (一句话标题) 和 `long_description` (生动的叙事段落)。
    *   **结构化输出**: 将所有分析结果填入 `TEMPLATE_JSON` 结构的对应字段中，生成最终的JSON输出。

---

### **关键概念区分：观察 vs. 推断**

*   `observedAttributes` **(可观察属性)**: **只记录事实**。这是对图像内容的客观描述，是任何人都能直接看到的。例如，“身穿一件蓝色T恤”，“右手拿着手机”。**避免任何猜测**。
*   `inferredAttributes` **(推断属性)**: **进行合理的猜测**。这是基于观察到的事实进行的逻辑推理。例如，根据蓝色T恤上的大学Logo和周围的校园环境，推断其身份“可能是学生”。

---

### **输出JSON模板 (TEMPLATE_JSON)**

```json
{
    "summary": {
        "short_description": "string | 一句话新闻标题式摘要，精准概括人物的核心身份、状态和行为。",
        "long_description": "string | 一段生动的叙事性描述，如同电影旁白，融合所有关键信息，为读者构建一幅完整的视觉和情景画面。"
    },
    "observedAttributes": {
        "comment": "此部分只包含可被直接视觉验证的、客观的事实，避免任何主观猜测。",
        "physicalAppearance": {
            "skin_tone_suggestion": "string | e.g., '白皙', '象牙白', '自然色', '小麦色', '橄榄色', '古铜色', '深棕色'",
            "bodyComposition": "string | 对身体肌肉和脂肪分布的客观描述，e.g., '体态纤细', '骨架分明', '肌肉线条明显', '身形圆润', '体格壮硕'",
            "hair": {
                "length": "string | e.g., '光头', '寸头', '短发(耳上)', '齐耳短发', '及肩中发', '长发(过肩)', '及腰长发'",
                "style": "string | e.g., '直发', '自然卷', '大波浪', '羊毛卷', '马尾辫', '丸子头', '编发', '平头', '背头', '油头', '中分', '脏辫'",
                "color": "string | e.g., '纯黑', '深棕', '亚麻金', '奶奶灰', '酒红', '蓝色挑染', '渐变色(如粉到紫)'"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "string | e.g., 'T恤', '衬衫', '卫衣', '夹克', '大衣'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '条纹', '格子', '品牌Logo'",
                "material": "string | e.g., '棉质', '牛仔布', '丝绸', '羊毛', '皮革', '防水尼龙'",
                "details": "array[string] | e.g., ['V领', '带兜帽', '袖子卷起']"
            },
            "lowerBody": {
                "type": "string | e.g., '牛仔裤', '休闲裤', '运动裤', '短裙'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '水洗效果', '格纹'",
                "material": "string | e.g., '牛仔布', '棉布', '西装面料', '灯芯绒'",
                "details": "array[string] | e.g., ['膝盖破洞', '裤脚磨边', '系有皮带']"
            },
            "footwear": {
                "type": "string | e.g., '运动鞋', '帆布鞋', '皮鞋', '靴子'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '拼色', '格子(如Vans棋盘格)', '迷彩', '动物纹理(如豹纹)', '品牌Logo重复图案'",
                "material": "string | e.g., '皮革', '帆布', '绒面', '橡胶'",
                "details": "array[string] | e.g., ['厚底', '高帮设计', '品牌标志醒目']"
            }
        },
        "accessories": [
            {
                "item_name": "string | 穿戴在身上的小型饰品, e.g., '眼镜', '手表', '首饰', '帽子'",
                "type": "string | e.g., '太阳镜', '电子手表', '金项链', '棒球帽'",
                "color": "array[string]",
                "material": "string | e.g., '金属', '皮革', '塑料', '布料', '珍珠', '宝石'",
                "details": "array[string] | e.g., ['戴在头上', '挂在脖子上']"
            }
        ],
        "carriedItems": [
            {
                "item_name": "string | 携带的、非手持的大件物品, e.g., '包', '行李箱'",
                "type": "string | e.g., '双肩包', '手提电脑包', '硬壳拉杆箱'",
                "color": "array[string]",
                "material": "string | e.g., '皮革', '帆布', '尼龙', 'PC/ABS塑料'",
                "details": "array[string] | e.g., ['背在身后', '放置在地面']"
            }
        ],
        "heldObjects": [
            {
                "item_name": "string | 当前正用手持握的物品, e.g., '手机', '水瓶', '雨伞'",
                "type": "string | e.g., '智能手机', '保温杯', '折叠伞'",
                "color": "array[string]",
                "material": "string | e.g., '塑料', '金属', '玻璃', '纸质'",
                "details": "array[string] | e.g., ['右手持握', '屏幕亮起']"
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "string | e.g., '正对', '背对', '左侧面', '左前45度'",
            "posture": "string | e.g., '直立', '坐着', '行走中', '倚靠'",
            "gestures": "array[string] | e.g., ['双手插兜', '双臂交叉于胸前', '正在用手机打字']"
        }
    },
    "inferredAttributes": {
        "comment": "此部分包含基于视觉观察进行的合理推断，应使用审慎的、非绝对的词汇。",
        "demographics": {
            "estimated_gender": "string | '看似男性', '看似女性', '性别特征不明显'",
            "estimated_age_group": "string | '婴幼儿', '儿童', '青少年', '青年', '中年', '老年'",
            "possible_identity": "array[string] | e.g., ['学生', '游客', '上班族', '时尚博主']"
        },
        "emotion_and_expression": {
            "facial_expression_observed": "string | e.g., '面部肌肉放松', '嘴角上扬', '眉毛紧蹙'",
            "inferred_emotion": "string | e.g., '心情愉悦', '专注', '困惑', '情绪中性'",
            "inferred_mood_from_posture": "string | e.g., '姿态放松', '身体紧绷显得紧张', '步伐轻快看似愉悦', '垂头丧气略显沮丧'"
        },
        "contextualAnalysis": {
            "actions": "array[string] | e.g., ['打电话', '喝水', '拍照', '等候']",
            "interactions": {
                "with_people": "string | e.g., '与人并肩交谈', '独自一人'",
                "with_objects": "string | e.g., '倚靠栏杆', '操作笔记本电脑'",
                "with_vehicles": "string | e.g., '正在骑自行车', '准备上公交车'"
            },
            "estimated_location_type": "string | e.g., '室外街道', '公园', '地铁站台', '购物中心'",
            "estimated_purpose_or_goal": "string | e.g., '似乎在等候某人', '可能正在去上班的路上', '像是在进行休闲购物', '看起来在锻炼身体'"
        }
    }
}
```

---

### **高质量输出示例 (学习以下示例的风格、细节程度和推理逻辑)**

**(示例1: 商务女性)**
```json
{
    "summary": {
        "short_description": "一位身穿卡其色风衣的职业女性在现代建筑前，正严肃地进行手机通话。",
        "long_description": "在繁华都市的街角，一位干练的职业女性正驻足通话。她身着经典的卡其色风衣，内搭简约的白衬衫和黑色西裤，脚踩一双黑色高跟鞋，尽显专业气质。她左手持握着手机贴近耳边，表情严肃而专注，右手不时做出细微的手势，似乎在与电话那头的人讨论着重要事宜。身旁的皮质托特包暗示了她的通勤者身份，整个人散发出一种高效而自信的气场。"
    },
    "observedAttributes": {
        "physicalAppearance": {
            "skin_tone_suggestion": "白皙",
            "bodyComposition": "身形高挑匀称",
            "hair": {
                "length": "及肩中发",
                "style": "大波浪",
                "color": "深棕"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "风衣",
                "color": ["卡其色"],
                "pattern": "纯色",
                "material": "棉质/防水面料",
                "details": ["腰带系起", "双排扣设计", "内搭白色衬衫"]
            },
            "lowerBody": {
                "type": "西裤",
                "color": ["黑色"],
                "pattern": "纯色",
                "material": "西装面料",
                "details": ["九分裤剪裁"]
            },
            "footwear": {
                "type": "高跟鞋",
                "color": ["黑色"],
                "pattern": "纯色",
                "material": "皮革",
                "details": ["尖头设计"]
            }
        },
        "accessories": [
            {
                "item_name": "手表",
                "type": "机械表",
                "color": ["银色", "白色"],
                "material": "金属",
                "details": ["戴在左手腕"]
            }
        ],
        "carriedItems": [
            {
                "item_name": "包",
                "type": "手提托特包",
                "color": ["棕色"],
                "material": "皮革",
                "details": ["放置在身体右侧地面"]
            }
        ],
        "heldObjects": [
            {
                "item_name": "手机",
                "type": "智能手机",
                "color": ["黑色"],
                "material": "玻璃/金属",
                "details": ["左手持握贴近耳边"]
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "右前45度",
            "posture": "直立",
            "gestures": ["左手持手机通话", "右手偶尔做出手势"]
        }
    },
    "inferredAttributes": {
        "demographics": {
            "estimated_gender": "看似女性",
            "estimated_age_group": "青年",
            "possible_identity": ["上班族", "商务人士", "管理者"]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "眉毛微蹙",
            "inferred_emotion": "专注",
            "inferred_mood_from_posture": "身体挺直显得干练"
        },
        "contextualAnalysis": {
            "actions": ["打电话"],
            "interactions": {
                "with_people": "通过电话与人互动",
                "with_objects": "使用手机",
                "with_vehicles": "无"
            },
            "estimated_location_type": "室外街道",
            "estimated_purpose_or_goal": "正在进行商务通话"
        }
    }
}
```
**(示例2: 在公园草地上创作的年轻学生)**
```json
{
    "summary": {
        "short_description": "一位戴着针织帽的年轻人在公园草地上盘腿而坐，一边听音乐一边在素描本上画画。",
        "long_description": "在阳光和煦的公园草坪上，一位充满艺术气息的年轻人正沉浸在自己的创作世界里。他戴着一顶灰色的针织帽，脖子上挂着头戴式耳机，身上穿着一件敞开的格子衬衫，内搭印花T恤。他盘腿而坐，面前摊开着一个素描本，右手拿着铅笔在上面不停地描绘。他身旁放着一个磨旧的帆布双肩包，整个画面充满了青春、自由和创造力。"
    },
    "observedAttributes": {
        "physicalAppearance": {
            "skin_tone_suggestion": "自然色",
            "bodyComposition": "体态纤细",
            "hair": {
                "length": "短发(耳上)",
                "style": "大部分被帽子遮挡",
                "color": "黑色带蓝色挑染"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "格子衬衫",
                "color": [
                    "红色",
                    "黑色"
                ],
                "pattern": "格子",
                "material": "棉质",
                "details": [
                    "敞开穿着",
                    "袖子卷起",
                    "内搭白色印花T恤"
                ]
            },
            "lowerBody": {
                "type": "牛仔裤",
                "color": [
                    "黑色"
                ],
                "pattern": "水洗效果",
                "material": "牛仔布",
                "details": [
                    "膝盖处有破洞"
                ]
            },
            "footwear": {
                "type": "靴子(马丁靴)",
                "color": [
                    "黑色"
                ],
                "pattern": "纯色",
                "material": "皮革",
                "details": [
                    "高帮设计",
                    "鞋带部分松开"
                ]
            }
        },
        "accessories": [
            {
                "item_name": "帽子",
                "type": "针织帽",
                "color": [
                    "深灰色"
                ],
                "material": "羊毛",
                "details": [
                    "戴在头上"
                ]
            },
            {
                "item_name": "耳机",
                "type": "头戴式耳机",
                "color": [
                    "黑色",
                    "银色"
                ],
                "material": "塑料/金属",
                "details": [
                    "挂在脖子上"
                ]
            }
        ],
        "carriedItems": [
            {
                "item_name": "包",
                "type": "双肩包",
                "color": [
                    "军绿色"
                ],
                "material": "帆布",
                "details": [
                    "放置在身体左侧草地上"
                ]
            }
        ],
        "heldObjects": [
            {
                "item_name": "笔记本",
                "type": "素描本",
                "color": [
                    "牛皮纸色"
                ],
                "material": "纸质",
                "details": [
                    "摊开放在腿上"
                ]
            },
            {
                "item_name": "笔",
                "type": "铅笔",
                "color": [
                    "原木色"
                ],
                "material": "木质",
                "details": [
                    "右手持握正在绘画"
                ]
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "正对",
            "posture": "坐着",
            "gestures": [
                "低头看笔记本",
                "右手执笔绘画"
            ]
        }
    },
    "inferredAttributes": {
        "demographics": {
            "estimated_gender": "性别特征不明显",
            "estimated_age_group": "青少年(12-17)",
            "possible_identity": [
                "学生",
                "艺术爱好者",
                "街头艺人"
            ]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "面部肌肉放松",
            "inferred_emotion": "专注",
            "inferred_mood_from_posture": "姿态放松，沉浸在自己的世界"
        },
        "contextualAnalysis": {
            "actions": [
                "绘画",
                "听音乐",
                "休息"
            ],
            "interactions": {
                "with_people": "独自一人",
                "with_objects": "使用素描本和笔",
                "with_vehicles": "无"
            },
            "estimated_location_type": "公园绿地",
            "estimated_purpose_or_goal": "似乎在进行素描创作或记录灵感"
        }
    }
}
```
**(示例3: 在湖边步道锻炼的活力长者)**
```json
{
    "summary": {
        "short_description": "一位戴着太阳镜、身穿运动装备的老年男士在湖边步道上快走，显得精力充沛。",
        "long_description": "在风景宜人的湖边步道上，一位精神矍铄的老年男士正在进行日常锻炼。他戴着一副运动太阳镜，身穿一件亮色的速干T恤和运动短裤，脚上的徒步鞋显示出他对户外活动的热爱。他步伐稳健地向前快走，双臂自然摆动，展现出与其年龄不符的活力。他左手手腕上的运动手环和腰间的小包表明他是一位经验丰富的健身者，正在享受健康积极的退休生活。"
    },
    "observedAttributes": {
        "physicalAppearance": {
            "skin_tone_suggestion": "古铜色",
            "bodyComposition": "身形健朗",
            "hair": {
                "length": "短发(耳上)",
                "style": "自然发型",
                "color": "灰色/银色"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "T恤",
                "color": [
                    "橙色"
                ],
                "pattern": "品牌Logo",
                "material": "速干面料",
                "details": [
                    "圆领"
                ]
            },
            "lowerBody": {
                "type": "运动短裤",
                "color": [
                    "深灰色"
                ],
                "pattern": "纯色",
                "material": "尼龙",
                "details": [
                    "侧边有反光条"
                ]
            },
            "footwear": {
                "type": "运动鞋",
                "color": [
                    "蓝色",
                    "白色"
                ],
                "pattern": "拼色",
                "material": "网面/橡胶",
                "details": [
                    "专业的徒步鞋款式"
                ]
            }
        },
        "accessories": [
            {
                "item_name": "眼镜",
                "type": "太阳镜",
                "color": [
                    "黑色"
                ],
                "material": "塑料",
                "details": [
                    "戴在脸上",
                    "运动款式"
                ]
            },
            {
                "item_name": "手表",
                "type": "电子手表",
                "color": [
                    "黑色"
                ],
                "material": "塑料/硅胶",
                "details": [
                    "运动手环款式",
                    "戴在左手腕"
                ]
            }
        ],
        "carriedItems": [
            {
                "item_name": "包",
                "type": "腰包",
                "color": [
                    "黑色"
                ],
                "material": "尼龙",
                "details": [
                    "系在腰间"
                ]
            }
        ],
        "heldObjects": [],
        "postureAndOrientation": {
            "body_orientation": "左侧面",
            "posture": "行走中",
            "gestures": [
                "双臂自然前后摆动"
            ]
        }
    },
    "inferredAttributes": {
        "demographics": {
            "estimated_gender": "看似男性",
            "estimated_age_group": "老年(56+)",
            "possible_identity": [
                "徒步者",
                "退休人士",
                "运动爱好者"
            ]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "面部肌肉放松",
            "inferred_emotion": "心情愉悦",
            "inferred_mood_from_posture": "步伐轻快看似愉悦"
        },
        "contextualAnalysis": {
            "actions": [
                "快走",
                "锻炼"
            ],
            "interactions": {
                "with_people": "独自一人",
                "with_objects": "无",
                "with_vehicles": "无"
            },
            "estimated_location_type": "公园",
            "estimated_purpose_or_goal": "正在进行户外锻炼，享受自然风光"
        }
    }
}
```
**最终指令：**
请根据下面提供的 **[全景图]** 和 **[人物小图]**，严格遵循上述所有规则、指令和JSON模板，生成最终的描述。现在，开始分析。
"""

prompt_single = """**角色：** 你是一个专业的人体与场景分析AI，具备敏锐的观察力和严谨的逻辑推理能力。你的任务是分析图像，并以高度结构化的JSON格式输出对特定人物的详细描述。

**输入：**
1.  **[全景图]**: 一张包含完整场景和多个元素的宽幅或全景图像。
2.  **[人物坐标]**: 一个定义了目标人物在 **[全景图]** 中位置的边界框（Bounding Box）坐标，格式为 `[x_min, y_min, x_max, y_max]`，代表了需要你重点分析的人物区域。

**目标：**
根据提供的 **[全景图]** 和 **[人物坐标]**，生成一个关于该坐标区域内人物的详细描述。这份描述必须严格遵循下面提供的 `TEMPLATE_JSON` 格式。

**核心规则：**
*   **单一图像，双重焦点**: 你将只收到一张 **[全景图]**。你必须利用 **[人物坐标]** 在全景图上定位并“虚拟放大”目标人物，以此来识别其外貌、服饰、姿态等细节；同时，你必须利用完整的 **[全景图]** 来理解人物所处的环境、可能的行为、与周围的互动关系，以及推断其身份和目的。
*   **严格遵循格式**: 输出**必须**是一个完整的、可被解析的JSON对象，其结构、键名、数据类型（字符串、列表、对象）都必须与 `TEMPLATE_JSON` 完全一致。任何字段如果无法从图像中获取信息，应使用 `null` 或空列表 `[]` 表示，而不是省略该字段。
*   **观察与推断分离**: 严格区分“可观察属性”和“推断属性”两个部分，这是评价你工作质量的核心标准。

---

### **任务分解与指令**

1.  **全局分析 (Analyze Globally)**: 首先审视 **[全景图]**，了解整体环境（如街道、公园、室内）、光线、天气、时间段以及图中其他人和物的分布，为 `contextualAnalysis` 部分的推断打下基础。
2.  **细节观察 (Observe Details)**: 接着，根据 **[人物坐标]** 定位到 **[全景图]** 中的目标人物区域。聚焦于此区域并仔细辨认人物的每一个可见细节，包括但不限于：
    *   **外貌**: 发型、可能的肤色、体态。
    *   **服饰**: 上下身衣物和鞋子的类型、颜色、图案、材质和特殊设计。
    *   **配饰与物品**: 所有穿戴、携带或手持的物品。
    *   **姿态**: 身体的朝向、站姿/坐姿、手部动作。
    *   这些信息将填充 `observedAttributes` 部分。
3.  **推理与判断 (Infer and Judge)**: 结合从全景图的全局环境和目标人物的局部细节中获得的信息进行逻辑推理。
    *   基于外貌和服饰，推断性别、年龄段和可能的社会身份 (`demographics`)。
    *   基于面部表情和身体语言，推断情绪和心情 (`emotion_and_expression`)。
    *   基于人物行为和所处环境，推断其具体活动、互动情况、地点类型和可能的目标 (`contextualAnalysis`)。
    *   **使用审慎的词汇**: 在 `inferredAttributes` 部分，多使用“看似”、“可能”、“似乎”、“推断为”等非绝对性词语。
4.  **整合与撰写 (Synthesize and Write)**:
    *   **撰写摘要**: 最后，将所有观察和推断的关键信息融合成流畅的文字，分别填入 `short_description` (一句话标题) 和 `long_description` (生动的叙事段落)。
    *   **结构化输出**: 将所有分析结果填入 `TEMPLATE_JSON` 结构的对应字段中，生成最终的JSON输出。

---

### **关键概念区分：观察 vs. 推断**

*   `observedAttributes` **(可观察属性)**: **只记录事实**。这是对图像内容的客观描述，是任何人都能直接看到的。例如，“身穿一件蓝色T恤”，“右手拿着手机”。**避免任何猜测**。
*   `inferredAttributes` **(推断属性)**: **进行合理的猜测**。这是基于观察到的事实进行的逻辑推理。例如，根据蓝色T恤上的大学Logo和周围的校园环境，推断其身份“可能是学生”。

---

### **输出JSON模板 (TEMPLATE_JSON)**

```json
{{
    "summary": {{
        "short_description": "string | 一句话新闻标题式摘要，精准概括人物的核心身份、状态和行为。",
        "long_description": "string | 一段生动的叙事性描述，如同电影旁白，融合所有关键信息，为读者构建一幅完整的视觉和情景画面。"
    }},
    "observedAttributes": {{
        "comment": "此部分只包含可被直接视觉验证的、客观的事实，避免任何主观猜测。",
        "physicalAppearance": {{
            "skin_tone_suggestion": "string | e.g., '白皙', '象牙白', '自然色', '小麦色', '橄榄色', '古铜色', '深棕色'",
            "bodyComposition": "string | 对身体肌肉和脂肪分布的客观描述，e.g., '体态纤细', '骨架分明', '肌肉线条明显', '身形圆润', '体格壮硕'",
            "hair": {{
                "length": "string | e.g., '光头', '寸头', '短发(耳上)', '齐耳短发', '及肩中发', '长发(过肩)', '及腰长发'",
                "style": "string | e.g., '直发', '自然卷', '大波浪', '羊毛卷', '马尾辫', '丸子头', '编发', '平头', '背头', '油头', '中分', '脏辫'",
                "color": "string | e.g., '纯黑', '深棕', '亚麻金', '奶奶灰', '酒红', '蓝色挑染', '渐变色(如粉到紫)'"
            }}
        }},
        "apparel": {{
            "upperBody": {{
                "type": "string | e.g., 'T恤', '衬衫', '卫衣', '夹克', '大衣'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '条纹', '格子', '品牌Logo'",
                "material": "string | e.g., '棉质', '牛仔布', '丝绸', '羊毛', '皮革', '防水尼龙'",
                "details": "array[string] | e.g., ['V领', '带兜帽', '袖子卷起']"
            }},
            "lowerBody": {{
                "type": "string | e.g., '牛仔裤', '休闲裤', '运动裤', '短裙'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '水洗效果', '格纹'",
                "material": "string | e.g., '牛仔布', '棉布', '西装面料', '灯芯绒'",
                "details": "array[string] | e.g., ['膝盖破洞', '裤脚磨边', '系有皮带']"
            }},
            "footwear": {{
                "type": "string | e.g., '运动鞋', '帆布鞋', '皮鞋', '靴子'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '拼色', '格子(如Vans棋盘格)', '迷彩', '动物纹理(如豹纹)', '品牌Logo重复图案'",
                "material": "string | e.g., '皮革', '帆布', '绒面', '橡胶'",
                "details": "array[string] | e.g., ['厚底', '高帮设计', '品牌标志醒目']"
            }}
        }},
        "accessories": [
            {{
                "item_name": "string | 穿戴在身上的小型饰品, e.g., '眼镜', '手表', '首饰', '帽子'",
                "type": "string | e.g., '太阳镜', '电子手表', '金项链', '棒球帽'",
                "color": "array[string]",
                "material": "string | e.g., '金属', '皮革', '塑料', '布料', '珍珠', '宝石'",
                "details": "array[string] | e.g., ['戴在头上', '挂在脖子上']"
            }}
        ],
        "carriedItems": [
            {{
                "item_name": "string | 携带的、非手持的大件物品, e.g., '包', '行李箱'",
                "type": "string | e.g., '双肩包', '手提电脑包', '硬壳拉杆箱'",
                "color": "array[string]",
                "material": "string | e.g., '皮革', '帆布', '尼龙', 'PC/ABS塑料'",
                "details": "array[string] | e.g., ['背在身后', '放置在地面']"
            }}
        ],
        "heldObjects": [
            {{
                "item_name": "string | 当前正用手持握的物品, e.g., '手机', '水瓶', '雨伞'",
                "type": "string | e.g., '智能手机', '保温杯', '折叠伞'",
                "color": "array[string]",
                "material": "string | e.g., '塑料', '金属', '玻璃', '纸质'",
                "details": "array[string] | e.g., ['右手持握', '屏幕亮起']"
            }}
        ],
        "postureAndOrientation": {{
            "body_orientation": "string | e.g., '正对', '背对', '左侧面', '左前45度'",
            "posture": "string | e.g., '直立', '坐着', '行走中', '倚靠'",
            "gestures": "array[string] | e.g., ['双手插兜', '双臂交叉于胸前', '正在用手机打字']"
        }}
    }},
    "inferredAttributes": {{
        "comment": "此部分包含基于视觉观察进行的合理推断，应使用审慎的、非绝对的词汇。",
        "demographics": {{
            "estimated_gender": "string | '看似男性', '看似女性', '性别特征不明显'",
            "estimated_age_group": "string | '婴幼儿', '儿童', '青少年', '青年', '中年', '老年'",
            "possible_identity": "array[string] | e.g., ['学生', '游客', '上班族', '时尚博主']"
        }},
        "emotion_and_expression": {{
            "facial_expression_observed": "string | e.g., '面部肌肉放松', '嘴角上扬', '眉毛紧蹙'",
            "inferred_emotion": "string | e.g., '心情愉悦', '专注', '困惑', '情绪中性'",
            "inferred_mood_from_posture": "string | e.g., '姿态放松', '身体紧绷显得紧张', '步伐轻快看似愉悦', '垂头丧气略显沮丧'"
        }},
        "contextualAnalysis": {{
            "actions": "array[string] | e.g., ['打电话', '喝水', '拍照', '等候']",
            "interactions": {{
                "with_people": "string | e.g., '与人并肩交谈', '独自一人'",
                "with_objects": "string | e.g., '倚靠栏杆', '操作笔记本电脑'",
                "with_vehicles": "string | e.g., '正在骑自行车', '准备上公交车'"
            }},
            "estimated_location_type": "string | e.g., '室外街道', '公园', '地铁站台', '购物中心'",
            "estimated_purpose_or_goal": "string | e.g., '似乎在等候某人', '可能正在去上班的路上', '像是在进行休闲购物', '看起来在锻炼身体'"
        }}
    }}
}}
```

---

### **高质量输出示例 (学习以下示例的风格、细节程度和推理逻辑)**

**(示例1: 商务女性)**
```json
{{
    "summary": {{
        "short_description": "一位身穿卡其色风衣的职业女性在现代建筑前，正严肃地进行手机通话。",
        "long_description": "在繁华都市的街角，一位干练的职业女性正驻足通话。她身着经典的卡其色风衣，内搭简约的白衬衫和黑色西裤，脚踩一双黑色高跟鞋，尽显专业气质。她左手持握着手机贴近耳边，表情严肃而专注，右手不时做出细微的手势，似乎在与电话那头的人讨论着重要事宜。身旁的皮质托特包暗示了她的通勤者身份，整个人散发出一种高效而自信的气场。"
    }},
    "observedAttributes": {{
        "physicalAppearance": {{
            "skin_tone_suggestion": "白皙",
            "bodyComposition": "身形高挑匀称",
            "hair": {{
                "length": "及肩中发",
                "style": "大波浪",
                "color": "深棕"
            }}
        }},
        "apparel": {{
            "upperBody": {{
                "type": "风衣",
                "color": ["卡其色"],
                "pattern": "纯色",
                "material": "棉质/防水面料",
                "details": ["腰带系起", "双排扣设计", "内搭白色衬衫"]
            }},
            "lowerBody": {{
                "type": "西裤",
                "color": ["黑色"],
                "pattern": "纯色",
                "material": "西装面料",
                "details": ["九分裤剪裁"]
            }},
            "footwear": {{
                "type": "高跟鞋",
                "color": ["黑色"],
                "pattern": "纯色",
                "material": "皮革",
                "details": ["尖头设计"]
            }}
        }},
        "accessories": [
            {{
                "item_name": "手表",
                "type": "机械表",
                "color": ["银色", "白色"],
                "material": "金属",
                "details": ["戴在左手腕"]
            }}
        ],
        "carriedItems": [
            {{
                "item_name": "包",
                "type": "手提托特包",
                "color": ["棕色"],
                "material": "皮革",
                "details": ["放置在身体右侧地面"]
            }}
        ],
        "heldObjects": [
            {{
                "item_name": "手机",
                "type": "智能手机",
                "color": ["黑色"],
                "material": "玻璃/金属",
                "details": ["左手持握贴近耳边"]
            }}
        ],
        "postureAndOrientation": {{
            "body_orientation": "右前45度",
            "posture": "直立",
            "gestures": ["左手持手机通话", "右手偶尔做出手势"]
        }}
    }},
    "inferredAttributes": {{
        "demographics": {{
            "estimated_gender": "看似女性",
            "estimated_age_group": "青年",
            "possible_identity": ["上班族", "商务人士", "管理者"]
        }},
        "emotion_and_expression": {{
            "facial_expression_observed": "眉毛微蹙",
            "inferred_emotion": "专注",
            "inferred_mood_from_posture": "身体挺直显得干练"
        }},
        "contextualAnalysis": {{
            "actions": ["打电话"],
            "interactions": {{
                "with_people": "通过电话与人互动",
                "with_objects": "使用手机",
                "with_vehicles": "无"
            }},
            "estimated_location_type": "室外街道",
            "estimated_purpose_or_goal": "正在进行商务通话"
        }}
    }}
}}
```
**(示例2: 在公园草地上创作的年轻学生)**
```json
{{
    "summary": {{
        "short_description": "一位戴着针织帽的年轻人在公园草地上盘腿而坐，一边听音乐一边在素描本上画画。",
        "long_description": "在阳光和煦的公园草坪上，一位充满艺术气息的年轻人正沉浸在自己的创作世界里。他戴着一顶灰色的针织帽，脖子上挂着头戴式耳机，身上穿着一件敞开的格子衬衫，内搭印花T恤。他盘腿而坐，面前摊开着一个素描本，右手拿着铅笔在上面不停地描绘。他身旁放着一个磨旧的帆布双肩包，整个画面充满了青春、自由和创造力。"
    }},
    "observedAttributes": {{
        "physicalAppearance": {{
            "skin_tone_suggestion": "自然色",
            "bodyComposition": "体态纤细",
            "hair": {{
                "length": "短发(耳上)",
                "style": "大部分被帽子遮挡",
                "color": "黑色带蓝色挑染"
            }}
        }},
        "apparel": {{
            "upperBody": {{
                "type": "格子衬衫",
                "color": [
                    "红色",
                    "黑色"
                ],
                "pattern": "格子",
                "material": "棉质",
                "details": [
                    "敞开穿着",
                    "袖子卷起",
                    "内搭白色印花T恤"
                ]
            }},
            "lowerBody": {{
                "type": "牛仔裤",
                "color": [
                    "黑色"
                ],
                "pattern": "水洗效果",
                "material": "牛仔布",
                "details": [
                    "膝盖处有破洞"
                ]
            }},
            "footwear": {{
                "type": "靴子(马丁靴)",
                "color": [
                    "黑色"
                ],
                "pattern": "纯色",
                "material": "皮革",
                "details": [
                    "高帮设计",
                    "鞋带部分松开"
                ]
            }}
        }},
        "accessories": [
            {{
                "item_name": "帽子",
                "type": "针织帽",
                "color": [
                    "深灰色"
                ],
                "material": "羊毛",
                "details": [
                    "戴在头上"
                ]
            }},
            {{
                "item_name": "耳机",
                "type": "头戴式耳机",
                "color": [
                    "黑色",
                    "银色"
                ],
                "material": "塑料/金属",
                "details": [
                    "挂在脖子上"
                ]
            }}
        ],
        "carriedItems": [
            {{
                "item_name": "包",
                "type": "双肩包",
                "color": [
                    "军绿色"
                ],
                "material": "帆布",
                "details": [
                    "放置在身体左侧草地上"
                ]
            }}
        ],
        "heldObjects": [
            {{
                "item_name": "笔记本",
                "type": "素描本",
                "color": [
                    "牛皮纸色"
                ],
                "material": "纸质",
                "details": [
                    "摊开放在腿上"
                ]
            }},
            {{
                "item_name": "笔",
                "type": "铅笔",
                "color": [
                    "原木色"
                ],
                "material": "木质",
                "details": [
                    "右手持握正在绘画"
                ]
            }}
        ],
        "postureAndOrientation": {{
            "body_orientation": "正对",
            "posture": "坐着",
            "gestures": [
                "低头看笔记本",
                "右手执笔绘画"
            ]
        }}
    }},
    "inferredAttributes": {{
        "demographics": {{
            "estimated_gender": "性别特征不明显",
            "estimated_age_group": "青少年(12-17)",
            "possible_identity": [
                "学生",
                "艺术爱好者",
                "街头艺人"
            ]
        }},
        "emotion_and_expression": {{
            "facial_expression_observed": "面部肌肉放松",
            "inferred_emotion": "专注",
            "inferred_mood_from_posture": "姿态放松，沉浸在自己的世界"
        }},
        "contextualAnalysis": {{
            "actions": [
                "绘画",
                "听音乐",
                "休息"
            ],
            "interactions": {{
                "with_people": "独自一人",
                "with_objects": "使用素描本和笔",
                "with_vehicles": "无"
            }},
            "estimated_location_type": "公园绿地",
            "estimated_purpose_or_goal": "似乎在进行素描创作或记录灵感"
        }}
    }}
}}
```
**(示例3: 在湖边步道锻炼的活力长者)**
```json
{{
    "summary": {{
        "short_description": "一位戴着太阳镜、身穿运动装备的老年男士在湖边步道上快走，显得精力充沛。",
        "long_description": "在风景宜人的湖边步道上，一位精神矍铄的老年男士正在进行日常锻炼。他戴着一副运动太阳镜，身穿一件亮色的速干T恤和运动短裤，脚上的徒步鞋显示出他对户外活动的热爱。他步伐稳健地向前快走，双臂自然摆动，展现出与其年龄不符的活力。他左手手腕上的运动手环和腰间的小包表明他是一位经验丰富的健身者，正在享受健康积极的退休生活。"
    }},
    "observedAttributes": {{
        "physicalAppearance": {{
            "skin_tone_suggestion": "古铜色",
            "bodyComposition": "身形健朗",
            "hair": {{
                "length": "短发(耳上)",
                "style": "自然发型",
                "color": "灰色/银色"
            }}
        }},
        "apparel": {{
            "upperBody": {{
                "type": "T恤",
                "color": [
                    "橙色"
                ],
                "pattern": "品牌Logo",
                "material": "速干面料",
                "details": [
                    "圆领"
                ]
            }},
            "lowerBody": {{
                "type": "运动短裤",
                "color": [
                    "深灰色"
                ],
                "pattern": "纯色",
                "material": "尼龙",
                "details": [
                    "侧边有反光条"
                ]
            }},
            "footwear": {{
                "type": "运动鞋",
                "color": [
                    "蓝色",
                    "白色"
                ],
                "pattern": "拼色",
                "material": "网面/橡胶",
                "details": [
                    "专业的徒步鞋款式"
                ]
            }}
        }},
        "accessories": [
            {{
                "item_name": "眼镜",
                "type": "太阳镜",
                "color": [
                    "黑色"
                ],
                "material": "塑料",
                "details": [
                    "戴在脸上",
                    "运动款式"
                ]
            }},
            {{
                "item_name": "手表",
                "type": "电子手表",
                "color": [
                    "黑色"
                ],
                "material": "塑料/硅胶",
                "details": [
                    "运动手环款式",
                    "戴在左手腕"
                ]
            }}
        ],
        "carriedItems": [
            {{
                "item_name": "包",
                "type": "腰包",
                "color": [
                    "黑色"
                ],
                "material": "尼龙",
                "details": [
                    "系在腰间"
                ]
            }}
        ],
        "heldObjects": [],
        "postureAndOrientation": {{
            "body_orientation": "左侧面",
            "posture": "行走中",
            "gestures": [
                "双臂自然前后摆动"
            ]
        }}
    }},
    "inferredAttributes": {{
        "demographics": {{
            "estimated_gender": "看似男性",
            "estimated_age_group": "老年(56+)",
            "possible_identity": [
                "徒步者",
                "退休人士",
                "运动爱好者"
            ]
        }},
        "emotion_and_expression": {{
            "facial_expression_observed": "面部肌肉放松",
            "inferred_emotion": "心情愉悦",
            "inferred_mood_from_posture": "步伐轻快看似愉悦"
        }},
        "contextualAnalysis": {{
            "actions": [
                "快走",
                "锻炼"
            ],
            "interactions": {{
                "with_people": "独自一人",
                "with_objects": "无",
                "with_vehicles": "无"
            }},
            "estimated_location_type": "公园",
            "estimated_purpose_or_goal": "正在进行户外锻炼，享受自然风光"
        }}
    }}
}}
```
**最终指令：**
请根据下面提供的 **[全景图]** 和 **[人物坐标]**，严格遵循上述所有规则、指令和JSON模板，生成最终的描述。现在，开始分析。
1. 输入图片 (Image):

2. 人物坐标:
 {target_person}
"""

prompt_double_things_1 = """**角色：** 你是一个专业的人体与场景分析AI，具备敏锐的观察力和严谨的逻辑推理能力。你的任务是分析图像，并以高度结构化的JSON格式输出对特定人物的详细描述。

**输入：**
1.  **[全景图]**: 一张包含完整场景和多个元素的宽幅或全景图像。
2.  **[人物小图]**: 从 [全景图] 中精确裁剪出来的、聚焦于单个目标人物的特写或半身图像。

**目标：**
根据提供的 **[全景图]** 和 **[人物小图]**，生成一个关于该人物的详细描述。这份描述必须严格遵循下面提供的 `TEMPLATE_JSON` 格式。

**核心规则：**
*   **双重图像分析**: 你必须同时利用两张图片。**[人物小图]** 用于识别人物本身的外貌、服饰、姿态等细节；**[全景图]** 用于理解人物所处的环境、可能的行为、与周围的互动关系，以及推断其身份和目的。
*   **严格遵循格式**: 输出**必须**是一个完整的、可被解析的JSON对象，其结构、键名、数据类型（字符串、列表、对象）都必须与 `TEMPLATE_JSON` 完全一致。任何字段如果无法从图像中获取信息，应使用 `null` 或空列表 `[]` 表示，而不是省略该字段。
*   **观察与推断分离**: 严格区分“可观察属性”和“推断属性”两个部分，这是评价你工作质量的核心标准。

---

### **任务分解与指令**

1.  **全局分析 (Analyze Globally)**: 首先审视 **[全景图]**，了解整体环境（如街道、公园、室内）、光线、天气、时间段以及图中其他人和物的分布，为 `contextualAnalysis` 部分的推断打下基础。
2.  **细节观察 (Observe Details)**: 接着聚焦于 **[人物小图]**，仔细辨认人物的每一个可见细节，包括但不限于：
    *   **外貌**: 发型、可能的肤色、体态。
    *   **服饰**: 上下身衣物和鞋子的类型、颜色、图案、材质和特殊设计。
    *   **配饰与物品**: 所有穿戴、携带或手持的物品。
        *   **特别注意**: 对于 `accessories`, `carriedItems`, `heldObjects` 中的每个物品，根据其 `item_name`, `type`, `color`, `material` 等属性，生成简短的中文描述短语填入 `describe_zh`，并将其准确翻译为英文填入 `describe_en`。例如，一个物品类型为'腰包'，颜色为['黑色']，则 `describe_zh` 可以是 '黑色腰包'，`describe_en` 是 'black waist bag'；若材质为'尼龙'，则更佳描述为 `describe_zh`: '黑色尼龙腰包', `describe_en`: 'black nylon waist bag'。
    *   **姿态**: 身体的朝向、站姿/坐姿、手部动作。
    *   这些信息将填充 `observedAttributes` 部分。
3.  **推理与判断 (Infer and Judge)**: 结合从两张图片中获得的信息进行逻辑推理。
    *   基于外貌和服饰，推断性别、年龄段和可能的社会身份 (`demographics`)。
    *   基于面部表情和身体语言，推断情绪和心情 (`emotion_and_expression`)。
    *   基于人物行为和所处环境，推断其具体活动、互动情况、地点类型和可能的目标 (`contextualAnalysis`)。
    *   **使用审慎的词汇**: 在 `inferredAttributes` 部分，多使用“看似”、“可能”、“似乎”、“推断为”等非绝对性词语。
4.  **整合与撰写 (Synthesize and Write)**:
    *   **撰写摘要**: 最后，将所有观察和推断的关键信息融合成流畅的文字，分别填入 `short_description` (一句话标题) 和 `long_description` (生动的叙事段落)。
    *   **结构化输出**: 将所有分析结果填入 `TEMPLATE_JSON` 结构的对应字段中，生成最终的JSON输出。

---

### **关键概念区分：观察 vs. 推断**

*   `observedAttributes` **(可观察属性)**: **只记录事实**。这是对图像内容的客观描述，是任何人都能直接看到的。例如，“身穿一件蓝色T恤”，“右手拿着手机”。**避免任何猜测**。
*   `inferredAttributes` **(推断属性)**: **进行合理的猜测**。这是基于观察到的事实进行的逻辑推理。例如，根据蓝色T恤上的大学Logo和周围的校园环境，推断其身份“可能是学生”。

---

### **输出JSON模板 (TEMPLATE_JSON)**

```json
{
    "summary": {
        "short_description": "string | 一句话新闻标题式摘要，精准概括人物的核心身份、状态和行为。",
        "long_description": "string | 一段生动的叙事性描述，如同电影旁白，融合所有关键信息，为读者构建一幅完整的视觉和情景画面。"
    },
    "observedAttributes": {
        "comment": "此部分只包含可被直接视觉验证的、客观的事实，避免任何主观猜测。",
        "physicalAppearance": {
            "skin_tone_suggestion": "string | e.g., '白皙', '象牙白', '自然色', '小麦色', '橄榄色', '古铜色', '深棕色'",
            "bodyComposition": "string | 对身体肌肉和脂肪分布的客观描述，e.g., '体态纤细', '骨架分明', '肌肉线条明显', '身形圆润', '体格壮硕'",
            "hair": {
                "length": "string | e.g., '光头', '寸头', '短发(耳上)', '齐耳短发', '及肩中发', '长发(过肩)', '及腰长发'",
                "style": "string | e.g., '直发', '自然卷', '大波浪', '羊毛卷', '马尾辫', '丸子头', '编发', '平头', '背头', '油头', '中分', '脏辫'",
                "color": "string | e.g., '纯黑', '深棕', '亚麻金', '奶奶灰', '酒红', '蓝色挑染', '渐变色(如粉到紫)'"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "string | e.g., 'T恤', '衬衫', '卫衣', '夹克', '大衣'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '条纹', '格子', '品牌Logo'",
                "material": "string | e.g., '棉质', '牛仔布', '丝绸', '羊毛', '皮革', '防水尼龙'",
                "details": "array[string] | e.g., ['V领', '带兜帽', '袖子卷起']"
            },
            "lowerBody": {
                "type": "string | e.g., '牛仔裤', '休闲裤', '运动裤', '短裙'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '水洗效果', '格纹'",
                "material": "string | e.g., '牛仔布', '棉布', '西装面料', '灯芯绒'",
                "details": "array[string] | e.g., ['膝盖破洞', '裤脚磨边', '系有皮带']"
            },
            "footwear": {
                "type": "string | e.g., '运动鞋', '帆布鞋', '皮鞋', '靴子'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '拼色', '格子(如Vans棋盘格)', '迷彩', '动物纹理(如豹纹)', '品牌Logo重复图案'",
                "material": "string | e.g., '皮革', '帆布', '绒面', '橡胶'",
                "details": "array[string] | e.g., ['厚底', '高帮设计', '品牌标志醒目']"
            }
        },
        "accessories": [
            {
                "item_name": "string | 穿戴在身上的小型饰品, e.g., '眼镜', '手表', '首饰', '帽子'",
                "type": "string | e.g., '太阳镜', '电子手表', '金项链', '棒球帽'",
                "color": "array[string]",
                "material": "string | e.g., '金属', '皮革', '塑料', '布料', '珍珠', '宝石'",
                "details": "array[string] | e.g., ['戴在头上', '挂在脖子上']",
                "describe_zh": "string | 对该饰品的简短中文描述，综合颜色、类型等，如 '黑色棒球帽', '银色金属手表'",
                "describe_en": "string | describe_zh 对应的英文翻译, e.g., 'black baseball cap', 'silver metal watch'"
            }
        ],
        "carriedItems": [
            {
                "item_name": "string | 携带的、非手持的大件物品, e.g., '包', '行李箱'",
                "type": "string | e.g., '双肩包', '手提电脑包', '硬壳拉杆箱'",
                "color": "array[string]",
                "material": "string | e.g., '皮革', '帆布', '尼龙', 'PC/ABS塑料'",
                "details": "array[string] | e.g., ['背在身后', '放置在地面']",
                "describe_zh": "string | 对该携带物品的简短中文描述，综合颜色、类型等，如 '黑色尼龙腰包', '军绿色帆布双肩包'",
                "describe_en": "string | describe_zh 对应的英文翻译, e.g., 'black nylon waist bag', 'army green canvas backpack'"
            }
        ],
        "heldObjects": [
            {
                "item_name": "string | 当前正用手持握的物品, e.g., '手机', '水瓶', '雨伞'",
                "type": "string | e.g., '智能手机', '保温杯', '折叠伞'",
                "color": "array[string]",
                "material": "string | e.g., '塑料', '金属', '玻璃', '纸质'",
                "details": "array[string] | e.g., ['右手持握', '屏幕亮起']",
                "describe_zh": "string | 对该手持物品的简短中文描述，综合颜色、类型等，如 '黑色智能手机', '牛皮纸色素描本'",
                "describe_en": "string | describe_zh 对应的英文翻译, e.g., 'black smartphone', 'kraft paper color sketchbook'"
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "string | e.g., '正对', '背对', '左侧面', '左前45度'",
            "posture": "string | e.g., '直立', '坐着', '行走中', '倚靠'",
            "gestures": "array[string] | e.g., ['双手插兜', '双臂交叉于胸前', '正在用手机打字']"
        }
    },
    "inferredAttributes": {
        "comment": "此部分包含基于视觉观察进行的合理推断，应使用审慎的、非绝对的词汇。",
        "demographics": {
            "estimated_gender": "string | '看似男性', '看似女性', '性别特征不明显'",
            "estimated_age_group": "string | '婴幼儿', '儿童', '青少年', '青年', '中年', '老年'",
            "possible_identity": "array[string] | e.g., ['学生', '游客', '上班族', '时尚博主']"
        },
        "emotion_and_expression": {
            "facial_expression_observed": "string | e.g., '面部肌肉放松', '嘴角上扬', '眉毛紧蹙'",
            "inferred_emotion": "string | e.g., '心情愉悦', '专注', '困惑', '情绪中性'",
            "inferred_mood_from_posture": "string | e.g., '姿态放松', '身体紧绷显得紧张', '步伐轻快看似愉悦', '垂头丧气略显沮丧'"
        },
        "contextualAnalysis": {
            "actions": "array[string] | e.g., ['打电话', '喝水', '拍照', '等候']",
            "interactions": {
                "with_people": "string | e.g., '与人并肩交谈', '独自一人'",
                "with_objects": "string | e.g., '倚靠栏杆', '操作笔记本电脑'",
                "with_vehicles": "string | e.g., '正在骑自行车', '准备上公交车'"
            },
            "estimated_location_type": "string | e.g., '室外街道', '公园', '地铁站台', '购物中心'",
            "estimated_purpose_or_goal": "string | e.g., '似乎在等候某人', '可能正在去上班的路上', '像是在进行休闲购物', '看起来在锻炼身体'"
        }
    }
}
```

---

### **高质量输出示例 (学习以下示例的风格、细节程度和推理逻辑)**

**(示例1: 商务女性)**
```json
{
    "summary": {
        "short_description": "一位身穿卡其色风衣的职业女性在现代建筑前，正严肃地进行手机通话。",
        "long_description": "在繁华都市的街角，一位干练的职业女性正驻足通话。她身着经典的卡其色风衣，内搭简约的白衬衫和黑色西裤，脚踩一双黑色高跟鞋，尽显专业气质。她左手持握着手机贴近耳边，表情严肃而专注，右手不时做出细微的手势，似乎在与电话那头的人讨论着重要事宜。身旁的皮质托特包暗示了她的通勤者身份，整个人散发出一种高效而自信的气场。"
    },
    "observedAttributes": {
        "physicalAppearance": {
            "skin_tone_suggestion": "白皙",
            "bodyComposition": "身形高挑匀称",
            "hair": {
                "length": "及肩中发",
                "style": "大波浪",
                "color": "深棕"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "风衣",
                "color": ["卡其色"],
                "pattern": "纯色",
                "material": "棉质/防水面料",
                "details": ["腰带系起", "双排扣设计", "内搭白色衬衫"]
            },
            "lowerBody": {
                "type": "西裤",
                "color": ["黑色"],
                "pattern": "纯色",
                "material": "西装面料",
                "details": ["九分裤剪裁"]
            },
            "footwear": {
                "type": "高跟鞋",
                "color": ["黑色"],
                "pattern": "纯色",
                "material": "皮革",
                "details": ["尖头设计"]
            }
        },
        "accessories": [
            {
                "item_name": "手表",
                "type": "机械表",
                "color": ["银色", "白色"],
                "material": "金属",
                "details": ["戴在左手腕"],
                "describe_zh": "银白色金属机械表",
                "describe_en": "silver and white metal mechanical watch"
            }
        ],
        "carriedItems": [
            {
                "item_name": "包",
                "type": "手提托特包",
                "color": ["棕色"],
                "material": "皮革",
                "details": ["放置在身体右侧地面"],
                "describe_zh": "棕色皮革手提托特包",
                "describe_en": "brown leather tote bag"
            }
        ],
        "heldObjects": [
            {
                "item_name": "手机",
                "type": "智能手机",
                "color": ["黑色"],
                "material": "玻璃/金属",
                "details": ["左手持握贴近耳边"],
                "describe_zh": "黑色智能手机",
                "describe_en": "black smartphone"
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "右前45度",
            "posture": "直立",
            "gestures": ["左手持手机通话", "右手偶尔做出手势"]
        }
    },
    "inferredAttributes": {
        "demographics": {
            "estimated_gender": "看似女性",
            "estimated_age_group": "青年",
            "possible_identity": ["上班族", "商务人士", "管理者"]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "眉毛微蹙",
            "inferred_emotion": "专注",
            "inferred_mood_from_posture": "身体挺直显得干练"
        },
        "contextualAnalysis": {
            "actions": ["打电话"],
            "interactions": {
                "with_people": "通过电话与人互动",
                "with_objects": "使用手机",
                "with_vehicles": "无"
            },
            "estimated_location_type": "室外街道",
            "estimated_purpose_or_goal": "正在进行商务通话"
        }
    }
}
```
**(示例2: 在公园草地上创作的年轻学生)**
```json
{
    "summary": {
        "short_description": "一位戴着针织帽的年轻人在公园草地上盘腿而坐，一边听音乐一边在素描本上画画。",
        "long_description": "在阳光和煦的公园草坪上，一位充满艺术气息的年轻人正沉浸在自己的创作世界里。他戴着一顶灰色的针织帽，脖子上挂着头戴式耳机，身上穿着一件敞开的格子衬衫，内搭印花T恤。他盘腿而坐，面前摊开着一个素描本，右手拿着铅笔在上面不停地描绘。他身旁放着一个磨旧的帆布双肩包，整个画面充满了青春、自由和创造力。"
    },
    "observedAttributes": {
        "physicalAppearance": {
            "skin_tone_suggestion": "自然色",
            "bodyComposition": "体态纤细",
            "hair": {
                "length": "短发(耳上)",
                "style": "大部分被帽子遮挡",
                "color": "黑色带蓝色挑染"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "格子衬衫",
                "color": [
                    "红色",
                    "黑色"
                ],
                "pattern": "格子",
                "material": "棉质",
                "details": [
                    "敞开穿着",
                    "袖子卷起",
                    "内搭白色印花T恤"
                ]
            },
            "lowerBody": {
                "type": "牛仔裤",
                "color": [
                    "黑色"
                ],
                "pattern": "水洗效果",
                "material": "牛仔布",
                "details": [
                    "膝盖处有破洞"
                ]
            },
            "footwear": {
                "type": "靴子(马丁靴)",
                "color": [
                    "黑色"
                ],
                "pattern": "纯色",
                "material": "皮革",
                "details": [
                    "高帮设计",
                    "鞋带部分松开"
                ]
            }
        },
        "accessories": [
            {
                "item_name": "帽子",
                "type": "针织帽",
                "color": [
                    "深灰色"
                ],
                "material": "羊毛",
                "details": [
                    "戴在头上"
                ],
                "describe_zh": "深灰色羊毛针织帽",
                "describe_en": "dark gray wool knit cap"
            },
            {
                "item_name": "耳机",
                "type": "头戴式耳机",
                "color": [
                    "黑色",
                    "银色"
                ],
                "material": "塑料/金属",
                "details": [
                    "挂在脖子上"
                ],
                "describe_zh": "黑色银色头戴式耳机",
                "describe_en": "black and silver over-ear headphones"
            }
        ],
        "carriedItems": [
            {
                "item_name": "包",
                "type": "双肩包",
                "color": [
                    "军绿色"
                ],
                "material": "帆布",
                "details": [
                    "放置在身体左侧草地上"
                ],
                "describe_zh": "军绿色帆布双肩包",
                "describe_en": "army green canvas backpack"
            }
        ],
        "heldObjects": [
            {
                "item_name": "笔记本",
                "type": "素描本",
                "color": [
                    "牛皮纸色"
                ],
                "material": "纸质",
                "details": [
                    "摊开放在腿上"
                ],
                "describe_zh": "牛皮纸色素描本",
                "describe_en": "kraft paper color sketchbook"
            },
            {
                "item_name": "笔",
                "type": "铅笔",
                "color": [
                    "原木色"
                ],
                "material": "木质",
                "details": [
                    "右手持握正在绘画"
                ],
                "describe_zh": "原木色铅笔",
                "describe_en": "natural wood color pencil"
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "正对",
            "posture": "坐着",
            "gestures": [
                "低头看笔记本",
                "右手执笔绘画"
            ]
        }
    },
    "inferredAttributes": {
        "demographics": {
            "estimated_gender": "性别特征不明显",
            "estimated_age_group": "青少年(12-17)",
            "possible_identity": [
                "学生",
                "艺术爱好者",
                "街头艺人"
            ]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "面部肌肉放松",
            "inferred_emotion": "专注",
            "inferred_mood_from_posture": "姿态放松，沉浸在自己的世界"
        },
        "contextualAnalysis": {
            "actions": [
                "绘画",
                "听音乐",
                "休息"
            ],
            "interactions": {
                "with_people": "独自一人",
                "with_objects": "使用素描本和笔",
                "with_vehicles": "无"
            },
            "estimated_location_type": "公园绿地",
            "estimated_purpose_or_goal": "似乎在进行素描创作或记录灵感"
        }
    }
}
```
**(示例3: 在湖边步道锻炼的活力长者)**
```json
{
    "summary": {
        "short_description": "一位戴着太阳镜、身穿运动装备的老年男士在湖边步道上快走，显得精力充沛。",
        "long_description": "在风景宜人的湖边步道上，一位精神矍铄的老年男士正在进行日常锻炼。他戴着一副运动太阳镜，身穿一件亮色的速干T恤和运动短裤，脚上的徒步鞋显示出他对户外活动的热爱。他步伐稳健地向前快走，双臂自然摆动，展现出与其年龄不符的活力。他左手手腕上的运动手环和腰间的小包表明他是一位经验丰富的健身者，正在享受健康积极的退休生活。"
    },
    "observedAttributes": {
        "physicalAppearance": {
            "skin_tone_suggestion": "古铜色",
            "bodyComposition": "身形健朗",
            "hair": {
                "length": "短发(耳上)",
                "style": "自然发型",
                "color": "灰色/银色"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "T恤",
                "color": [
                    "橙色"
                ],
                "pattern": "品牌Logo",
                "material": "速干面料",
                "details": [
                    "圆领"
                ]
            },
            "lowerBody": {
                "type": "运动短裤",
                "color": [
                    "深灰色"
                ],
                "pattern": "纯色",
                "material": "尼龙",
                "details": [
                    "侧边有反光条"
                ]
            },
            "footwear": {
                "type": "运动鞋",
                "color": [
                    "蓝色",
                    "白色"
                ],
                "pattern": "拼色",
                "material": "网面/橡胶",
                "details": [
                    "专业的徒步鞋款式"
                ]
            }
        },
        "accessories": [
            {
                "item_name": "眼镜",
                "type": "太阳镜",
                "color": [
                    "黑色"
                ],
                "material": "塑料",
                "details": [
                    "戴在脸上",
                    "运动款式"
                ],
                "describe_zh": "黑色塑料太阳镜",
                "describe_en": "black plastic sunglasses"
            },
            {
                "item_name": "手表",
                "type": "电子手表",
                "color": [
                    "黑色"
                ],
                "material": "塑料/硅胶",
                "details": [
                    "运动手环款式",
                    "戴在左手腕"
                ],
                "describe_zh": "黑色运动手环式电子手表",
                "describe_en": "black sports band style electronic watch"
            }
        ],
        "carriedItems": [
            {
                "item_name": "包",
                "type": "腰包",
                "color": [
                    "黑色"
                ],
                "material": "尼龙",
                "details": [
                    "系在腰间"
                ],
                "describe_zh":"黑色尼龙腰包",
                "describe_en":"black nylon waist bag"
            }
        ],
        "heldObjects": [],
        "postureAndOrientation": {
            "body_orientation": "左侧面",
            "posture": "行走中",
            "gestures": [
                "双臂自然前后摆动"
            ]
        }
    },
    "inferredAttributes": {
        "demographics": {
            "estimated_gender": "看似男性",
            "estimated_age_group": "老年(56+)",
            "possible_identity": [
                "徒步者",
                "退休人士",
                "运动爱好者"
            ]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "面部肌肉放松",
            "inferred_emotion": "心情愉悦",
            "inferred_mood_from_posture": "步伐轻快看似愉悦"
        },
        "contextualAnalysis": {
            "actions": [
                "快走",
                "锻炼"
            ],
            "interactions": {
                "with_people": "独自一人",
                "with_objects": "无",
                "with_vehicles": "无"
            },
            "estimated_location_type": "公园",
            "estimated_purpose_or_goal": "正在进行户外锻炼，享受自然风光"
        }
    }
}
```
**最终指令：**
请根据下面提供的 **[全景图]** 和 **[人物小图]**，严格遵循上述所有规则、指令和JSON模板，生成最终的描述。现在，开始分析。
"""

prompt_double_things_type = """**角色：** 你是一个专业的人体与场景分析AI，具备敏锐的观察力和严谨的逻辑推理能力。你的任务是分析图像，并以高度结构化的JSON格式输出对特定人物的详细描述。

**输入：**
1.  **[全景图]**: 一张包含完整场景和多个元素的宽幅或全景图像。
2.  **[人物小图]**: 从 [全景图] 中精确裁剪出来的、聚焦于单个目标人物的特写或半身图像。

**目标：**
根据提供的 **[全景图]** 和 **[人物小图]**，生成一个关于该人物的详细描述。这份描述必须严格遵循下面提供的 `TEMPLATE_JSON` 格式。

**核心规则：**
*   **双重图像分析**: 你必须同时利用两张图片。**[人物小图]** 用于识别人物本身的外貌、服饰、姿态等细节；**[全景图]** 用于理解人物所处的环境、可能的行为、与周围的互动关系，以及推断其身份和目的。
*   **严格遵循格式**: 输出**必须**是一个完整的、可被解析的JSON对象，其结构、键名、数据类型（字符串、列表、对象）都必须与 `TEMPLATE_JSON` 完全一致。任何字段如果无法从图像中获取信息，应使用 `null` 或空列表 `[]` 表示，而不是省略该字段。
*   **观察与推断分离**: 严格区分“可观察属性”和“推断属性”两个部分，这是评价你工作质量的核心标准。

---

### **任务分解与指令**

1.  **全局分析 (Analyze Globally)**: 首先审视 **[全景图]**，了解整体环境（如街道、公园、室内）、光线、天气、时间段以及图中其他人和物的分布，为 `contextualAnalysis` 部分的推断打下基础。
2.  **细节观察 (Observe Details)**: 接着聚焦于 **[人物小图]**，仔细辨认人物的每一个可见细节，包括但不限于：
    *   **外貌**: 发型、可能的肤色、体态。
    *   **服饰**: 上下身衣物和鞋子的类型、颜色、图案、材质和特殊设计。
    *   **配饰与物品**: 所有穿戴、携带或手持的物品。
        *   **特别注意**: 对于 `accessories`, `carriedItems`, `heldObjects` 中的每个物品，`describe_zh` 字段**仅**填写该物品的 `type` (中文)，`describe_en` 字段则填写 `describe_zh` 对应的英文翻译。例如，一个物品的 `type` 是 '腰包'，则 `describe_zh` 应为 '腰包'，`describe_en` 应为 'waist bag'；若物品的 `type` 是 '棒球帽'，则 `describe_zh` 应为 '棒球帽'，`describe_en` 应为 'baseball cap'。**不再包含颜色、材质等其他描述。**
    *   **姿态**: 身体的朝向、站姿/坐姿、手部动作。
    *   这些信息将填充 `observedAttributes` 部分。
3.  **推理与判断 (Infer and Judge)**: 结合从两张图片中获得的信息进行逻辑推理。
    *   基于外貌和服饰，推断性别、年龄段和可能的社会身份 (`demographics`)。
    *   基于面部表情和身体语言，推断情绪和心情 (`emotion_and_expression`)。
    *   基于人物行为和所处环境，推断其具体活动、互动情况、地点类型和可能的目标 (`contextualAnalysis`)。
    *   **使用审慎的词汇**: 在 `inferredAttributes` 部分，多使用“看似”、“可能”、“似乎”、“推断为”等非绝对性词语。
4.  **整合与撰写 (Synthesize and Write)**:
    *   **撰写摘要**: 最后，将所有观察和推断的关键信息融合成流畅的文字，分别填入 `short_description` (一句话标题) 和 `long_description` (生动的叙事段落)。
    *   **结构化输出**: 将所有分析结果填入 `TEMPLATE_JSON` 结构的对应字段中，生成最终的JSON输出。

---

### **关键概念区分：观察 vs. 推断**

*   `observedAttributes` **(可观察属性)**: **只记录事实**。这是对图像内容的客观描述，是任何人都能直接看到的。例如，“身穿一件蓝色T恤”，“右手拿着手机”。**避免任何猜测**。
*   `inferredAttributes` **(推断属性)**: **进行合理的猜测**。这是基于观察到的事实进行的逻辑推理。例如，根据蓝色T恤上的大学Logo和周围的校园环境，推断其身份“可能是学生”。

---

### **输出JSON模板 (TEMPLATE_JSON)**

```json
{
    "summary": {
        "short_description": "string | 一句话新闻标题式摘要，精准概括人物的核心身份、状态和行为。",
        "long_description": "string | 一段生动的叙事性描述，如同电影旁白，融合所有关键信息，为读者构建一幅完整的视觉和情景画面。"
    },
    "observedAttributes": {
        "comment": "此部分只包含可被直接视觉验证的、客观的事实，避免任何主观猜测。",
        "physicalAppearance": {
            "skin_tone_suggestion": "string | e.g., '白皙', '象牙白', '自然色', '小麦色', '橄榄色', '古铜色', '深棕色'",
            "bodyComposition": "string | 对身体肌肉和脂肪分布的客观描述，e.g., '体态纤细', '骨架分明', '肌肉线条明显', '身形圆润', '体格壮硕'",
            "hair": {
                "length": "string | e.g., '光头', '寸头', '短发(耳上)', '齐耳短发', '及肩中发', '长发(过肩)', '及腰长发'",
                "style": "string | e.g., '直发', '自然卷', '大波浪', '羊毛卷', '马尾辫', '丸子头', '编发', '平头', '背头', '油头', '中分', '脏辫'",
                "color": "string | e.g., '纯黑', '深棕', '亚麻金', '奶奶灰', '酒红', '蓝色挑染', '渐变色(如粉到紫)'"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "string | e.g., 'T恤', '衬衫', '卫衣', '夹克', '大衣'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '条纹', '格子', '品牌Logo'",
                "material": "string | e.g., '棉质', '牛仔布', '丝绸', '羊毛', '皮革', '防水尼龙'",
                "details": "array[string] | e.g., ['V领', '带兜帽', '袖子卷起']"
            },
            "lowerBody": {
                "type": "string | e.g., '牛仔裤', '休闲裤', '运动裤', '短裙'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '水洗效果', '格纹'",
                "material": "string | e.g., '牛仔布', '棉布', '西装面料', '灯芯绒'",
                "details": "array[string] | e.g., ['膝盖破洞', '裤脚磨边', '系有皮带']"
            },
            "footwear": {
                "type": "string | e.g., '运动鞋', '帆布鞋', '皮鞋', '靴子'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '拼色', '格子(如Vans棋盘格)', '迷彩', '动物纹理(如豹纹)', '品牌Logo重复图案'",
                "material": "string | e.g., '皮革', '帆布', '绒面', '橡胶'",
                "details": "array[string] | e.g., ['厚底', '高帮设计', '品牌标志醒目']"
            }
        },
        "accessories": [
            {
                "item_name": "string | 穿戴在身上的小型饰品, e.g., '眼镜', '手表', '首饰', '帽子'",
                "type": "string | e.g., '太阳镜', '电子手表', '金项链', '棒球帽'",
                "color": "array[string]",
                "material": "string | e.g., '金属', '皮革', '塑料', '布料', '珍珠', '宝石'",
                "details": "array[string] | e.g., ['戴在头上', '挂在脖子上']",
                "describe_zh": "string | 对该饰品的中文描述，仅包含物品的 `type`，如 '棒球帽', '手表'",
                "describe_en": "string | describe_zh 对应的英文翻译, e.g., 'baseball cap', 'watch'"
            }
        ],
        "carriedItems": [
            {
                "item_name": "string | 携带的、非手持的大件物品, e.g., '包', '行李箱'",
                "type": "string | e.g., '双肩包', '手提电脑包', '硬壳拉杆箱'",
                "color": "array[string]",
                "material": "string | e.g., '皮革', '帆布', '尼龙', 'PC/ABS塑料'",
                "details": "array[string] | e.g., ['背在身后', '放置在地面']",
                "describe_zh": "string | 对该携带物品的中文描述，仅包含物品的 `type`，如 '双肩包', '腰包'",
                "describe_en": "string | describe_zh 对应的英文翻译, e.g., 'backpack', 'waist bag'"
            }
        ],
        "heldObjects": [
            {
                "item_name": "string | 当前正用手持握的物品, e.g., '手机', '水瓶', '雨伞'",
                "type": "string | e.g., '智能手机', '保温杯', '折叠伞'",
                "color": "array[string]",
                "material": "string | e.g., '塑料', '金属', '玻璃', '纸质'",
                "details": "array[string] | e.g., ['右手持握', '屏幕亮起']",
                "describe_zh": "string | 对该手持物品的中文描述，仅包含物品的 `type`，如 '智能手机', '素描本'",
                "describe_en": "string | describe_zh 对应的英文翻译, e.g., 'smartphone', 'sketchbook'"
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "string | e.g., '正对', '背对', '左侧面', '左前45度'",
            "posture": "string | e.g., '直立', '坐着', '行走中', '倚靠'",
            "gestures": "array[string] | e.g., ['双手插兜', '双臂交叉于胸前', '正在用手机打字']"
        }
    },
    "inferredAttributes": {
        "comment": "此部分包含基于视觉观察进行的合理推断，应使用审慎的、非绝对的词汇。",
        "demographics": {
            "estimated_gender": "string | '看似男性', '看似女性', '性别特征不明显'",
            "estimated_age_group": "string | '婴幼儿', '儿童', '青少年', '青年', '中年', '老年'",
            "possible_identity": "array[string] | e.g., ['学生', '游客', '上班族', '时尚博主']"
        },
        "emotion_and_expression": {
            "facial_expression_observed": "string | e.g., '面部肌肉放松', '嘴角上扬', '眉毛紧蹙'",
            "inferred_emotion": "string | e.g., '心情愉悦', '专注', '困惑', '情绪中性'",
            "inferred_mood_from_posture": "string | e.g., '姿态放松', '身体紧绷显得紧张', '步伐轻快看似愉悦', '垂头丧气略显沮丧'"
        },
        "contextualAnalysis": {
            "actions": "array[string] | e.g., ['打电话', '喝水', '拍照', '等候']",
            "interactions": {
                "with_people": "string | e.g., '与人并肩交谈', '独自一人'",
                "with_objects": "string | e.g., '倚靠栏杆', '操作笔记本电脑'",
                "with_vehicles": "string | e.g., '正在骑自行车', '准备上公交车'"
            },
            "estimated_location_type": "string | e.g., '室外街道', '公园', '地铁站台', '购物中心'",
            "estimated_purpose_or_goal": "string | e.g., '似乎在等候某人', '可能正在去上班的路上', '像是在进行休闲购物', '看起来在锻炼身体'"
        }
    }
}
```

---

### **高质量输出示例 (学习以下示例的风格、细节程度和推理逻辑)**

**(示例1: 商务女性)**
```json
{
    "summary": {
        "short_description": "一位身穿卡其色风衣的职业女性在现代建筑前，正严肃地进行手机通话。",
        "long_description": "在繁华都市的街角，一位干练的职业女性正驻足通话。她身着经典的卡其色风衣，内搭简约的白衬衫和黑色西裤，脚踩一双黑色高跟鞋，尽显专业气质。她左手持握着手机贴近耳边，表情严肃而专注，右手不时做出细微的手势，似乎在与电话那头的人讨论着重要事宜。身旁的皮质托特包暗示了她的通勤者身份，整个人散发出一种高效而自信的气场。"
    },
    "observedAttributes": {
        "physicalAppearance": {
            "skin_tone_suggestion": "白皙",
            "bodyComposition": "身形高挑匀称",
            "hair": {
                "length": "及肩中发",
                "style": "大波浪",
                "color": "深棕"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "风衣",
                "color": ["卡其色"],
                "pattern": "纯色",
                "material": "棉质/防水面料",
                "details": ["腰带系起", "双排扣设计", "内搭白色衬衫"]
            },
            "lowerBody": {
                "type": "西裤",
                "color": ["黑色"],
                "pattern": "纯色",
                "material": "西装面料",
                "details": ["九分裤剪裁"]
            },
            "footwear": {
                "type": "高跟鞋",
                "color": ["黑色"],
                "pattern": "纯色",
                "material": "皮革",
                "details": ["尖头设计"]
            }
        },
        "accessories": [
            {
                "item_name": "手表",
                "type": "机械表",
                "color": ["银色", "白色"],
                "material": "金属",
                "details": ["戴在左手腕"],
                "describe_zh": "机械表",
                "describe_en": "mechanical watch"
            }
        ],
        "carriedItems": [
            {
                "item_name": "包",
                "type": "手提托特包",
                "color": ["棕色"],
                "material": "皮革",
                "details": ["放置在身体右侧地面"],
                "describe_zh": "手提托特包",
                "describe_en": "tote bag"
            }
        ],
        "heldObjects": [
            {
                "item_name": "手机",
                "type": "智能手机",
                "color": ["黑色"],
                "material": "玻璃/金属",
                "details": ["左手持握贴近耳边"],
                "describe_zh": "智能手机",
                "describe_en": "smartphone"
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "右前45度",
            "posture": "直立",
            "gestures": ["左手持手机通话", "右手偶尔做出手势"]
        }
    },
    "inferredAttributes": {
        "demographics": {
            "estimated_gender": "看似女性",
            "estimated_age_group": "青年",
            "possible_identity": ["上班族", "商务人士", "管理者"]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "眉毛微蹙",
            "inferred_emotion": "专注",
            "inferred_mood_from_posture": "身体挺直显得干练"
        },
        "contextualAnalysis": {
            "actions": ["打电话"],
            "interactions": {
                "with_people": "通过电话与人互动",
                "with_objects": "使用手机",
                "with_vehicles": "无"
            },
            "estimated_location_type": "室外街道",
            "estimated_purpose_or_goal": "正在进行商务通话"
        }
    }
}
```
**(示例2: 在公园草地上创作的年轻学生)**
```json
{
    "summary": {
        "short_description": "一位戴着针织帽的年轻人在公园草地上盘腿而坐，一边听音乐一边在素描本上画画。",
        "long_description": "在阳光和煦的公园草坪上，一位充满艺术气息的年轻人正沉浸在自己的创作世界里。他戴着一顶灰色的针织帽，脖子上挂着头戴式耳机，身上穿着一件敞开的格子衬衫，内搭印花T恤。他盘腿而坐，面前摊开着一个素描本，右手拿着铅笔在上面不停地描绘。他身旁放着一个磨旧的帆布双肩包，整个画面充满了青春、自由和创造力。"
    },
    "observedAttributes": {
        "physicalAppearance": {
            "skin_tone_suggestion": "自然色",
            "bodyComposition": "体态纤细",
            "hair": {
                "length": "短发(耳上)",
                "style": "大部分被帽子遮挡",
                "color": "黑色带蓝色挑染"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "格子衬衫",
                "color": [
                    "红色",
                    "黑色"
                ],
                "pattern": "格子",
                "material": "棉质",
                "details": [
                    "敞开穿着",
                    "袖子卷起",
                    "内搭白色印花T恤"
                ]
            },
            "lowerBody": {
                "type": "牛仔裤",
                "color": [
                    "黑色"
                ],
                "pattern": "水洗效果",
                "material": "牛仔布",
                "details": [
                    "膝盖处有破洞"
                ]
            },
            "footwear": {
                "type": "靴子(马丁靴)",
                "color": [
                    "黑色"
                ],
                "pattern": "纯色",
                "material": "皮革",
                "details": [
                    "高帮设计",
                    "鞋带部分松开"
                ]
            }
        },
        "accessories": [
            {
                "item_name": "帽子",
                "type": "针织帽",
                "color": [
                    "深灰色"
                ],
                "material": "羊毛",
                "details": [
                    "戴在头上"
                ],
                "describe_zh": "针织帽",
                "describe_en": "knit cap"
            },
            {
                "item_name": "耳机",
                "type": "头戴式耳机",
                "color": [
                    "黑色",
                    "银色"
                ],
                "material": "塑料/金属",
                "details": [
                    "挂在脖子上"
                ],
                "describe_zh": "头戴式耳机",
                "describe_en": "over-ear headphones"
            }
        ],
        "carriedItems": [
            {
                "item_name": "包",
                "type": "双肩包",
                "color": [
                    "军绿色"
                ],
                "material": "帆布",
                "details": [
                    "放置在身体左侧草地上"
                ],
                "describe_zh": "双肩包",
                "describe_en": "backpack"
            }
        ],
        "heldObjects": [
            {
                "item_name": "笔记本",
                "type": "素描本",
                "color": [
                    "牛皮纸色"
                ],
                "material": "纸质",
                "details": [
                    "摊开放在腿上"
                ],
                "describe_zh": "素描本",
                "describe_en": "sketchbook"
            },
            {
                "item_name": "笔",
                "type": "铅笔",
                "color": [
                    "原木色"
                ],
                "material": "木质",
                "details": [
                    "右手持握正在绘画"
                ],
                "describe_zh": "铅笔",
                "describe_en": "pencil"
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "正对",
            "posture": "坐着",
            "gestures": [
                "低头看笔记本",
                "右手执笔绘画"
            ]
        }
    },
    "inferredAttributes": {
        "demographics": {
            "estimated_gender": "性别特征不明显",
            "estimated_age_group": "青少年(12-17)",
            "possible_identity": [
                "学生",
                "艺术爱好者",
                "街头艺人"
            ]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "面部肌肉放松",
            "inferred_emotion": "专注",
            "inferred_mood_from_posture": "姿态放松，沉浸在自己的世界"
        },
        "contextualAnalysis": {
            "actions": [
                "绘画",
                "听音乐",
                "休息"
            ],
            "interactions": {
                "with_people": "独自一人",
                "with_objects": "使用素描本和笔",
                "with_vehicles": "无"
            },
            "estimated_location_type": "公园绿地",
            "estimated_purpose_or_goal": "似乎在进行素描创作或记录灵感"
        }
    }
}
```
**(示例3: 在湖边步道锻炼的活力长者)**
```json
{
    "summary": {
        "short_description": "一位戴着太阳镜、身穿运动装备的老年男士在湖边步道上快走，显得精力充沛。",
        "long_description": "在风景宜人的湖边步道上，一位精神矍铄的老年男士正在进行日常锻炼。他戴着一副运动太阳镜，身穿一件亮色的速干T恤和运动短裤，脚上的徒步鞋显示出他对户外活动的热爱。他步伐稳健地向前快走，双臂自然摆动，展现出与其年龄不符的活力。他左手手腕上的运动手环和腰间的小包表明他是一位经验丰富的健身者，正在享受健康积极的退休生活。"
    },
    "observedAttributes": {
        "physicalAppearance": {
            "skin_tone_suggestion": "古铜色",
            "bodyComposition": "身形健朗",
            "hair": {
                "length": "短发(耳上)",
                "style": "自然发型",
                "color": "灰色/银色"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "T恤",
                "color": [
                    "橙色"
                ],
                "pattern": "品牌Logo",
                "material": "速干面料",
                "details": [
                    "圆领"
                ]
            },
            "lowerBody": {
                "type": "运动短裤",
                "color": [
                    "深灰色"
                ],
                "pattern": "纯色",
                "material": "尼龙",
                "details": [
                    "侧边有反光条"
                ]
            },
            "footwear": {
                "type": "运动鞋",
                "color": [
                    "蓝色",
                    "白色"
                ],
                "pattern": "拼色",
                "material": "网面/橡胶",
                "details": [
                    "专业的徒步鞋款式"
                ]
            }
        },
        "accessories": [
            {
                "item_name": "眼镜",
                "type": "太阳镜",
                "color": [
                    "黑色"
                ],
                "material": "塑料",
                "details": [
                    "戴在脸上",
                    "运动款式"
                ],
                "describe_zh": "太阳镜",
                "describe_en": "sunglasses"
            },
            {
                "item_name": "手表",
                "type": "电子手表",
                "color": [
                    "黑色"
                ],
                "material": "塑料/硅胶",
                "details": [
                    "运动手环款式",
                    "戴在左手腕"
                ],
                "describe_zh": "电子手表",
                "describe_en": "electronic watch"
            }
        ],
        "carriedItems": [
            {
                "item_name": "包",
                "type": "腰包",
                "color": [
                    "黑色"
                ],
                "material": "尼龙",
                "details": [
                    "系在腰间"
                ],
                "describe_zh":"腰包",
                "describe_en":"waist bag"
            }
        ],
        "heldObjects": [],
        "postureAndOrientation": {
            "body_orientation": "左侧面",
            "posture": "行走中",
            "gestures": [
                "双臂自然前后摆动"
            ]
        }
    },
    "inferredAttributes": {
        "demographics": {
            "estimated_gender": "看似男性",
            "estimated_age_group": "老年(56+)",
            "possible_identity": [
                "徒步者",
                "退休人士",
                "运动爱好者"
            ]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "面部肌肉放松",
            "inferred_emotion": "心情愉悦",
            "inferred_mood_from_posture": "步伐轻快看似愉悦"
        },
        "contextualAnalysis": {
            "actions": [
                "快走",
                "锻炼"
            ],
            "interactions": {
                "with_people": "独自一人",
                "with_objects": "无",
                "with_vehicles": "无"
            },
            "estimated_location_type": "公园",
            "estimated_purpose_or_goal": "正在进行户外锻炼，享受自然风光"
        }
    }
}
```
**最终指令：**
请根据下面提供的 **[全景图]** 和 **[人物小图]**，严格遵循上述所有规则、指令和JSON模板，生成最终的描述。现在，开始分析。
"""



prompt_double_things_item = """**角色：** 你是一个专业的人体与场景分析AI，具备敏锐的观察力和严谨的逻辑推理能力。你的任务是分析图像，并以高度结构化的JSON格式输出对特定人物的详细描述。

**输入：**
1.  **[全景图]**: 一张包含完整场景和多个元素的宽幅或全景图像。
2.  **[人物小图]**: 从 [全景图] 中精确裁剪出来的、聚焦于单个目标人物的特写或半身图像。

**目标：**
根据提供的 **[全景图]** 和 **[人物小图]**，生成一个关于该人物的详细描述。这份描述必须严格遵循下面提供的 `TEMPLATE_JSON` 格式。

**核心规则：**
*   **双重图像分析**: 你必须同时利用两张图片。**[人物小图]** 用于识别人物本身的外貌、服饰、姿态等细节；**[全景图]** 用于理解人物所处的环境、可能的行为、与周围的互动关系，以及推断其身份和目的。
*   **严格遵循格式**: 输出**必须**是一个完整的、可被解析的JSON对象，其结构、键名、数据类型（字符串、列表、对象）都必须与 `TEMPLATE_JSON` 完全一致。任何字段如果无法从图像中获取信息，应使用 `null` 或空列表 `[]` 表示，而不是省略该字段。
*   **观察与推断分离**: 严格区分“可观察属性”和“推断属性”两个部分，这是评价你工作质量的核心标准。

---

### **任务分解与指令**

1.  **全局分析 (Analyze Globally)**: 首先审视 **[全景图]**，了解整体环境（如街道、公园、室内）、光线、天气、时间段以及图中其他人和物的分布，为 `contextualAnalysis` 部分的推断打下基础。
2.  **细节观察 (Observe Details)**: 接着聚焦于 **[人物小图]**，仔细辨认人物的每一个可见细节，包括但不限于：
    *   **外貌**: 发型、可能的肤色、体态。
    *   **服饰**: 上下身衣物和鞋子的类型、颜色、图案、材质和特殊设计。
    *   **配饰与物品**: 所有穿戴、携带或手持的物品。
    *   **姿态**: 身体的朝向、站姿/坐姿、手部动作。
    *   这些信息将填充 `observedAttributes` 部分。
3.  **推理与判断 (Infer and Judge)**: 结合从两张图片中获得的信息进行逻辑推理。
    *   基于外貌和服饰，推断性别、年龄段和可能的社会身份 (`demographics`)。
    *   基于面部表情和身体语言，推断情绪和心情 (`emotion_and_expression`)。
    *   基于人物行为和所处环境，推断其具体活动、互动情况、地点类型和可能的目标 (`contextualAnalysis`)。
    *   **使用审慎的词汇**: 在 `inferredAttributes` 部分，多使用“看似”、“可能”、“似乎”、“推断为”等非绝对性词语。
4.  **整合与撰写 (Synthesize and Write)**:
    *   **撰写摘要**: 最后，将所有观察和推断的关键信息融合成流畅的文字，分别填入 `short_description` (一句话标题) 和 `long_description` (生动的叙事段落)。
    *   **结构化输出**: 将所有分析结果填入 `TEMPLATE_JSON` 结构的对应字段中，生成最终的JSON输出。

---

### **关键概念区分：观察 vs. 推断**

*   `observedAttributes` **(可观察属性)**: **只记录事实**。这是对图像内容的客观描述，是任何人都能直接看到的。例如，“身穿一件蓝色T恤”，“右手拿着手机”。**避免任何猜测**。
*   `inferredAttributes` **(推断属性)**: **进行合理的猜测**。这是基于观察到的事实进行的逻辑推理。例如，根据蓝色T恤上的大学Logo和周围的校园环境，推断其身份“可能是学生”。

---

### **输出JSON模板 (TEMPLATE_JSON)**

```json
{
    "summary": {
        "short_description": "string | 一句话新闻标题式摘要，精准概括人物的核心身份、状态和行为。",
        "long_description": "string | 一段生动的叙事性描述，如同电影旁白，融合所有关键信息，为读者构建一幅完整的视觉和情景画面。"
    },
    "observedAttributes": {
        "comment": "此部分只包含可被直接视觉验证的、客观的事实，避免任何主观猜测。",
        "physicalAppearance": {
            "skin_tone_suggestion": "string | e.g., '白皙', '象牙白', '自然色', '小麦色', '橄榄色', '古铜色', '深棕色'",
            "bodyComposition": "string | 对身体肌肉和脂肪分布的客观描述，e.g., '体态纤细', '骨架分明', '肌肉线条明显', '身形圆润', '体格壮硕'",
            "hair": {
                "length": "string | e.g., '光头', '寸头', '短发(耳上)', '齐耳短发', '及肩中发', '长发(过肩)', '及腰长发'",
                "style": "string | e.g., '直发', '自然卷', '大波浪', '羊毛卷', '马尾辫', '丸子头', '编发', '平头', '背头', '油头', '中分', '脏辫'",
                "color": "string | e.g., '纯黑', '深棕', '亚麻金', '奶奶灰', '酒红', '蓝色挑染', '渐变色(如粉到紫)'"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "string | e.g., 'T恤', '衬衫', '卫衣', '夹克', '大衣'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '条纹', '格子', '品牌Logo'",
                "material": "string | e.g., '棉质', '牛仔布', '丝绸', '羊毛', '皮革', '防水尼龙'",
                "details": "array[string] | e.g., ['V领', '带兜帽', '袖子卷起']"
            },
            "lowerBody": {
                "type": "string | e.g., '牛仔裤', '休闲裤', '运动裤', '短裙'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '水洗效果', '格纹'",
                "material": "string | e.g., '牛仔布', '棉布', '西装面料', '灯芯绒'",
                "details": "array[string] | e.g., ['膝盖破洞', '裤脚磨边', '系有皮带']"
            },
            "footwear": {
                "type": "string | e.g., '运动鞋', '帆布鞋', '皮鞋', '靴子'",
                "color": "array[string]",
                "pattern": "string | e.g., '纯色', '拼色', '格子(如Vans棋盘格)', '迷彩', '动物纹理(如豹纹)', '品牌Logo重复图案'",
                "material": "string | e.g., '皮革', '帆布', '绒面', '橡胶'",
                "details": "array[string] | e.g., ['厚底', '高帮设计', '品牌标志醒目']"
            }
        },
        "accessories": [
            {
                "item_name": "string | 穿戴在身上的小型饰品, e.g., '眼镜', '手表', '首饰', '帽子'",
                "type": "string | e.g., '太阳镜', '电子手表', '金项链', '棒球帽'",
                "color": "array[string]",
                "material": "string | e.g., '金属', '皮革', '塑料', '布料', '珍珠', '宝石'",
                "details": "array[string] | e.g., ['戴在头上', '挂在脖子上']",
                "describe_zh": "string | 对该饰品的中文描述，直接使用 item_name 的值，如 '手表', '帽子'",
                "describe_en": "string | describe_zh (即 item_name) 对应的英文翻译, e.g., 'watch', 'hat'"
            }
        ],
        "carriedItems": [
            {
                "item_name": "string | 携带的、非手持的大件物品, e.g., '包', '行李箱'",
                "type": "string | e.g., '双肩包', '手提电脑包', '硬壳拉杆箱'",
                "color": "array[string]",
                "material": "string | e.g., '皮革', '帆布', '尼龙', 'PC/ABS塑料'",
                "details": "array[string] | e.g., ['背在身后', '放置在地面']",
                "describe_zh": "string | 对该携带物品的中文描述，直接使用 item_name 的值，如 '包', '行李箱'",
                "describe_en": "string | describe_zh (即 item_name) 对应的英文翻译, e.g., 'bag', 'suitcase'"
            }
        ],
        "heldObjects": [
            {
                "item_name": "string | 当前正用手持握的物品, e.g., '手机', '水瓶', '雨伞', '笔记本', '笔'",
                "type": "string | e.g., '智能手机', '保温杯', '折叠伞', '素描本', '铅笔'",
                "color": "array[string]",
                "material": "string | e.g., '塑料', '金属', '玻璃', '纸质'",
                "details": "array[string] | e.g., ['右手持握', '屏幕亮起']",
                "describe_zh": "string | 对该手持物品的中文描述，直接使用 item_name 的值，如 '手机', '笔记本'",
                "describe_en": "string | describe_zh (即 item_name) 对应的英文翻译, e.g., 'phone', 'notebook'"
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "string | e.g., '正对', '背对', '左侧面', '左前45度'",
            "posture": "string | e.g., '直立', '坐着', '行走中', '倚靠'",
            "gestures": "array[string] | e.g., ['双手插兜', '双臂交叉于胸前', '正在用手机打字']"
        }
    },
    "inferredAttributes": {
        "comment": "此部分包含基于视觉观察进行的合理推断，应使用审慎的、非绝对的词汇。",
        "demographics": {
            "estimated_gender": "string | '看似男性', '看似女性', '性别特征不明显'",
            "estimated_age_group": "string | '婴幼儿', '儿童', '青少年', '青年', '中年', '老年'",
            "possible_identity": "array[string] | e.g., ['学生', '游客', '上班族', '时尚博主']"
        },
        "emotion_and_expression": {
            "facial_expression_observed": "string | e.g., '面部肌肉放松', '嘴角上扬', '眉毛紧蹙'",
            "inferred_emotion": "string | e.g., '心情愉悦', '专注', '困惑', '情绪中性'",
            "inferred_mood_from_posture": "string | e.g., '姿态放松', '身体紧绷显得紧张', '步伐轻快看似愉悦', '垂头丧气略显沮丧'"
        },
        "contextualAnalysis": {
            "actions": "array[string] | e.g., ['打电话', '喝水', '拍照', '等候']",
            "interactions": {
                "with_people": "string | e.g., '与人并肩交谈', '独自一人'",
                "with_objects": "string | e.g., '倚靠栏杆', '操作笔记本电脑'",
                "with_vehicles": "string | e.g., '正在骑自行车', '准备上公交车'"
            },
            "estimated_location_type": "string | e.g., '室外街道', '公园', '地铁站台', '购物中心'",
            "estimated_purpose_or_goal": "string | e.g., '似乎在等候某人', '可能正在去上班的路上', '像是在进行休闲购物', '看起来在锻炼身体'"
        }
    }
}
```

---

### **高质量输出示例 (学习以下示例的风格、细节程度和推理逻辑)**

**(示例1: 商务女性)**
```json
{
    "summary": {
        "short_description": "一位身穿卡其色风衣的职业女性在现代建筑前，正严肃地进行手机通话。",
        "long_description": "在繁华都市的街角，一位干练的职业女性正驻足通话。她身着经典的卡其色风衣，内搭简约的白衬衫和黑色西裤，脚踩一双黑色高跟鞋，尽显专业气质。她左手持握着手机贴近耳边，表情严肃而专注，右手不时做出细微的手势，似乎在与电话那头的人讨论着重要事宜。身旁的皮质托特包暗示了她的通勤者身份，整个人散发出一种高效而自信的气场。"
    },
    "observedAttributes": {
        "physicalAppearance": {
            "skin_tone_suggestion": "白皙",
            "bodyComposition": "身形高挑匀称",
            "hair": {
                "length": "及肩中发",
                "style": "大波浪",
                "color": "深棕"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "风衣",
                "color": ["卡其色"],
                "pattern": "纯色",
                "material": "棉质/防水面料",
                "details": ["腰带系起", "双排扣设计", "内搭白色衬衫"]
            },
            "lowerBody": {
                "type": "西裤",
                "color": ["黑色"],
                "pattern": "纯色",
                "material": "西装面料",
                "details": ["九分裤剪裁"]
            },
            "footwear": {
                "type": "高跟鞋",
                "color": ["黑色"],
                "pattern": "纯色",
                "material": "皮革",
                "details": ["尖头设计"]
            }
        },
        "accessories": [
            {
                "item_name": "手表",
                "type": "机械表",
                "color": ["银色", "白色"],
                "material": "金属",
                "details": ["戴在左手腕"],
                "describe_zh": "手表",
                "describe_en": "watch"
            }
        ],
        "carriedItems": [
            {
                "item_name": "包",
                "type": "手提托特包",
                "color": ["棕色"],
                "material": "皮革",
                "details": ["放置在身体右侧地面"],
                "describe_zh": "包",
                "describe_en": "bag"
            }
        ],
        "heldObjects": [
            {
                "item_name": "手机",
                "type": "智能手机",
                "color": ["黑色"],
                "material": "玻璃/金属",
                "details": ["左手持握贴近耳边"],
                "describe_zh": "手机",
                "describe_en": "phone"
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "右前45度",
            "posture": "直立",
            "gestures": ["左手持手机通话", "右手偶尔做出手势"]
        }
    },
    "inferredAttributes": {
        "demographics": {
            "estimated_gender": "看似女性",
            "estimated_age_group": "青年",
            "possible_identity": ["上班族", "商务人士", "管理者"]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "眉毛微蹙",
            "inferred_emotion": "专注",
            "inferred_mood_from_posture": "身体挺直显得干练"
        },
        "contextualAnalysis": {
            "actions": ["打电话"],
            "interactions": {
                "with_people": "通过电话与人互动",
                "with_objects": "使用手机",
                "with_vehicles": "无"
            },
            "estimated_location_type": "室外街道",
            "estimated_purpose_or_goal": "正在进行商务通话"
        }
    }
}
```
**(示例2: 在公园草地上创作的年轻学生)**
```json
{
    "summary": {
        "short_description": "一位戴着针织帽的年轻人在公园草地上盘腿而坐，一边听音乐一边在素描本上画画。",
        "long_description": "在阳光和煦的公园草坪上，一位充满艺术气息的年轻人正沉浸在自己的创作世界里。他戴着一顶灰色的针织帽，脖子上挂着头戴式耳机，身上穿着一件敞开的格子衬衫，内搭印花T恤。他盘腿而坐，面前摊开着一个素描本，右手拿着铅笔在上面不停地描绘。他身旁放着一个磨旧的帆布双肩包，整个画面充满了青春、自由和创造力。"
    },
    "observedAttributes": {
        "physicalAppearance": {
            "skin_tone_suggestion": "自然色",
            "bodyComposition": "体态纤细",
            "hair": {
                "length": "短发(耳上)",
                "style": "大部分被帽子遮挡",
                "color": "黑色带蓝色挑染"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "格子衬衫",
                "color": [
                    "红色",
                    "黑色"
                ],
                "pattern": "格子",
                "material": "棉质",
                "details": [
                    "敞开穿着",
                    "袖子卷起",
                    "内搭白色印花T恤"
                ]
            },
            "lowerBody": {
                "type": "牛仔裤",
                "color": [
                    "黑色"
                ],
                "pattern": "水洗效果",
                "material": "牛仔布",
                "details": [
                    "膝盖处有破洞"
                ]
            },
            "footwear": {
                "type": "靴子(马丁靴)",
                "color": [
                    "黑色"
                ],
                "pattern": "纯色",
                "material": "皮革",
                "details": [
                    "高帮设计",
                    "鞋带部分松开"
                ]
            }
        },
        "accessories": [
            {
                "item_name": "帽子",
                "type": "针织帽",
                "color": [
                    "深灰色"
                ],
                "material": "羊毛",
                "details": [
                    "戴在头上"
                ],
                "describe_zh": "帽子",
                "describe_en": "hat"
            },
            {
                "item_name": "耳机",
                "type": "头戴式耳机",
                "color": [
                    "黑色",
                    "银色"
                ],
                "material": "塑料/金属",
                "details": [
                    "挂在脖子上"
                ],
                "describe_zh": "耳机",
                "describe_en": "headphones"
            }
        ],
        "carriedItems": [
            {
                "item_name": "包",
                "type": "双肩包",
                "color": [
                    "军绿色"
                ],
                "material": "帆布",
                "details": [
                    "放置在身体左侧草地上"
                ],
                "describe_zh": "包",
                "describe_en": "bag"
            }
        ],
        "heldObjects": [
            {
                "item_name": "笔记本",
                "type": "素描本",
                "color": [
                    "牛皮纸色"
                ],
                "material": "纸质",
                "details": [
                    "摊开放在腿上"
                ],
                "describe_zh": "笔记本",
                "describe_en": "notebook"
            },
            {
                "item_name": "笔",
                "type": "铅笔",
                "color": [
                    "原木色"
                ],
                "material": "木质",
                "details": [
                    "右手持握正在绘画"
                ],
                "describe_zh": "笔",
                "describe_en": "pen"
            }
        ],
        "postureAndOrientation": {
            "body_orientation": "正对",
            "posture": "坐着",
            "gestures": [
                "低头看笔记本",
                "右手执笔绘画"
            ]
        }
    },
    "inferredAttributes": {
        "demographics": {
            "estimated_gender": "性别特征不明显",
            "estimated_age_group": "青少年(12-17)",
            "possible_identity": [
                "学生",
                "艺术爱好者",
                "街头艺人"
            ]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "面部肌肉放松",
            "inferred_emotion": "专注",
            "inferred_mood_from_posture": "姿态放松，沉浸在自己的世界"
        },
        "contextualAnalysis": {
            "actions": [
                "绘画",
                "听音乐",
                "休息"
            ],
            "interactions": {
                "with_people": "独自一人",
                "with_objects": "使用素描本和笔",
                "with_vehicles": "无"
            },
            "estimated_location_type": "公园绿地",
            "estimated_purpose_or_goal": "似乎在进行素描创作或记录灵感"
        }
    }
}
```
**(示例3: 在湖边步道锻炼的活力长者)**
```json
{
    "summary": {
        "short_description": "一位戴着太阳镜、身穿运动装备的老年男士在湖边步道上快走，显得精力充沛。",
        "long_description": "在风景宜人的湖边步道上，一位精神矍铄的老年男士正在进行日常锻炼。他戴着一副运动太阳镜，身穿一件亮色的速干T恤和运动短裤，脚上的徒步鞋显示出他对户外活动的热爱。他步伐稳健地向前快走，双臂自然摆动，展现出与其年龄不符的活力。他左手手腕上的运动手环和腰间的小包表明他是一位经验丰富的健身者，正在享受健康积极的退休生活。"
    },
    "observedAttributes": {
        "physicalAppearance": {
            "skin_tone_suggestion": "古铜色",
            "bodyComposition": "身形健朗",
            "hair": {
                "length": "短发(耳上)",
                "style": "自然发型",
                "color": "灰色/银色"
            }
        },
        "apparel": {
            "upperBody": {
                "type": "T恤",
                "color": [
                    "橙色"
                ],
                "pattern": "品牌Logo",
                "material": "速干面料",
                "details": [
                    "圆领"
                ]
            },
            "lowerBody": {
                "type": "运动短裤",
                "color": [
                    "深灰色"
                ],
                "pattern": "纯色",
                "material": "尼龙",
                "details": [
                    "侧边有反光条"
                ]
            },
            "footwear": {
                "type": "运动鞋",
                "color": [
                    "蓝色",
                    "白色"
                ],
                "pattern": "拼色",
                "material": "网面/橡胶",
                "details": [
                    "专业的徒步鞋款式"
                ]
            }
        },
        "accessories": [
            {
                "item_name": "眼镜",
                "type": "太阳镜",
                "color": [
                    "黑色"
                ],
                "material": "塑料",
                "details": [
                    "戴在脸上",
                    "运动款式"
                ],
                "describe_zh": "眼镜",
                "describe_en": "glasses"
            },
            {
                "item_name": "手表",
                "type": "电子手表",
                "color": [
                    "黑色"
                ],
                "material": "塑料/硅胶",
                "details": [
                    "运动手环款式",
                    "戴在左手腕"
                ],
                "describe_zh": "手表",
                "describe_en": "watch"
            }
        ],
        "carriedItems": [
            {
                "item_name": "包",
                "type": "腰包",
                "color": [
                    "黑色"
                ],
                "material": "尼龙",
                "details": [
                    "系在腰间"
                ],
                "describe_zh":"包",
                "describe_en":"bag"
            }
        ],
        "heldObjects": [],
        "postureAndOrientation": {
            "body_orientation": "左侧面",
            "posture": "行走中",
            "gestures": [
                "双臂自然前后摆动"
            ]
        }
    },
    "inferredAttributes": {
        "demographics": {
            "estimated_gender": "看似男性",
            "estimated_age_group": "老年(56+)",
            "possible_identity": [
                "徒步者",
                "退休人士",
                "运动爱好者"
            ]
        },
        "emotion_and_expression": {
            "facial_expression_observed": "面部肌肉放松",
            "inferred_emotion": "心情愉悦",
            "inferred_mood_from_posture": "步伐轻快看似愉悦"
        },
        "contextualAnalysis": {
            "actions": [
                "快走",
                "锻炼"
            ],
            "interactions": {
                "with_people": "独自一人",
                "with_objects": "无",
                "with_vehicles": "无"
            },
            "estimated_location_type": "公园",
            "estimated_purpose_or_goal": "正在进行户外锻炼，享受自然风光"
        }
    }
}
```
**最终指令：**
请根据下面提供的 **[全景图]** 和 **[人物小图]**，严格遵循上述所有规则、指令和JSON模板，生成最终的描述。现在，开始分析。
"""

# 全局变量存储结果和已处理文件集合
all_results = []
processed_pairs_set = set()  # 已处理的图片对集合
processed_items_set = set()  # 已处理的图片+坐标组合集合
write_lock = asyncio.Lock()  # 异步锁用于并发安全写入

class BatchProcessUtil:
    def __init__(self, batch_size=10):
        self.batch_size = batch_size
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    async def process(self, return_exceptions=False):
        results = []
        for i in range(0, len(self.tasks), self.batch_size):
            start = asyncio.get_event_loop().time()
            batch_tasks = self.tasks[i:i + self.batch_size]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=return_exceptions)
            results.extend(batch_results)
            print(f"Completed: {i + self.batch_size}/{len(self.tasks)}, Cost: {int(asyncio.get_event_loop().time() - start)}s")
        return results

    def clear(self):
        self.tasks.clear()

async def save_json(output_json_path):
    global all_results
    async with write_lock:
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"Failed to write JSON: {str(e)}")

def parse_label_file(label_path, img_width, img_height):
    """解析标签文件并返回归一化到1000尺度的坐标框列表和isuse标记
    Args:
        label_path: 标签文件路径
        img_width: 对应图片的宽度
        img_height: 对应图片的高度
    Returns:
        tuple: (normalized_bboxes, isuses) 归一化坐标框列表和isuse标记列表
    """
    normalized_bboxes = []
    isuses = []
    
    if not os.path.exists(label_path):
        return normalized_bboxes, isuses
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(float(parts[0]))
                if class_id == 0:
                    # 原始坐标转换并归一化
                    x1, y1, x2, y2 = map(lambda x: int(float(x)), parts[1:5])
                    
                    # 坐标归一化计算 (防止除零错误)
                    width_scale = img_width if img_width > 0 else 1
                    height_scale = img_height if img_height > 0 else 1
                    
                    x1_new = int(x1 / width_scale * 1000)
                    y1_new = int(y1 / height_scale * 1000)
                    x2_new = int(x2 / width_scale * 1000)
                    y2_new = int(y2 / height_scale * 1000)
                    
                    
                    # x1_new = x1
                    # y1_new = y1
                    # x2_new = x2
                    # y2_new = y2
                    
                    
                    normalized_bboxes.append([x1_new, y1_new, x2_new, y2_new])
                    isuse = parts[-1] if len(parts) > 5 else "1"  # 默认isuse=1
                    isuses.append(isuse)
    
    return normalized_bboxes, isuses

async def process_image_pair(llm_util, large_image_path, small_image_path, output_json_path):
    global all_results
    try:
        # 调用模型处理图片对
        model_output = await llm_util.call_doubao(
            prompt_double_things_item, 
            [
                llm_util.path2base64(large_image_path),
                llm_util.path2base64(small_image_path)
            ]
        )
        
        # 构建结果记录
        new_record = {
            "large_image": large_image_path,
            "small_image": small_image_path,
            "model_output": model_output
        }
        
    except Exception as e:
        print(f"Processing failed: {large_image_path} | {small_image_path} | Error: {str(e)}")
        new_record = {
            "large_image": large_image_path,
            "small_image": small_image_path,
            "error": str(e)
        }
    
    async with write_lock:
        all_results.append(new_record)
    await save_json(output_json_path)
    return new_record

async def process_image_pairs_concurrently(llm_util, large_image_folder, small_images_root_folder, output_json_path):
    batch_processor = BatchProcessUtil(batch_size=8)
    global processed_pairs_set
    
    # 获取所有全景图
    large_images = [f for f in os.listdir(large_image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # 遍历每个全景图对应的子文件夹
    for large_image in tqdm(large_images, desc="Processing large images"):
        large_image_path = os.path.join(large_image_folder, large_image)
        base_name = os.path.splitext(large_image)[0]
        
        # 找到对应的小图文件夹
        small_image_folder = os.path.join(small_images_root_folder, base_name)
        if not os.path.isdir(small_image_folder):
            print(f"Warning: Small image folder not found for {large_image}")
            continue
            
        # 获取小图文件夹中的所有小图
        small_images = [f for f in os.listdir(small_image_folder) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 处理每个小图和全景图的组合
        for small_image in small_images:
            small_image_path = os.path.join(small_image_folder, small_image)
            pair_key = f"{large_image_path}_{small_image_path}"
            
            if pair_key not in processed_pairs_set:
                batch_processor.add_task(
                    process_image_pair(llm_util, large_image_path, small_image_path, output_json_path)
                )
                processed_pairs_set.add(pair_key)
    
    # 处理批次
    if batch_processor.tasks:
        await batch_processor.process()
    
    # 最终保存
    await save_json(output_json_path)


async def process_image_with_boxes(llm_util, image_path, label_path, output_json_path):
    """处理图片和对应的标签文件"""
    batch_processor = BatchProcessUtil(batch_size=8)
    global processed_items_set
    
    # 获取图片尺寸
    try:
        with Image.open(image_path) as img:
            img_width, img_height = img.size
    except Exception as e:
        print(f"Failed to get image size {image_path}: {str(e)}")
        return
    
    # 解析标签文件(直接获取归一化后的坐标)
    boxes, isuses = parse_label_file(label_path, img_width, img_height)
    
    if not boxes:
        print(f"No class 0 boxes found in {label_path}")
        return
    
    # 读取图片base64
    try:
        image_base64 = llm_util.path2base64(image_path)
    except Exception as e:
        print(f"Failed to read image {image_path}: {str(e)}")
        return
    
    # 为每个坐标框创建处理任务
    for i, box in enumerate(boxes):
        if isuses[i]!='yes':
            continue
        item_key = f"{image_path}_{box}"
        if item_key not in processed_items_set:
            batch_processor.add_task(
                process_single_box(llm_util, image_path, image_base64, box, output_json_path)
            )
            processed_items_set.add(item_key)
    # 执行批处理
    if batch_processor.tasks:
        await batch_processor.process()
    
    # 保存结果
    await save_json(output_json_path)

async def process_single_box(llm_util, image_path, image_base64, box, output_json_path):
    """处理单个坐标框"""
    global all_results
     
    
    try:
        x1, y1, x2, y2 = box
        
        # 格式化坐标信息
        coords = f"{x1} {y1} {x2} {y2}"
        target_person = f"[{coords}]"
        # 构建提示词
        prompt_format = prompt_single.format(
            target_person=target_person
        )
        # print(prompt_format)
        # 调用模型处理
        model_output = await llm_util.call_doubao(
            prompt_format, 
            [llm_util.path2base64(image_path)]
        )
        # print(model_output)
        # 构建结果记录
        new_record = {
            "prompt_format":prompt_format,
            "large_image": image_path,
            "box": box,
            "model_output": model_output
        }
    except Exception as e:
        print(f"Processing failed: {image_path} | Box: {box} | Error: {str(e)}")
        new_record = {
            "large_image": image_path,
            "box": box,
            "error": str(e)
        }
    
    async with write_lock:
        all_results.append(new_record)
    
    return new_record









async def main():
    parser = argparse.ArgumentParser(description="处理全景图和小图对并生成分析结果")
    parser.add_argument("--large_image_folder", default="/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图", 
                        help="包含全景图的文件夹路径")
    parser.add_argument("--small_images_root_folder", default="/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/persons_cut", 
                        help="包含小图子文件夹的根目录路径")
    parser.add_argument("--label_folder", default="/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/annotated_txt_quality", help="标签文件夹路径")
    
    parser.add_argument("--output_json", default="/host/algorithm4/zhanglu/dataset/tmp/0609/筛选小图_depth_v3/persons_result_gemini_0625_附属物itemname.json",
                        help="输出JSON文件路径")
    args = parser.parse_args()

    global all_results, processed_pairs_set, processed_items_set
    
    # 检查文件夹是否存在
    if not os.path.isdir(args.large_image_folder):
        print(f"Error: Large image folder not found {args.large_image_folder}")
        return
    if not os.path.isdir(args.small_images_root_folder):
        print(f"Error: Small images root folder not found {args.small_images_root_folder}")
        return
    if not os.path.isdir(args.label_folder):
        print(f"Error: Label folder not found {args.label_folder}")
        return    
    
    # 加载已有输出结果（用于断点续处理）
    if os.path.exists(args.output_json):
        try:
            with open(args.output_json, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
            processed_pairs_set = {f"{item['large_image']}_{item['small_image']}" 
                                 for item in all_results if 'large_image' in item and 'small_image' in item}
            print(f"✅ Loaded existing results: {len(all_results)} records")
        except Exception as e:
            print(f"⚠️ Failed to load existing results: {e}")

    # 初始化模型工具
    llm_util = LlmUtil(sys_content="You are a multimodal AI analysis expert.")

    # 处理图片对
    await process_image_pairs_concurrently(
        llm_util, 
        args.large_image_folder,
        args.small_images_root_folder,
        args.output_json
    )
    print(f"Processing completed. Total {len(all_results)} records saved to {args.output_json}")


    # 处理单张图
    # for image_file in os.listdir(args.large_image_folder):
    #     if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
    #         continue
            
    #     image_path = os.path.join(args.large_image_folder, image_file)
    #     base_name = os.path.splitext(image_file)[0]
    #     label_path = os.path.join(args.label_folder, f"{base_name}.txt")
        
    #     if not os.path.exists(label_path):
    #         print(f"Warning: Label file not found for {image_file}")
    #         continue
            
    #     # 处理图片和标签
    #     await process_image_with_boxes(
    #         llm_util,
    #         image_path,
    #         label_path,
    #         args.output_json
    #     )

    print(f"Processing completed. Total {len(all_results)} records saved to {args.output_json}")





if __name__ == '__main__':
    asyncio.run(main())