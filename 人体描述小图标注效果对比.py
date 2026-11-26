#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比两个目录下同名小图 JSON（每个 large 图下有若干 small 图）：
- 通过 --large_dir / --small_dir 提供图片路径；
- large_dir 是一堆大图；
- small_dir 是一堆文件夹，每个文件夹名称对应一个大图，文件夹内是小图；
- --dir1 和 --dir2 的 JSON 存储结构与 small_dir 相同；
- 遍历 large 图，找到对应的 small 图文件夹；
- 对比 --dir1 与 --dir2 中同名 small 图的 json，完整展示并集，差异红色；
- “可能的身份”部分会强制用红色字体高亮；
- 若 observedAttributes.carriedItems/heldObjects[*].grounding 形如 "<x_min,y_min,x_max,y_max>"（相对 1000×1000），在大图上叠加 dir1=绿、dir2=红；
- 输出一张对比图到 --out 目录，保持与 small_dir 相同的子文件夹结构。
"""

import os
import json
import math
import argparse
from typing import Any, Dict, Tuple, List, Union

from PIL import Image, ImageDraw, ImageFont

# ================== 可调样式 ==================
COLUMN_WIDTH = 560
IMAGE_RESIZE_WIDTH = 480
MARGIN = 20
PADDING = 12
LINE_SPACING = 6
INDENT_WIDTH = 22
BG_COLOR = (255, 255, 255)

COLOR_SAME = (0, 0, 0)
COLOR_DIFF = (220, 0, 0)
COLOR_HEADER = (40, 40, 40)
COLOR_SECTION = (0, 120, 0)
COLOR_LABEL = (70, 70, 70)
COLOR_DIR1_BOX = (0, 180, 0)     # dir1 grounding 颜色
COLOR_DIR2_BOX = (220, 0, 0)     # dir2 grounding 颜色
BOX_THICKNESS = 2

VALUE_MISSING_PLACEHOLDER = "null"
VIRTUAL_CANVAS_SIZE = 1000.0
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# ================== 术语映射（适配新 JSON） ==================
TEMPLATE_MAP = {
    # 元信息
    "imageAnalysisMetadata": "图像分析元信息",
    "visibility": "可见性",
    "occlusion": "遮挡情况",

    # 摘要
    "summary": "摘要信息",
    "short_description": "一句话摘要",
    "long_description": "详细描述",

    # 可观察属性
    "observedAttributes": "可观察属性",
    "physicalAppearance": "外貌特征",
    "skin_tone_suggestion": "肤色建议",
    "bodyComposition": "身体构成",
    "hair": "发型",
    "length": "长度",
    "style": "样式",
    "color": "颜色",

    "apparel": "穿着服饰",
    "upperBody": "上身",
    "lowerBody": "下身",
    "footwear": "鞋履",
    "type": "类型",
    "pattern": "图案",
    "material": "材质",
    "details": "细节",
    "brand_or_logo": "品牌/Logo",

    "accessories": "穿戴配饰",
    "carriedItems": "携带物品",
    "heldObjects": "手持物品",
    "item_name": "物品名称",
    "grounding": "定位框",

    "postureAndOrientation": "姿态与朝向",
    "body_orientation": "身体朝向",
    "posture": "姿态",
    "gestures": "手势/姿势",

    # 推断属性
    "inferredAttributes": "推断属性",
    "demographics": "人口统计学特征",
    "estimated_gender": "推断性别",
    "estimated_age_group": "推断年龄段",
    "possible_identity": "可能的身份",
    "value": "值",
    "confidence": "置信度",

    "emotion_and_expression": "情绪与表情",
    "facial_expression_observed": "观察到的面部表情",
    "inferred_emotion": "推断的情绪",
    "inferred_mood_from_posture": "从姿态推断的心情",

    "contextualAnalysis": "情景分析",
    "actions": "行为",
    "special_behaviors": "特殊行为",
    "relationships": "关系",
    "with_people": "与人的互动",
    "with_vehicles": "与车辆的互动",
    "with_objects": "与物体的互动",
    "object": "物体",
    "interaction": "交互",

    "estimated_location_type": "推断的地点类型",
    "estimated_purpose_or_goal": "推断的目的或目标",
}

# ================== 字体 ==================
def load_fonts(font_path: str):
    try:
        font = ImageFont.truetype(font_path, 16)
        header_font = ImageFont.truetype(font_path, 20)
        section_font = ImageFont.truetype(font_path, 18)
        image_label_font = ImageFont.truetype(font_path, 14)
    except IOError:
        print(f"警告: 字体 {font_path} 未找到，使用默认字体（可能不支持中文）。")
        font = ImageFont.load_default()
        header_font = ImageFont.load_default()
        section_font = ImageFont.load_default()
        image_label_font = ImageFont.load_default()
    return font, header_font, section_font, image_label_font

# ================== 文本工具 ==================
def line_height(font: ImageFont.FreeTypeFont) -> int:
    try:
        b = font.getbbox("字")
        return max(1, b[3] - b[1])
    except Exception:
        return int(font.size * 1.2)

def wrap_text(text: Any, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    if text is None:
        s = VALUE_MISSING_PLACEHOLDER
    else:
        s = str(text)
    lines = []
    for para in s.split("\n"):
        if para == "":
            lines.append("")
            continue
        cur = ""
        for ch in para:
            t = cur + ch
            if font.getlength(t) <= max_width:
                cur = t
            else:
                lines.append(cur)
                cur = ch
        lines.append(cur)
    return lines or [""]

# ================== JSON 加载 ==================
def load_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            s = f.read().strip()
        if s.startswith("```json"):
            s = s[7:]
            if s.endswith("```"):
                s = s[:-3]
            s = s.strip()
        s = s.replace("None", "null").replace("True", "true").replace("False", "false")
        return json.loads(s)
    except Exception as e:
        print(f"错误: 解析 JSON 失败 {path}: {e}")
        return {}

# ================== grounding 绘制 ==================
def draw_groundings(img: Image.Image, data: Dict[str, Any], orig_w: int, orig_h: int, color: Tuple[int, int, int]):
    if not isinstance(data, dict) or orig_w == 0 or orig_h == 0 or VIRTUAL_CANVAS_SIZE == 0:
        return
    draw = ImageDraw.Draw(img)
    disp_w, disp_h = img.size

    items: List[str] = []
    obs = data.get("observedAttributes", {})
    for key in ("carriedItems", "heldObjects"):
        arr = obs.get(key)
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    g = it.get("grounding")
                    if isinstance(g, str):
                        items.append(g)

    for gstr in items:
        try:
            vals = [float(x.strip()) for x in gstr.strip("<>").split(",")]
            if len(vals) != 4:
                continue
            x_min_1000, y_min_1000, x_max_1000, y_max_1000 = vals
            ax1 = int(x_min_1000 / VIRTUAL_CANVAS_SIZE * orig_w)
            ay1 = int(y_min_1000 / VIRTUAL_CANVAS_SIZE * orig_h)
            ax2 = int(x_max_1000 / VIRTUAL_CANVAS_SIZE * orig_w)
            ay2 = int(y_max_1000 / VIRTUAL_CANVAS_SIZE * orig_h)

            sx = disp_w / max(1, orig_w)
            sy = disp_h / max(1, orig_h)
            x1 = max(0, min(int(ax1 * sx), disp_w - 1))
            y1 = max(0, min(int(ay1 * sy), disp_h - 1))
            x2 = max(0, min(int(ax2 * sx), disp_w - 1))
            y2 = max(0, min(int(ay2 * sy), disp_h - 1))
            if x1 < x2 and y1 < y2:
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=BOX_THICKNESS)
        except Exception:
            continue

# ================== 深度并集对比绘制 ==================
JSONLike = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

def pretty_key(k: Any) -> str:
    return TEMPLATE_MAP.get(str(k), str(k))

def draw_union_recursive(
    draw: ImageDraw.ImageDraw,
    left: JSONLike, right: JSONLike,
    x: int, y: int, indent_level: int,
    col_start_x: int,
    font: ImageFont.FreeTypeFont,
    force_red: bool = False
) -> int:
    """
    并集键递归展示。
    - 本列与对列不同处用红色。
    - force_red=True时，强制使用红色字体，用于高亮特定部分（如“可能的身份”）。
    """
    base_indent = "│  " * max(0, indent_level - 1)
    lh = line_height(font)

    if isinstance(left, dict) or isinstance(right, dict):
        ld = left if isinstance(left, dict) else {}
        rd = right if isinstance(right, dict) else {}
        keys = sorted(set(ld.keys()) | set(rd.keys()), key=lambda k: str(k))
        for k in keys:
            lv = ld.get(k)
            rv = rd.get(k)
            
            # 检查是否是需要高亮的 'possible_identity' 部分
            is_identity_section = force_red or (str(k) == "possible_identity")
            
            label_color = COLOR_DIFF if is_identity_section else COLOR_LABEL
            label = f"{base_indent}└─ {pretty_key(k)}:"
            draw.text((x, y), label, font=font, fill=label_color)

            if isinstance(lv, (dict, list)) or isinstance(rv, (dict, list)):
                y += lh + LINE_SPACING
                y = draw_union_recursive(draw, lv, rv, x, y, indent_level + 1, col_start_x, font, force_red=is_identity_section)
            else:
                my_val = lv
                other_val = rv
                
                # 决定值的颜色
                if is_identity_section:
                    color = COLOR_DIFF
                else:
                    color = COLOR_DIFF if (my_val != other_val) else COLOR_SAME
                
                text = VALUE_MISSING_PLACEHOLDER if my_val is None else str(my_val)
                vx = x + font.getlength(label) + PADDING // 2
                maxw = COLUMN_WIDTH - (x - col_start_x) - (font.getlength(label) + PADDING) - 2
                for i, line in enumerate(wrap_text(text, font, int(maxw))):
                    if i > 0:
                        y += lh + LINE_SPACING
                    draw.text((vx, y), line, font=font, fill=color)
                y += lh + LINE_SPACING
        return y

    if isinstance(left, list) or isinstance(right, list):
        ll = left if isinstance(left, list) else []
        rl = right if isinstance(right, list) else []
        m = max(len(ll), len(rl))
        if m == 0:
            color = COLOR_DIFF if force_red else COLOR_SAME
            draw.text((x, y), f"{base_indent}└─ []", font=font, fill=color)
            return y + lh + LINE_SPACING
        for i in range(m):
            lv = ll[i] if i < len(ll) else None
            rv = rl[i] if i < len(rl) else None
            if isinstance(lv, (dict, list)) or isinstance(rv, (dict, list)):
                # 传递 force_red 标志
                y = draw_union_recursive(draw, lv, rv, x, y, indent_level, col_start_x, font, force_red=force_red)
            else:
                my_val = lv
                other_val = rv

                # 决定值的颜色
                if force_red:
                    color = COLOR_DIFF
                else:
                    color = COLOR_DIFF if (my_val != other_val) else COLOR_SAME
                
                text = VALUE_MISSING_PLACEHOLDER if my_val is None else str(my_val)
                for j, line in enumerate(wrap_text(f"{base_indent}└─ {text}", font, COLUMN_WIDTH - (x - col_start_x) - 2)):
                    if j > 0:
                        y += lh + LINE_SPACING
                    draw.text((x, y), line, font=font, fill=color)
                y += lh + LINE_SPACING
        return y

    # 处理顶层就是基本类型的情况
    my_val = left
    other_val = right
    
    # 决定值的颜色
    if force_red:
        color = COLOR_DIFF
    else:
        color = COLOR_DIFF if (my_val != other_val) else COLOR_SAME
        
    text = VALUE_MISSING_PLACEHOLDER if my_val is None else str(my_val)
    for i, line in enumerate(wrap_text(f"{base_indent}└─ {text}", font, COLUMN_WIDTH - (x - col_start_x) - 2)):
        if i > 0:
            y += lh + LINE_SPACING
        draw.text((x, y), line, font=font, fill=color)
    y += lh + LINE_SPACING
    return y


def draw_json_two_columns(
    draw: ImageDraw.ImageDraw,
    left_data: Dict[str, Any], right_data: Dict[str, Any],
    x_left: int, x_right: int, y_start: int,
    section_font: ImageFont.FreeTypeFont, font: ImageFont.FreeTypeFont
) -> Tuple[int, int]:
    """顶层模块按优先级，其他键跟随（并集）"""
    lh_sec = line_height(section_font)
    pri = ["imageAnalysisMetadata", "summary", "observedAttributes", "inferredAttributes"]
    keys_union = list(dict.fromkeys(pri + sorted(set(left_data.keys()) | set(right_data.keys()))))
    y_l, y_r = y_start, y_start
    for k in keys_union:
        lv = left_data.get(k)
        rv = right_data.get(k)
        title = f"├─ {TEMPLATE_MAP.get(k, k)}"
        # 左列
        draw.text((x_left, y_l), title, font=section_font, fill=COLOR_SECTION)
        y_l += lh_sec + LINE_SPACING
        y_l = draw_union_recursive(draw, lv, rv, x_left + INDENT_WIDTH, y_l, 1, x_left, font)
        y_l += PADDING
        # 右列
        draw.text((x_right, y_r), title, font=section_font, fill=COLOR_SECTION)
        y_r += lh_sec + LINE_SPACING
        y_r = draw_union_recursive(draw, rv, lv, x_right + INDENT_WIDTH, y_r, 1, x_right, font)
        y_r += PADDING
    return y_l, y_r

# ================== 生成对比图 ==================
def make_compare_canvas(
    json1_path: str, json2_path: str,
    large_img_path: str, small_img_path: str,
    out_dir: str,
    fonts: Tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]
):
    base_small = os.path.splitext(os.path.basename(small_img_path))[0]
    font, header_font, section_font, image_label_font = fonts

    data1 = load_json(json1_path)
    data2 = load_json(json2_path)

    # 加载显示用大图
    def load_and_resize(path: str) -> Tuple[Image.Image, int, int]:
        try:
            im = Image.open(path)
            w, h = im.size
            if w <= 0:
                raise ValueError("invalid width")
            dw = IMAGE_RESIZE_WIDTH
            dh = int(h / w * dw)
            return im.resize((dw, dh), Image.LANCZOS), w, h
        except Exception:
            ph = Image.new("RGB", (IMAGE_RESIZE_WIDTH, int(IMAGE_RESIZE_WIDTH * 0.6)), (230, 230, 230))
            d = ImageDraw.Draw(ph)
            d.text((10, 10), f"加载失败:\n{os.path.basename(path)}", font=font, fill=(90, 90, 90))
            return ph, ph.width, ph.height

    large_disp, orig_w, orig_h = load_and_resize(large_img_path)
    # 叠加两侧 grounding
    draw_groundings(large_disp, data1, orig_w, orig_h, COLOR_DIR1_BOX)
    draw_groundings(large_disp, data2, orig_w, orig_h, COLOR_DIR2_BOX)

    # small 图
    try:
        s_img = Image.open(small_img_path)
        sw, sh = s_img.size
        if sw > 0:
            s_img = s_img.resize((IMAGE_RESIZE_WIDTH, int(sh / sw * IMAGE_RESIZE_WIDTH)), Image.LANCZOS)
        else:
            raise ValueError("invalid small size")
    except Exception:
        s_img = Image.new("RGB", (IMAGE_RESIZE_WIDTH, int(IMAGE_RESIZE_WIDTH * 0.6)), (230, 230, 230))
        d = ImageDraw.Draw(s_img)
        d.text((10, 10), f"加载失败:\n{os.path.basename(small_img_path)}", font=font, fill=(90, 90, 90))

    # 画布
    canvas_w = MARGIN + IMAGE_RESIZE_WIDTH + MARGIN + 2 * COLUMN_WIDTH + PADDING + MARGIN
    canvas_h = 20000
    canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # 左侧图片区
    y = MARGIN
    draw.text((MARGIN, y), "大图（叠加框：左=绿 dir1，右=红 dir2）", font=image_label_font, fill=COLOR_LABEL)
    y += line_height(image_label_font) + 4
    canvas.paste(large_disp, (MARGIN, y))
    y += large_disp.height + 8
    draw.text((MARGIN, y), "裁剪图（small）", font=image_label_font, fill=COLOR_LABEL)
    y += line_height(image_label_font) + 4
    canvas.paste(s_img, (MARGIN, y))
    y += s_img.height + 8
    image_column_h = y + MARGIN

    # 右侧两列 JSON
    x_left = MARGIN + IMAGE_RESIZE_WIDTH + MARGIN
    x_right = x_left + COLUMN_WIDTH + PADDING
    left_title = f"目录1：{os.path.basename(os.path.dirname(json1_path))}/{os.path.basename(json1_path)}"
    right_title = f"目录2：{os.path.basename(os.path.dirname(json2_path))}/{os.path.basename(json2_path)}"
    draw.text((x_left, MARGIN), left_title, font=header_font, fill=COLOR_HEADER)
    draw.text((x_right, MARGIN), right_title, font=header_font, fill=COLOR_HEADER)
    y_start = MARGIN + line_height(header_font) + PADDING

    y_l, y_r = draw_json_two_columns(draw, data1, data2, x_left, x_right, y_start, section_font, font)

    final_h = max(image_column_h, y_l, y_r) + MARGIN
    canvas = canvas.crop((0, 0, canvas_w, final_h))

    # 确保输出子目录存在
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"{os.path.splitext(os.path.basename(large_img_path))[0]}__{base_small}_compare.jpg"
    out_path = os.path.join(out_dir, out_name)
    canvas.save(out_path, "JPEG", quality=95)
    print(f"  输出：{out_path}")

# ================== 小工具 ==================
def list_images_by_ext(folder: str, exts=IMG_EXTS) -> List[str]:
    """列出指定文件夹下所有符合扩展名的图片文件路径，并排序"""
    if not os.path.isdir(folder):
        return []
    return sorted([os.path.join(folder, f) for f in os.listdir(folder)
                   if os.path.splitext(f)[1].lower() in exts])

def find_small_images_for_large(small_dir: str, large_basename: str) -> List[str]:
    """
    在 small_dir 中查找名为 large_basename 的子文件夹，并返回其中所有图片的路径。
    """
    subfolder_path = os.path.join(small_dir, large_basename)
    if not os.path.isdir(subfolder_path):
        return []
    return list_images_by_ext(subfolder_path, IMG_EXTS)

# ================== 主逻辑 ==================
def main():
    ap = argparse.ArgumentParser(description="对比两个目录下同名小图 JSON（按 large 遍历）")
    ap.add_argument("--dir1", default="/host/ssddata/disk2/zhanglu/dataset/codetr/tmp_250909/特殊人群图片_codetr_results_tmp_cut_gemini_xiaotu", help="JSON 目录 1")
    ap.add_argument("--dir2", default="/host/ssddata/disk2/zhanglu/dataset/codetr/tmp_250909/特殊人群图片_codetr_results_tmp_cut_gemini_xiaotu", help="JSON 目录 2")
    ap.add_argument("--large_dir", default="/host/ssddata/disk2/zhanglu/dataset/codetr/tmp_250909/特殊人群图片", help="large 图目录")
    ap.add_argument("--small_dir", default="/host/ssddata/disk2/zhanglu/dataset/codetr/tmp_250909/特殊人群图片_codetr_results_tmp_cut", help="small 图目录（文件名与 json 一致，仅后缀不同）")
    ap.add_argument("--out", default="/host/ssddata/disk2/zhanglu/dataset/codetr/tmp_250909/特殊人群图片_codetr_results_tmp_cut_gemini_xiaotu_show", help="输出对比图目录")
    ap.add_argument("--font", default="/host/ssddata/disk2/zhanglu/SimHei.ttf", help="中文字体路径")
    ap.add_argument("--json_suffix", default=".json", help="json 后缀，默认 .json")
    args = ap.parse_args()

    # 检查字体文件是否存在
    if not os.path.exists(args.font):
        print(f"错误：字体文件 '{args.font}' 不存在。请使用 --font 参数指定有效路径。")
        # 尝试在常见系统路径查找
        common_paths = [
            "/System/Library/Fonts/STHeitiMedium.ttc",  # macOS
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", # Ubuntu
            "c:/Windows/Fonts/simhei.ttf", # Windows
        ]
        found = False
        for p in common_paths:
            if os.path.exists(p):
                print(f"提示：检测到可用字体 '{p}'，可尝试使用。")
                args.font = p
                found = True
                break
        if not found:
             print("未找到备用字体，程序将使用可能乱码的默认字体。")

    fonts = load_fonts(args.font)

    large_imgs = list_images_by_ext(args.large_dir, IMG_EXTS)
    if not large_imgs:
        print(f"未在 --large_dir '{args.large_dir}' 中发现图像。")
        return

    print(f"发现 large 图 {len(large_imgs)} 张，开始遍历……")
    for large_path in large_imgs:
        large_base = os.path.splitext(os.path.basename(large_path))[0]
        
        # 在小图目录中查找同名子文件夹下的小图
        small_imgs = find_small_images_for_large(args.small_dir, large_base)
        
        if not small_imgs:
            print(f"- {large_base}: 在 '{args.small_dir}' 中未找到名为 '{large_base}' 的子文件夹或其中没有图片，跳过。")
            continue

        print(f"- {large_base}: 找到 small 图 {len(small_imgs)} 张")
        
        # 确定输出子目录
        output_subfolder = os.path.join(args.out, large_base)
        
        for spath in small_imgs:
            small_name = os.path.basename(spath)
            small_base = os.path.splitext(small_name)[0]
            json_name = small_base + args.json_suffix
            
            # 构造 JSON 文件的完整路径，包含子文件夹
            j1 = os.path.join(args.dir1, large_base, json_name)
            j2 = os.path.join(args.dir2, large_base, json_name)

            if not os.path.exists(j1) and not os.path.exists(j2):
                print(f"  · {small_base}: 两个 json 都不存在，跳过。")
                continue

            # 缺失一侧也允许生成，显示为占位（差异标红）
            if not os.path.exists(j1):
                print(f"  · 警告：{os.path.join(large_base, json_name)} 不在 dir1 中，左列将为空。")
            if not os.path.exists(j2):
                print(f"  · 警告：{os.path.join(large_base, json_name)} 不在 dir2 中，右列将为空。")

            make_compare_canvas(
                j1 if os.path.exists(j1) else os.devnull,
                j2 if os.path.exists(j2) else os.devnull,
                large_path,
                spath,
                output_subfolder, # 传入特定的输出子目录
                fonts
            )

    print("全部完成！")

if __name__ == "__main__":
    main()

