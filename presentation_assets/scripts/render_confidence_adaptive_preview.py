from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PPTX_PATH = OUT_DIR / "confidence_adaptive_dual_path_distillation_editable.pptx"
PREVIEW_PATH = OUT_DIR / "confidence_adaptive_dual_path_distillation_preview.png"
QA_PATH = OUT_DIR / "confidence_adaptive_dual_path_distillation_qa.json"

SLIDE_W = 10.0
SLIDE_H = 7.5
SCALE = 160

BLUE = "#0017D5"
DARK_BLUE = "#071075"
SKY = "#CAE6FF"
GREEN = "#15A900"
GREEN_LIGHT = "#C7F7B8"
PURPLE = "#7C35B8"
RED = "#FF2B18"
BLACK = "#111111"
WHITE = "#FFFFFF"
GRAY = "#6B7280"


def px(v: float) -> int:
    return round(v * SCALE)


def pxy(x: float, y: float) -> tuple[int, int]:
    return px(x), px(y)


def rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def mix(c1: str, c2: str, t: float) -> str:
    a = rgb(c1)
    b = rgb(c2)
    out = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return "#%02X%02X%02X" % out


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/ARIALBD.TTF" if bold else "C:/Windows/Fonts/ARIAL.TTF"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def text(
    draw: ImageDraw.ImageDraw,
    label: str,
    x: float,
    y: float,
    size: int,
    color: str = BLACK,
    bold: bool = False,
    anchor: str = "la",
    align: str = "left",
) -> None:
    draw.multiline_text(pxy(x, y), label, fill=color, font=font(size, bold), anchor=anchor, align=align, spacing=2)


def arrow(draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float, color: str = BLACK, width: int = 2) -> None:
    p1 = pxy(x1, y1)
    p2 = pxy(x2, y2)
    draw.line([p1, p2], fill=color, width=width)
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    length = max(8, width * 5)
    spread = 0.45
    pts = [
        p2,
        (round(p2[0] - length * math.cos(ang - spread)), round(p2[1] - length * math.sin(ang - spread))),
        (round(p2[0] - length * math.cos(ang + spread)), round(p2[1] - length * math.sin(ang + spread))),
    ]
    draw.polygon(pts, fill=color)


def dot(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, r: float = 0.035) -> None:
    draw.ellipse([px(x - r), px(y - r), px(x + r), px(y + r)], fill=color, outline=color)


def rounded(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, outline: str, width: int = 2, radius: float = 0.07) -> None:
    draw.rounded_rectangle([px(x), px(y), px(x + w), px(y + h)], radius=px(radius), outline=outline, width=width, fill=None)


def rect(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, fill: str | None, outline: str = BLACK, width: int = 1) -> None:
    draw.rectangle([px(x), px(y), px(x + w), px(y + h)], fill=fill, outline=outline, width=width)


def matrix(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, values: list[list[float]], low: str, high: str) -> None:
    rows = len(values)
    cols = len(values[0])
    for r, row in enumerate(values):
        for c, v in enumerate(row):
            rect(draw, x + c * w / cols, y + r * h / rows, w / cols, h / rows, mix(low, high, v), "#8291AA", 1)
    rect(draw, x, y, w, h, None, "#65708A", 2)


def tensor(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, dx: float, dy: float, base_light: str, base_dark: str) -> None:
    top = [pxy(x, y), pxy(x + dx, y - dy), pxy(x + w + dx, y - dy), pxy(x + w, y)]
    side = [pxy(x + w, y), pxy(x + w + dx, y - dy), pxy(x + w + dx, y + h - dy), pxy(x + w, y + h)]
    front = [pxy(x, y), pxy(x + w, y), pxy(x + w, y + h), pxy(x, y + h)]
    draw.polygon(top, fill=mix(base_light, WHITE, 0.35), outline="#26324B")
    draw.polygon(side, fill=mix(base_dark, base_light, 0.58), outline="#26324B")
    draw.polygon(front, fill=mix(base_light, WHITE, 0.08), outline="#26324B")
    for i in range(1, 5):
        gx = x + w * i / 5
        gy = y + h * i / 5
        draw.line([pxy(gx, y), pxy(gx, y + h)], fill="#26324B", width=1)
        draw.line([pxy(x, gy), pxy(x + w, gy)], fill="#26324B", width=1)
        draw.line([pxy(gx, y), pxy(gx + dx, y - dy)], fill="#26324B", width=1)
        draw.line([pxy(x + w, gy), pxy(x + w + dx, gy - dy)], fill="#26324B", width=1)
    for i in range(1, 4):
        t = i / 4
        draw.line([pxy(x + dx * t, y - dy * t), pxy(x + w + dx * t, y - dy * t)], fill="#26324B", width=1)
        draw.line([pxy(x + w + dx * t, y - dy * t), pxy(x + w + dx * t, y + h - dy * t)], fill="#26324B", width=1)


def value_plot(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float) -> None:
    text(draw, "6a. Absolute-Value Distillation", x + 0.16, y + 0.09, 13, BLUE, True)
    text(draw, "(Value Matching)", x + 0.46, y + 0.31, 12, BLUE, True)
    px0, py0 = x + 0.36, y + 0.70
    arrow(draw, px0, py0 + 0.80, px0, py0 + 0.05, BLACK, 2)
    arrow(draw, px0, py0 + 0.80, px0 + w - 0.55, py0 + 0.80, BLACK, 2)
    xs = [px0 + d for d in [0.18, 0.40, 0.62, 0.84, 1.06]]
    ty, sy = py0 + 0.32, py0 + 0.66
    draw.line([pxy(xs[0], ty), pxy(xs[-1], ty)], fill=BLUE, width=3)
    draw.line([pxy(xs[0], sy), pxy(xs[-1], sy)], fill=GREEN, width=3)
    for xx in xs:
        draw.line([pxy(xx, ty), pxy(xx, sy)], fill=BLACK, width=1)
        dot(draw, xx, ty, BLUE, 0.03)
        dot(draw, xx, sy, GREEN, 0.03)
    text(draw, "Weighted by confidence (c)", x + 0.31, y + h - 0.32, 13, BLUE, True)


def trend_plot(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float) -> None:
    text(draw, "6b. Trend Distillation", x + 0.25, y + 0.10, 13, RED, True)
    text(draw, "(Trend Matching)", x + 0.43, y + 0.32, 12, RED, True)
    px0, py0 = x + 0.36, y + 0.66
    ph = 0.82
    pw = w - 0.55
    arrow(draw, px0, py0 + ph, px0, py0 + 0.05, BLACK, 2)
    arrow(draw, px0, py0 + ph, px0 + pw, py0 + ph, BLACK, 2)
    teacher = [0.42, 0.30, 0.16, 0.40, 0.32, 0.50, 0.62, 0.43, 0.49, 0.64]
    student = [0.78, 0.66, 0.52, 0.70, 0.82, 0.75, 0.66, 0.77, 0.71, 0.65]
    for vals, color in [(teacher, BLUE), (student, RED)]:
        pts = [(px0 + 0.12 + i * (pw - 0.25) / (len(vals) - 1), py0 + v * ph) for i, v in enumerate(vals)]
        draw.line([pxy(a, b) for a, b in pts], fill=color, width=3)
        for a, b in pts:
            dot(draw, a, b, color, 0.02)
    text(draw, "Weighted by complementary\nconfidence", x + 0.43, y + h - 0.46, 12, RED, True, align="center")


def render_preview() -> None:
    img = Image.new("RGB", (px(SLIDE_W), px(SLIDE_H)), WHITE)
    draw = ImageDraw.Draw(img)

    draw.line([pxy(0.45, 0.76), pxy(2.30, 0.76)], fill=BLUE, width=3)
    dot(draw, 0.45, 0.76, BLUE, 0.04)
    draw.line([pxy(7.78, 0.76), pxy(9.35, 0.76)], fill=BLUE, width=3)
    dot(draw, 9.35, 0.76, BLUE, 0.04)
    text(draw, "Confidence-Adaptive Dual-Path Distillation", 5.0, 0.52, 38, DARK_BLUE, True, anchor="ma")

    text(draw, "1. Teacher\nPrediction", 0.78, 1.83, 15, DARK_BLUE, True, anchor="ma", align="center")
    tensor(draw, 0.52, 2.55, 0.72, 1.15, 0.20, 0.18, SKY, "#74B9F7")
    text(draw, "2. Ground\nTruth", 1.82, 3.55, 15, DARK_BLUE, True, anchor="ma", align="center")
    tensor(draw, 1.50, 4.22, 0.62, 1.04, 0.18, 0.16, GREEN_LIGHT, "#5AD44B")

    text(draw, "3. Teacher\nPrediction Error", 3.32, 1.82, 15, DARK_BLUE, True, anchor="ma", align="center")
    matrix(draw, 2.86, 2.76, 0.87, 1.30, [
        [0.82, 0.30, 0.62, 0.16, 0.24],
        [0.42, 0.78, 0.22, 0.55, 0.12],
        [0.18, 0.30, 0.62, 0.70, 0.28],
        [0.30, 0.24, 0.18, 0.34, 0.75],
        [0.16, 0.40, 0.22, 0.26, 0.82],
    ], "#F4E8FF", "#5E2699")
    text(draw, "|Yhat_t - Y|", 3.28, 4.45, 17, BLACK, False, anchor="ma")
    arrow(draw, 1.35, 2.82, 2.58, 3.05, BLACK, 4)
    arrow(draw, 2.17, 4.55, 2.72, 3.72, BLACK, 4)

    rounded(draw, 4.05, 1.95, 1.27, 3.02, "#854BFF", 2)
    text(draw, "4. Error Decomposition", 4.68, 2.10, 13, BLUE, True, anchor="ma")
    text(draw, "Node-level Error", 4.67, 2.48, 12, BLUE, False, anchor="ma")
    matrix(draw, 4.37, 2.72, 0.15, 0.68, [[0.25], [0.82], [0.65], [0.18]], "#F4E8FF", "#5E2699")
    text(draw, "masked\nmean", 4.98, 2.98, 10, BLACK, False, anchor="ma", align="center")
    arrow(draw, 4.71, 3.10, 4.90, 3.10, BLACK, 2)
    draw.line([pxy(4.15, 3.80), pxy(5.22, 3.80)], fill="#854BFF", width=2)
    text(draw, "Horizon-level Error", 4.66, 4.10, 12, BLUE, False, anchor="ma")
    matrix(draw, 4.18, 4.43, 0.56, 0.15, [[0.18, 0.35, 0.68, 0.86, 0.50]], "#F4E8FF", "#5E2699")
    arrow(draw, 3.80, 3.36, 4.02, 3.36, BLACK, 3)

    text(draw, "5. Confidence\nMap", 5.98, 1.84, 15, DARK_BLUE, True, anchor="ma", align="center")
    matrix(draw, 5.55, 2.95, 0.62, 1.10, [
        [0.42, 0.70, 0.35, 0.82],
        [0.18, 0.66, 0.32, 0.50],
        [0.78, 0.26, 0.62, 0.42],
        [0.54, 0.36, 0.68, 0.30],
    ], "#D9F0FF", "#0060CC")
    for i in range(8):
        rect(draw, 6.28, 3.03 + i * 0.115, 0.08, 0.115, mix("#EEF6FF", "#0059C8", 1 - i / 7), mix("#EEF6FF", "#0059C8", 1 - i / 7), 1)
    text(draw, "high", 6.36, 2.81, 11, BLACK, False, anchor="ma")
    text(draw, "low", 6.36, 4.03, 11, BLACK, False, anchor="ma")
    arrow(draw, 5.34, 3.36, 5.52, 3.36, BLACK, 3)

    rounded(draw, 6.60, 1.36, 1.70, 2.20, BLUE, 2)
    value_plot(draw, 6.60, 1.36, 1.70, 2.20)
    rounded(draw, 6.60, 3.80, 1.70, 2.18, RED, 2)
    trend_plot(draw, 6.60, 3.80, 1.70, 2.18)

    draw.line([pxy(6.50, 2.45), pxy(6.50, 4.75)], fill=BLACK, width=2)
    arrow(draw, 6.32, 3.42, 6.50, 3.42, BLACK, 2)
    arrow(draw, 6.50, 2.45, 6.60, 2.45, BLACK, 2)
    arrow(draw, 6.50, 4.75, 6.60, 4.75, BLACK, 2)
    draw.line([pxy(8.38, 2.45), pxy(8.38, 4.75)], fill=BLACK, width=2)
    arrow(draw, 8.38, 3.55, 8.55, 3.55, BLACK, 2)
    rounded(draw, 8.55, 2.88, 1.00, 1.68, BLUE, 2)
    text(draw, "7. Distillation Loss", 9.05, 3.08, 13, BLUE, True, anchor="ma")
    dot(draw, 8.66, 3.56, BLUE, 0.018)
    text(draw, "Absolute-Value Loss\n(weighted by c)", 8.78, 3.49, 10, BLACK)
    dot(draw, 8.66, 4.08, BLUE, 0.018)
    text(draw, "Trend Loss\n(weighted by\ncomplementary\nconfidence)", 8.78, 4.00, 10, BLACK)

    rounded(draw, 0.40, 6.42, 9.15, 0.70, BLUE, 2)
    draw.ellipse([px(0.67), px(6.53), px(1.10), px(6.96)], fill=BLUE, outline=BLUE)
    draw.line([pxy(0.77, 6.75), pxy(0.86, 6.85), pxy(1.00, 6.62)], fill=WHITE, width=5)
    text(draw, "Soft routing: both paths are active for every node-horizon position.", 1.18, 6.56, 18, BLUE, True)
    text(draw, "High confidence emphasizes value matching; low confidence emphasizes trend consistency.", 1.18, 6.84, 16, BLUE)

    img.save(PREVIEW_PATH)


def inspect_pptx() -> dict:
    with zipfile.ZipFile(PPTX_PATH) as zf:
        names = zf.namelist()
        slide_xml = zf.read("ppt/slides/slide1.xml").decode("utf-8")
    media_files = [name for name in names if name.startswith("ppt/media/") and not name.endswith("/")]
    qa = {
        "pptx": str(PPTX_PATH),
        "preview": str(PREVIEW_PATH),
        "media_files": media_files,
        "picture_elements": slide_xml.count("<p:pic>"),
        "shape_elements": slide_xml.count("<p:sp>"),
        "text_runs": slide_xml.count("<a:t>"),
        "custom_geometry_elements": slide_xml.count("<a:custGeom>"),
        "slide_number_placeholder": "sldNum" in slide_xml or "Slide Number" in slide_xml,
        "is_editable_rebuild": not media_files and slide_xml.count("<p:sp>") > 100 and slide_xml.count("<a:t>") > 30,
    }
    QA_PATH.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return qa


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    render_preview()
    qa = inspect_pptx()
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
