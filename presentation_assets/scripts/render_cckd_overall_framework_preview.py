from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PPTX_PATH = OUT_DIR / "cckd_overall_framework_editable.pptx"
PREVIEW_PATH = OUT_DIR / "cckd_overall_framework_preview.png"
QA_PATH = OUT_DIR / "cckd_overall_framework_qa.json"

W, H = 13.333, 7.5
SCALE = 160

C = {
    "bg": "#FFFFFF",
    "paper": "#FBFCFF",
    "white": "#FFFFFF",
    "ink": "#111827",
    "muted": "#4B5563",
    "soft": "#CBD5E1",
    "grey": "#6B7280",
    "grey_soft": "#F3F4F6",
    "blue": "#0B56B3",
    "blue_deep": "#003D8F",
    "blue_soft": "#EFF6FF",
    "green": "#176B1E",
    "green_deep": "#0B5D12",
    "green_soft": "#F0FDF4",
    "violet": "#6D28D9",
    "violet_deep": "#3B168D",
    "violet_soft": "#F5F3FF",
    "orange": "#EA580C",
    "orange_soft": "#FFF7ED",
    "red": "#DC2626",
}


def px(v: float) -> int:
    return round(v * SCALE)


def pxy(x: float, y: float) -> tuple[int, int]:
    return px(x), px(y)


def rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def mix(c1: str, c2: str, t: float) -> str:
    a, b = rgb(c1), rgb(c2)
    out = tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))
    return "#%02X%02X%02X" % out


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/aptosdisplay-bold.ttf" if bold else "C:/Windows/Fonts/aptos.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def text(
    draw: ImageDraw.ImageDraw,
    value: str,
    x: float,
    y: float,
    size: int,
    color: str = "#111827",
    bold: bool = False,
    anchor: str = "la",
    align: str = "left",
) -> None:
    draw.multiline_text(pxy(x, y), value, fill=color, font=font(size, bold), anchor=anchor, align=align, spacing=3)


def box(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    w: float,
    h: float,
    stroke: str,
    fill: str = "#FFFFFF",
    width: int = 2,
    r: float = 0.07,
    dash: bool = False,
) -> None:
    if dash:
        draw.rounded_rectangle([px(x), px(y), px(x + w), px(y + h)], radius=px(r), fill=fill, outline=None)
        dashed_rect(draw, x, y, w, h, stroke, width)
    else:
        draw.rounded_rectangle([px(x), px(y), px(x + w), px(y + h)], radius=px(r), fill=fill, outline=stroke, width=width)


def dashed_rect(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, color: str, width: int = 2) -> None:
    segments = [(x, y, x + w, y), (x + w, y, x + w, y + h), (x + w, y + h, x, y + h), (x, y + h, x, y)]
    for x1, y1, x2, y2 in segments:
        dashed_line(draw, x1, y1, x2, y2, color, width)


def dashed_line(draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float, color: str, width: int = 2) -> None:
    p1, p2 = pxy(x1, y1), pxy(x2, y2)
    dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    steps = max(1, int(dist // 12))
    for i in range(steps):
        if i % 2 == 0:
            a, b = i / steps, min(1, (i + 1) / steps)
            draw.line(
                [
                    (p1[0] + (p2[0] - p1[0]) * a, p1[1] + (p2[1] - p1[1]) * a),
                    (p1[0] + (p2[0] - p1[0]) * b, p1[1] + (p2[1] - p1[1]) * b),
                ],
                fill=color,
                width=width,
            )


def arrow(draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float, color: str = "#111827", width: int = 2, dash: bool = False) -> None:
    if dash:
        dashed_line(draw, x1, y1, x2, y2, color, width)
    else:
        draw.line([pxy(x1, y1), pxy(x2, y2)], fill=color, width=width)
    p2 = pxy(x2, y2)
    ang = math.atan2(px(y2 - y1), px(x2 - x1))
    length = max(8, width * 5)
    spread = 0.46
    pts = [
        p2,
        (round(p2[0] - length * math.cos(ang - spread)), round(p2[1] - length * math.sin(ang - spread))),
        (round(p2[0] - length * math.cos(ang + spread)), round(p2[1] - length * math.sin(ang + spread))),
    ]
    draw.polygon(pts, fill=color)


def elbow(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]], color: str, width: int = 2, dash: bool = False) -> None:
    for i in range(len(pts) - 1):
        last = i == len(pts) - 2
        if last:
            arrow(draw, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], color, width, dash)
        elif dash:
            dashed_line(draw, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], color, width)
        else:
            draw.line([pxy(*pts[i]), pxy(*pts[i + 1])], fill=color, width=width)


def dot(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, r: float = 0.032) -> None:
    draw.ellipse([px(x - r), px(y - r), px(x + r), px(y + r)], fill=color, outline=color)


def tensor(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, dx: float, dy: float, tint: str = "blue") -> None:
    light = "#EAFBEF" if tint == "green" else "#EAF2FF"
    dark = "#86EFAC" if tint == "green" else "#93C5FD"
    draw.polygon([pxy(x, y), pxy(x + dx, y - dy), pxy(x + w + dx, y - dy), pxy(x + w, y)], fill=mix(light, "#FFFFFF", 0.2), outline="#334155")
    draw.polygon([pxy(x + w, y), pxy(x + w + dx, y - dy), pxy(x + w + dx, y + h - dy), pxy(x + w, y + h)], fill=mix(dark, light, 0.38), outline="#334155")
    draw.polygon([pxy(x, y), pxy(x + w, y), pxy(x + w, y + h), pxy(x, y + h)], fill=light, outline="#334155")
    for i in range(1, 5):
        gx, gy = x + w * i / 5, y + h * i / 5
        draw.line([pxy(gx, y), pxy(gx, y + h)], fill="#475569", width=1)
        draw.line([pxy(x, gy), pxy(x + w, gy)], fill="#475569", width=1)


def graph_icon(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, scale: float = 1.0) -> None:
    pts = [
        (x + 0.04 * scale, y + 0.22 * scale),
        (x + 0.20 * scale, y + 0.07 * scale),
        (x + 0.40 * scale, y + 0.20 * scale),
        (x + 0.56 * scale, y + 0.08 * scale),
        (x + 0.72 * scale, y + 0.26 * scale),
        (x + 0.50 * scale, y + 0.42 * scale),
        (x + 0.18 * scale, y + 0.42 * scale),
    ]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6), (6, 0), (1, 5), (2, 6)]:
        draw.line([pxy(*pts[a]), pxy(*pts[b])], fill=color, width=2)
    for px0, py0 in pts:
        dot(draw, px0, py0, color, 0.028 * scale)


def process_box(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, label: str, stroke: str) -> None:
    box(draw, x, y, w, h, stroke, "#FFFFFF", 2)
    text(draw, label, x + w / 2, y + 0.17, 12, "#111827", False, "ma", "center")


def stage_label(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, label: str, fill: str) -> None:
    draw.rounded_rectangle([px(x), px(y), px(x + w), px(y + 0.36)], radius=px(0.05), fill=fill, outline=fill)
    text(draw, label, x + 0.18, y + 0.08, 20, "#FFFFFF", True)


def confidence_card(draw: ImageDraw.ImageDraw, x: float, y: float) -> None:
    box(draw, x, y, 1.23, 2.62, C["violet"])
    text(draw, "Confidence\nEstimation", x + 0.62, y + 0.18, 16, C["violet_deep"], True, "ma", "center")
    text(draw, "teacher error\n\n->\n\ncontinuous\nconfidence c", x + 0.62, y + 0.97, 13, C["ink"], False, "ma", "center")


def dual_card(draw: ImageDraw.ImageDraw, x: float, y: float) -> None:
    box(draw, x, y, 1.50, 2.62, C["orange"])
    text(draw, "Dual-Path Soft\nDistillation", x + 0.75, y + 0.18, 16, C["orange"], True, "ma", "center")
    text(draw, "value path\nweighted by c", x + 0.75, y + 0.76, 12, C["ink"], False, "ma", "center")
    dot(draw, x + 0.20, y + 1.33, C["blue"], 0.035)
    arrow(draw, x + 0.24, y + 1.33, x + 1.34, y + 1.33, C["blue"], 3)
    text(draw, "1 - c", x + 0.80, y + 1.56, 13, C["ink"], False, "ma")
    dot(draw, x + 0.20, y + 1.76, C["orange"], 0.035)
    arrow(draw, x + 0.24, y + 1.76, x + 1.34, y + 1.76, C["orange"], 3)
    text(draw, "trend path\nweighted by 1 - c", x + 0.75, y + 1.96, 12, C["ink"], False, "ma", "center")
    text(draw, "both paths active", x + 0.75, y + 2.36, 11, C["muted"], False, "ma")


def curriculum_card(draw: ImageDraw.ImageDraw, x: float, y: float) -> None:
    box(draw, x, y, 1.70, 2.62, C["blue"])
    text(draw, "Adaptive Soft\nCurriculum", x + 0.85, y + 0.18, 16, C["blue_deep"], True, "ma", "center")
    text(draw, "horizon-wise\nsmooth weights", x + 0.85, y + 0.76, 12, C["ink"], False, "ma", "center")
    bars = [0.62, 0.52, 0.42, 0.34, 0.28, 0.21, 0.14, 0.08]
    for i, v in enumerate(bars):
        draw.rectangle([px(x + 0.18 + i * 0.17), px(y + 1.88 - v), px(x + 0.28 + i * 0.17), px(y + 1.88)], fill=mix("#DBEAFE", "#3B82F6", v), outline=C["blue"])
    draw.line([pxy(x + 0.14, y + 1.88), pxy(x + 1.44, y + 1.88)], fill=C["ink"], width=1)
    text(draw, "H1 H2 H3 ... H10 H11 H12", x + 0.85, y + 1.98, 9, C["ink"], False, "ma")
    arrow(draw, x + 1.10, y + 1.52, x + 1.47, y + 1.22, C["blue"], 2, True)
    text(draw, "all horizons active", x + 0.85, y + 2.36, 11, C["muted"], False, "ma")


def traffic_card(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, mini: bool = False) -> None:
    box(draw, x, y, w, h, C["grey"], "#FFFFFF", 2)
    text(draw, "Historical Traffic\nSequence X", x + w / 2, y + 0.18, 11 if mini else 14, C["ink"], True, "ma", "center")
    graph_icon(draw, x + (0.13 if mini else 0.36), y + (0.54 if mini else 0.74), C["blue"], 0.65 if mini else 0.75)
    tensor(draw, x + (0.74 if mini else 0.48), y + (0.55 if mini else 1.48), 0.36 if mini else 0.78, 0.42 if mini else 0.78, 0.12 if mini else 0.16, 0.10 if mini else 0.13)
    if not mini:
        arrow(draw, x + 0.25, y + 2.46, x + 1.28, y + 2.46, C["ink"], 2)
        arrow(draw, x + 0.20, y + 2.37, x + 0.20, y + 1.32, C["ink"], 2)
        text(draw, "Time Steps", x + 0.86, y + 2.55, 11, C["ink"], False, "ma")


def render() -> None:
    img = Image.new("RGB", (px(W), px(H)), C["bg"])
    d = ImageDraw.Draw(img)

    text(d, "CCKD Overall Framework", 6.67, 0.12, 40, "#000000", True, "ma")
    box(d, 0.10, 0.64, 13.12, 5.06, C["blue"], C["paper"], 2)
    stage_label(d, 0.10, 0.64, 1.52, "Training Stage", C["blue"])
    traffic_card(d, 0.30, 1.50, 1.40, 2.60)

    box(d, 2.50, 1.00, 2.62, 1.14, C["blue"])
    text(d, "GWNet Teacher", 3.81, 1.16, 18, C["blue_deep"], True, "ma")
    process_box(d, 2.64, 1.48, 1.14, 0.52, "Spatio-temporal\nblocks", C["blue"])
    process_box(d, 4.03, 1.48, 0.90, 0.52, "Prediction\nhead", C["blue"])
    box(d, 2.50, 3.16, 2.58, 1.12, C["green"])
    text(d, "Lightweight GCN Student", 3.79, 3.32, 17, C["green_deep"], True, "ma")
    process_box(d, 2.64, 3.61, 1.10, 0.52, "Lightweight\ngraph block", C["green"])
    process_box(d, 3.98, 3.61, 0.93, 0.52, "Prediction\nhead", C["green"])

    box(d, 5.60, 1.10, 0.86, 0.78, C["blue"], C["blue_soft"])
    text(d, "Teacher\nPrediction\nY_te", 6.03, 1.23, 12, C["blue_deep"], True, "ma", "center")
    box(d, 5.48, 2.28, 1.12, 0.64, C["grey"], C["grey_soft"])
    text(d, "Ground Truth\nY", 6.04, 2.43, 14, C["ink"], True, "ma", "center")
    box(d, 5.50, 3.42, 0.86, 0.78, C["green"], C["green_soft"])
    text(d, "Student\nPrediction\nY_st", 5.93, 3.55, 12, C["green_deep"], True, "ma", "center")
    box(d, 5.20, 4.50, 1.40, 0.74, C["grey"], C["grey_soft"])
    text(d, "Hard Supervision\nL_hard\n(e.g., MAE / MAPE)", 5.90, 4.62, 12, C["ink"], True, "ma", "center")

    box(d, 7.02, 0.90, 4.86, 3.30, C["blue"], "#FFFFFF", 2, dash=True)
    text(d, "CCKD Training Module", 9.45, 1.06, 23, C["blue_deep"], True, "ma")
    confidence_card(d, 7.12, 1.45)
    dual_card(d, 8.50, 1.45)
    curriculum_card(d, 10.10, 1.45)
    box(d, 12.12, 2.62, 1.05, 0.60, C["blue"])
    text(d, "Distillation Loss\nL_distill", 12.65, 2.78, 13, C["blue_deep"], True, "ma", "center")
    box(d, 9.25, 4.82, 1.50, 0.48, C["green"], C["green_soft"])
    text(d, "Total Loss L_total", 10.00, 4.99, 18, C["green_deep"], True, "ma")

    elbow(d, [(1.70, 2.50), (2.12, 2.50), (2.12, 1.57), (2.49, 1.57)], C["ink"], 3)
    elbow(d, [(1.70, 2.76), (2.12, 2.76), (2.12, 3.73), (2.49, 3.73)], C["ink"], 3)
    arrow(d, 5.12, 1.57, 5.58, 1.48, C["ink"], 3)
    arrow(d, 5.08, 3.73, 5.48, 3.78, C["ink"], 3)
    elbow(d, [(6.46, 1.49), (6.72, 1.49), (6.72, 1.78), (7.02, 1.78)], C["ink"], 3)
    arrow(d, 6.60, 2.60, 7.02, 2.58, C["ink"], 3)
    elbow(d, [(6.36, 3.80), (6.72, 3.80), (6.72, 2.86), (7.02, 2.86)], C["ink"], 3)
    arrow(d, 5.92, 2.92, 5.92, 3.42, C["ink"], 3)
    arrow(d, 5.92, 4.20, 5.92, 4.50, C["ink"], 3)
    elbow(d, [(6.60, 4.88), (8.96, 4.88), (8.96, 5.06), (9.25, 5.06)], C["ink"], 3)
    arrow(d, 11.88, 2.78, 12.12, 2.92, C["ink"], 3)
    elbow(d, [(12.65, 3.22), (12.65, 5.06), (10.75, 5.06)], C["ink"], 3)
    elbow(d, [(9.85, 5.30), (9.85, 5.48), (2.30, 5.48), (2.30, 4.15), (2.50, 4.15)], C["green_deep"], 2, True)
    text(d, "optimize student\n(backpropagation)", 3.74, 4.94, 13, C["green_deep"], False, "ma", "center")

    box(d, 0.10, 5.88, 13.12, 1.46, "#A3A3A3", "#F8FAFC", 2)
    stage_label(d, 0.10, 5.88, 1.52, "Inference Stage", "#555555")
    traffic_card(d, 2.32, 5.98, 1.40, 1.18, True)
    box(d, 4.48, 5.98, 2.38, 1.18, C["green"], C["green_soft"])
    text(d, "Trained Lightweight\nGCN Student", 5.67, 6.08, 15, C["green_deep"], True, "ma", "center")
    process_box(d, 4.62, 6.50, 1.00, 0.44, "Lightweight\ngraph block", C["green"])
    process_box(d, 5.85, 6.50, 0.92, 0.44, "Prediction\nhead", C["green"])
    box(d, 7.62, 6.08, 1.38, 1.00, C["grey"])
    text(d, "Forecast", 8.31, 6.20, 14, C["ink"], True, "ma")
    graph_icon(d, 7.76, 6.42, C["green_deep"], 0.86)
    box(d, 9.85, 6.27, 2.10, 0.66, C["grey"], "#FFFFFF", 1, dash=True)
    text(d, "Only the trained student is\nretained for inference.", 10.90, 6.48, 16, C["muted"], False, "ma", "center")
    arrow(d, 3.72, 6.58, 4.48, 6.58, C["ink"], 3)
    arrow(d, 6.86, 6.58, 7.62, 6.58, C["ink"], 3)

    img.save(PREVIEW_PATH)


def inspect_pptx() -> dict:
    with zipfile.ZipFile(PPTX_PATH) as zf:
        names = zf.namelist()
        slide_xml = zf.read("ppt/slides/slide1.xml").decode("utf-8")
    media_files = [name for name in names if name.startswith("ppt/media/") and not name.endswith("/")]
    qa = {
        "source_image_note": "attached image; original local path contains non-ASCII characters",
        "pptx": str(PPTX_PATH),
        "preview": str(PREVIEW_PATH),
        "media_files": media_files,
        "picture_elements": slide_xml.count("<p:pic>"),
        "shape_elements": slide_xml.count("<p:sp>"),
        "text_runs": slide_xml.count("<a:t>"),
        "custom_geometry_elements": slide_xml.count("<a:custGeom>"),
        "slide_number_placeholder": "sldNum" in slide_xml or "Slide Number" in slide_xml,
        "is_editable_rebuild": not media_files and slide_xml.count("<p:sp>") > 120 and slide_xml.count("<a:t>") > 60,
    }
    QA_PATH.write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")
    return qa


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    render()
    print(json.dumps(inspect_pptx(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
