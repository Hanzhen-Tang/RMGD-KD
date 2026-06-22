from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PPTX_PATH = OUT_DIR / "cckd_distillation_framework_optimized_editable.pptx"
PREVIEW_PATH = OUT_DIR / "cckd_distillation_framework_optimized_preview.png"
QA_PATH = OUT_DIR / "cckd_distillation_framework_optimized_qa.json"

W, H = 13.333, 7.5
SCALE = 160

C = {
    "bg": "#F8FAFC",
    "white": "#FFFFFF",
    "ink": "#172033",
    "muted": "#64748B",
    "line": "#94A3B8",
    "soft": "#CBD5E1",
    "blue": "#2563EB",
    "blue_deep": "#1E3A8A",
    "blue_soft": "#DBEAFE",
    "green": "#16A34A",
    "green_deep": "#166534",
    "green_soft": "#DCFCE7",
    "violet": "#7C3AED",
    "violet_deep": "#4C1D95",
    "violet_soft": "#EDE9FE",
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


def text(draw: ImageDraw.ImageDraw, value: str, x: float, y: float, size: int, color: str, bold: bool = False, anchor: str = "la", align: str = "left") -> None:
    draw.multiline_text(pxy(x, y), value, fill=color, font=font(size, bold), anchor=anchor, align=align, spacing=3)


def box(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, stroke: str, fill: str = "#FFFFFF", width: int = 2, r: float = 0.08) -> None:
    draw.rounded_rectangle([px(x), px(y), px(x + w), px(y + h)], radius=px(r), fill=fill, outline=stroke, width=width)


def arrow(draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float, color: str = "#172033", width: int = 2, dash: bool = False) -> None:
    p1, p2 = pxy(x1, y1), pxy(x2, y2)
    if dash:
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(1, int(dist // 18))
        for i in range(steps):
            if i % 2 == 0:
                a, b = i / steps, min(1, (i + 1) / steps)
                draw.line([(p1[0] + (p2[0] - p1[0]) * a, p1[1] + (p2[1] - p1[1]) * a), (p1[0] + (p2[0] - p1[0]) * b, p1[1] + (p2[1] - p1[1]) * b)], fill=color, width=width)
    else:
        draw.line([p1, p2], fill=color, width=width)
    ang = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
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
        arrow(draw, pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1], color, width, dash if i == len(pts) - 2 else False)


def dot(draw: ImageDraw.ImageDraw, x: float, y: float, color: str, r: float = 0.035) -> None:
    draw.ellipse([px(x - r), px(y - r), px(x + r), px(y + r)], fill=color, outline=color)


def heatmap(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, values: list[list[float]], low: str, high: str) -> None:
    rows, cols = len(values), len(values[0])
    for r, row in enumerate(values):
        for c, value in enumerate(row):
            draw.rectangle([px(x + c * w / cols), px(y + r * h / rows), px(x + (c + 1) * w / cols), px(y + (r + 1) * h / rows)], fill=mix(low, high, value), outline="#A78BFA")
    draw.rectangle([px(x), px(y), px(x + w), px(y + h)], outline="#7C3AED", width=2)


def tensor(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float, dx: float, dy: float, tint: str) -> None:
    light = "#EAF2FF" if tint == "blue" else "#EAFBEF"
    dark = "#93C5FD" if tint == "blue" else "#86EFAC"
    top = [pxy(x, y), pxy(x + dx, y - dy), pxy(x + w + dx, y - dy), pxy(x + w, y)]
    side = [pxy(x + w, y), pxy(x + w + dx, y - dy), pxy(x + w + dx, y + h - dy), pxy(x + w, y + h)]
    front = [pxy(x, y), pxy(x + w, y), pxy(x + w, y + h), pxy(x, y + h)]
    draw.polygon(top, fill=mix(light, "#FFFFFF", 0.2), outline="#172033")
    draw.polygon(side, fill=mix(dark, light, 0.42), outline="#172033")
    draw.polygon(front, fill=light, outline="#172033")
    for i in range(1, 4):
        gx, gy = x + w * i / 4, y + h * i / 4
        draw.line([pxy(gx, y), pxy(gx, y + h)], fill="#475569", width=1)
        draw.line([pxy(x, gy), pxy(x + w, gy)], fill="#475569", width=1)


def model_icon(draw: ImageDraw.ImageDraw, x: float, y: float, color: str) -> None:
    for i in range(3):
        draw.polygon([pxy(x + i * 0.12, y + 0.10), pxy(x + i * 0.12 + 0.12, y), pxy(x + i * 0.12 + 0.12, y + 0.56), pxy(x + i * 0.12, y + 0.66)], fill=mix(color, "#FFFFFF", 0.55), outline=color)


def graph_icon(draw: ImageDraw.ImageDraw, x: float, y: float, color: str) -> None:
    pts = [(x + 0.08, y + 0.24), (x + 0.32, y + 0.12), (x + 0.50, y + 0.32), (x + 0.36, y + 0.54), (x + 0.12, y + 0.50)]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 3), (0, 2)]:
        draw.line([pxy(*pts[a]), pxy(*pts[b])], fill=color, width=2)
    for a, b in pts:
        dot(draw, a, b, color, 0.035)


def render() -> None:
    img = Image.new("RGB", (px(W), px(H)), C["bg"])
    d = ImageDraw.Draw(img)
    text(d, "CCKD Distillation Framework", 0.48, 0.22, 32, C["ink"], True)
    text(d, "Optimized editable reconstruction: forward paths are solid; student optimization is dashed.", 12.85, 0.29, 13, C["muted"], False, "ra")
    d.line([pxy(0.48, 0.68), pxy(12.85, 0.68)], fill=C["soft"], width=2)

    box(d, 0.48, 1.18, 1.55, 2.88, C["blue"])
    text(d, "Input: Historical\nTraffic Data", 1.25, 1.31, 13, C["blue_deep"], True, "ma", "center")
    tensor(d, 0.77, 2.14, 0.68, 0.88, 0.18, 0.15, "blue")
    text(d, "X in R^(B x C x N x T_in)", 1.25, 3.76, 11, C["ink"], False, "ma")

    box(d, 2.38, 0.96, 1.94, 1.26, C["blue"])
    text(d, "Teacher Model\n(GWNet)", 3.35, 1.10, 13, C["blue_deep"], True, "ma", "center")
    model_icon(d, 2.60, 1.36, C["blue"])
    graph_icon(d, 3.36, 1.30, C["blue_deep"])
    box(d, 2.38, 3.08, 1.94, 1.34, C["green"])
    text(d, "Student Model\n(Lightweight GNN)", 3.35, 3.22, 13, C["green_deep"], True, "ma", "center")
    model_icon(d, 2.60, 3.55, C["green"])
    graph_icon(d, 3.36, 3.50, C["green_deep"])
    box(d, 4.60, 1.16, 0.86, 0.68, C["blue"])
    text(d, "Teacher\nPrediction\nYhat_te", 5.03, 1.23, 11, C["blue_deep"], True, "ma", "center")
    box(d, 4.60, 3.38, 0.86, 0.68, C["green"])
    text(d, "Student\nPrediction\nYhat_st", 5.03, 3.45, 11, C["green_deep"], True, "ma", "center")
    box(d, 5.78, 4.66, 1.28, 0.66, C["green"])
    text(d, "Final Forecast\nYhat_st,t+1:t+H", 6.42, 4.78, 11, C["blue_deep"], True, "ma", "center")

    arrow(d, 2.03, 2.02, 2.36, 1.58)
    arrow(d, 2.03, 3.24, 2.36, 3.76)
    arrow(d, 4.32, 1.58, 4.58, 1.50)
    arrow(d, 4.32, 3.77, 4.58, 3.74)
    arrow(d, 5.46, 3.72, 5.76, 4.92, C["green_deep"])

    box(d, 5.73, 0.86, 3.92, 2.62, C["violet"], "#FFFFFF", 3)
    text(d, "CCKD Distillation", 7.69, 1.01, 17, C["violet_deep"], True, "ma")
    box(d, 5.90, 1.22, 1.80, 1.92, C["violet"], C["violet_soft"])
    text(d, "1. Confidence-Adaptive\nDual-Path Distillation", 6.80, 1.34, 11, C["violet_deep"], True, "ma", "center")
    heatmap(d, 6.04, 1.72, 0.50, 0.78, [[.25,.75,.55],[.35,.95,.3],[.7,.45,.88],[.2,.62,.38]], "#F5F3FF", "#5B21B6")
    dot(d, 6.88, 1.90, C["blue"])
    text(d, "Absolute-Value\nDistillation", 7.00, 1.78, 10, C["blue_deep"], True)
    dot(d, 6.88, 2.36, C["red"])
    text(d, "Trend\nDistillation", 7.00, 2.25, 10, C["red"], True)
    text(d, "x", 7.94, 2.17, 24, C["ink"], False, "ma")
    box(d, 8.04, 1.22, 1.40, 1.92, C["violet"])
    text(d, "2. Soft Curriculum\nWeighting over Horizons", 8.74, 1.34, 10, C["violet_deep"], True, "ma", "center")
    for i in range(12):
        d.rectangle([px(8.22 + i * .075), px(1.95), px(8.22 + i * .075 + .07), px(2.13)], fill=mix("#F5F3FF", "#5B21B6", 1 - i / 12), outline="#C4B5FD")
    text(d, "H1      H2      ...      HH", 8.75, 2.27, 11, C["ink"], False, "ma")

    box(d, 11.04, 0.96, 1.34, 0.84, C["blue"])
    text(d, "Ground Truth\n(Training Label)\nY in R^(B x 1 x N x H)", 11.71, 1.08, 10, C["blue_deep"], True, "ma", "center")
    box(d, 10.74, 2.92, 1.38, 0.72, C["blue"], C["blue_soft"])
    text(d, "Supervised Loss\nL_sup", 11.43, 3.06, 13, C["blue_deep"], True, "ma", "center")
    box(d, 9.72, 4.56, 1.38, 0.72, C["violet"], C["violet_soft"])
    text(d, "Curriculum-Weighted\nDistillation Loss\nL_distill", 10.41, 4.66, 11, C["violet_deep"], True, "ma", "center")
    box(d, 12.02, 4.12, 1.10, 0.72, C["violet"])
    text(d, "Total Training Loss\nL_total", 12.57, 4.28, 11, C["violet_deep"], True, "ma", "center")
    text(d, "L_total = L_sup + lambda L_distill", 12.25, 5.06, 11, C["ink"], False, "ma")

    elbow(d, [(5.46, 1.50), (5.62, 1.50), (5.62, 2.08), (5.88, 2.08)], C["blue_deep"])
    elbow(d, [(5.46, 3.72), (5.62, 3.72), (5.62, 2.42), (5.88, 2.42)], C["green_deep"])
    arrow(d, 7.72, 3.48, 10.08, 4.54, C["violet_deep"])
    arrow(d, 11.71, 1.80, 11.44, 2.90)
    elbow(d, [(7.06, 4.98), (10.34, 4.98), (10.34, 3.28), (10.74, 3.28)], C["green_deep"])
    arrow(d, 11.43, 3.64, 12.02, 4.40, C["blue_deep"])
    arrow(d, 11.10, 4.92, 12.02, 4.60, C["violet_deep"])
    elbow(d, [(12.70, 4.84), (12.70, 6.43), (2.98, 6.43), (2.98, 4.45)], C["ink"], 2, True)
    text(d, "Backpropagation / optimize student", 6.70, 6.48, 11, C["ink"], False, "ma")

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
        "is_editable_rebuild": not media_files and slide_xml.count("<p:sp>") > 80 and slide_xml.count("<a:t>") > 35,
    }
    QA_PATH.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return qa


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    render()
    print(json.dumps(inspect_pptx(), indent=2))


if __name__ == "__main__":
    main()
