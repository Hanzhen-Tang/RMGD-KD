from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PPTX_PATH = OUT_DIR / "adaptive_soft_curriculum_compact_editable.pptx"
PREVIEW_PATH = OUT_DIR / "adaptive_soft_curriculum_compact_preview.png"
QA_PATH = OUT_DIR / "adaptive_soft_curriculum_compact_qa.json"

W, H = 13.333, 7.5
SCALE = 160

C = {
    "bg": "#FFFFFF",
    "ink": "#111827",
    "muted": "#5B6472",
    "blue": "#0B56B3",
    "blue_deep": "#003D8F",
    "blue_strong": "#004CC8",
    "white": "#FFFFFF",
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


def rotated_text(base: Image.Image, value: str, cx: float, cy: float, size: int, color: str, angle: int = 90, bold: bool = False) -> None:
    fnt = font(size, bold)
    scratch = Image.new("RGBA", (px(2.0), px(0.35)), (255, 255, 255, 0))
    d = ImageDraw.Draw(scratch)
    bbox = d.textbbox((0, 0), value, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(((scratch.width - tw) / 2, (scratch.height - th) / 2), value, fill=color, font=fnt)
    rotated = scratch.rotate(angle, expand=True)
    base.paste(rotated, (px(cx) - rotated.width // 2, px(cy) - rotated.height // 2), rotated)


def arrow(draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float, color: str, width: int = 2) -> None:
    p1, p2 = pxy(x1, y1), pxy(x2, y2)
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


def heatmap(draw: ImageDraw.ImageDraw, x: float, y: float, w: float, h: float) -> None:
    weights = [
        [0.96, 0.78, 0.55, 0.32, 0.23, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.09],
        [0.18, 0.22, 0.29, 0.42, 0.55, 0.70, 0.63, 0.48, 0.36, 0.27, 0.20, 0.16],
        [0.08, 0.11, 0.15, 0.20, 0.28, 0.38, 0.50, 0.64, 0.76, 0.86, 0.94, 1.00],
    ]
    rows, cols = len(weights), len(weights[0])
    for r, row in enumerate(weights):
        for c, value in enumerate(row):
            draw.rectangle(
                [
                    px(x + c * w / cols),
                    px(y + r * h / rows),
                    px(x + (c + 1) * w / cols),
                    px(y + (r + 1) * h / rows),
                ],
                fill=mix("#F4F9FF", "#0052CC", value),
                outline=C["white"],
                width=2,
            )


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
        "slide_number_placeholder": "sldNum" in slide_xml or "Slide Number" in slide_xml,
        "is_editable_rebuild": not media_files and slide_xml.count("<p:sp>") > 40 and slide_xml.count("<a:t>") > 10,
    }
    QA_PATH.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return qa


def render() -> None:
    img = Image.new("RGB", (px(W), px(H)), C["bg"])
    draw = ImageDraw.Draw(img)
    card = {"x": 4.82, "y": 0.82, "w": 3.70, "h": 5.62}
    draw.rounded_rectangle(
        [px(card["x"]), px(card["y"]), px(card["x"] + card["w"]), px(card["y"] + card["h"])],
        radius=px(0.08),
        fill=C["white"],
        outline=C["blue"],
        width=4,
    )
    text(draw, "Adaptive Soft\nCurriculum", card["x"] + card["w"] / 2, card["y"] + 0.34, 32, C["blue_deep"], True, "ma", "center")
    text(draw, "horizon-wise smooth weights", card["x"] + card["w"] / 2, card["y"] + 1.22, 22, C["ink"], False, "ma")

    hm = {"x": card["x"] + 0.92, "y": card["y"] + 1.83, "w": 2.48, "h": 1.45}
    heatmap(draw, hm["x"], hm["y"], hm["w"], hm["h"])
    arrow(draw, card["x"] + 0.34, hm["y"] + hm["h"], card["x"] + 0.34, hm["y"] - 0.02, C["ink"], 2)
    rotated_text(img, "Training progress", card["x"] + 0.12, hm["y"] + 0.72, 16, C["ink"], 90, True)
    text(draw, "Early", card["x"] + 0.84, hm["y"] + 0.20, 14, C["ink"], False, "ra")
    text(draw, "Middle", card["x"] + 0.84, hm["y"] + 0.68, 14, C["ink"], False, "ra")
    text(draw, "Late", card["x"] + 0.84, hm["y"] + 1.16, 14, C["ink"], False, "ra")
    arrow(draw, hm["x"], hm["y"] + hm["h"] + 0.18, hm["x"] + hm["w"] + 0.10, hm["y"] + hm["h"] + 0.18, C["ink"], 2)
    for label, t in [("H1", 0.04), ("H2", 0.27), ("H3", 0.50), ("...", 0.69), ("H10", 0.81), ("H11", 0.92), ("H12", 1.02)]:
        text(draw, label, hm["x"] + hm["w"] * t, hm["y"] + hm["h"] + 0.34, 12, C["ink"], False, "ma")
    text(draw, "Horizons (H)", hm["x"] + 1.24, hm["y"] + hm["h"] + 0.70, 17, C["ink"], True, "ma")
    arrow(draw, card["x"] + 0.80, card["y"] + 4.32, card["x"] + 2.98, card["y"] + 4.32, C["blue_strong"], 3)
    text(draw, "long horizons strengthen over training", card["x"] + card["w"] / 2, card["y"] + 4.50, 15, C["blue_strong"], False, "ma")
    text(draw, "all horizons active", card["x"] + card["w"] / 2, card["y"] + 5.18, 17, C["muted"], False, "ma")
    img.save(PREVIEW_PATH)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    render()
    print(json.dumps(inspect_pptx(), indent=2))


if __name__ == "__main__":
    main()
