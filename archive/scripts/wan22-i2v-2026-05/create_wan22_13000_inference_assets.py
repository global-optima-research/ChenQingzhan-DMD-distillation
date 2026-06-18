#!/usr/bin/env python3
from __future__ import annotations

import math
import os
import struct
import zlib
from pathlib import Path


OUT = Path("/data/chenqingzhan/inference_inputs/wan22_13000_10distinct_20260504")
W, H = 1280, 704

SCENES = [
    {
        "name": "garden_swing",
        "prompt": "A quiet garden swing moving gently under warm morning light, flowers and leaves swaying in a soft breeze, realistic natural motion",
        "sky": (120, 180, 230),
        "ground": (52, 126, 70),
        "accent": (210, 78, 82),
        "mode": "swing",
    },
    {
        "name": "red_balloon_city",
        "prompt": "A bright red balloon drifting between tall city buildings at golden hour, reflections in windows, slow cinematic camera movement",
        "sky": (245, 174, 90),
        "ground": (70, 76, 88),
        "accent": (220, 36, 48),
        "mode": "city",
    },
    {
        "name": "sailboat_lake",
        "prompt": "A small white sailboat crossing a calm blue lake with mountain silhouettes in the distance, gentle ripples and slow camera drift",
        "sky": (110, 180, 235),
        "ground": (36, 94, 150),
        "accent": (245, 245, 238),
        "mode": "sailboat",
    },
    {
        "name": "snow_cabin",
        "prompt": "A cozy wooden cabin in fresh snow at dusk, chimney smoke rising slowly while pine trees move in the cold wind",
        "sky": (80, 92, 130),
        "ground": (226, 236, 244),
        "accent": (190, 95, 45),
        "mode": "cabin",
    },
    {
        "name": "neon_street",
        "prompt": "A rainy neon street at night with colorful signs reflected on wet pavement, light rain and subtle handheld camera motion",
        "sky": (22, 28, 48),
        "ground": (28, 32, 45),
        "accent": (236, 53, 174),
        "mode": "neon",
    },
    {
        "name": "forest_waterfall",
        "prompt": "A green forest waterfall flowing over mossy rocks, mist drifting through shafts of sunlight, tranquil natural movement",
        "sky": (92, 150, 120),
        "ground": (36, 88, 55),
        "accent": (210, 238, 235),
        "mode": "waterfall",
    },
    {
        "name": "desert_monolith",
        "prompt": "A sandstone monolith in a wide desert landscape at sunrise, sand blowing across the foreground, slow epic camera pan",
        "sky": (245, 185, 108),
        "ground": (202, 132, 76),
        "accent": (126, 72, 52),
        "mode": "desert",
    },
    {
        "name": "koi_pond",
        "prompt": "Colorful koi fish swimming below lotus leaves in a quiet pond, soft ripples on the water, overhead cinematic view",
        "sky": (78, 148, 142),
        "ground": (38, 96, 92),
        "accent": (232, 112, 54),
        "mode": "koi",
    },
    {
        "name": "train_station",
        "prompt": "An old train arriving at a misty countryside station, steam rolling along the platform, nostalgic film look",
        "sky": (156, 170, 178),
        "ground": (88, 82, 76),
        "accent": (40, 74, 92),
        "mode": "train",
    },
    {
        "name": "observatory_stars",
        "prompt": "A hilltop observatory under a clear star-filled sky, telescope dome slowly rotating, Milky Way visible above",
        "sky": (15, 24, 58),
        "ground": (38, 48, 74),
        "accent": (226, 226, 210),
        "mode": "observatory",
    },
]


def clamp(v: float) -> int:
    return max(0, min(255, int(v)))


def blend(a, b, t):
    return tuple(clamp(a[i] * (1 - t) + b[i] * t) for i in range(3))


def write_png(path: Path, pixels: list[bytearray]) -> None:
    raw = b"".join(b"\x00" + bytes(row) for row in pixels)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def draw_rect(img, x0, y0, x1, y1, color):
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    for y in range(y0, y1):
        row = img[y]
        for x in range(x0, x1):
            i = x * 3
            row[i : i + 3] = bytes(color)


def draw_circle(img, cx, cy, r, color):
    r2 = r * r
    for y in range(max(0, cy - r), min(H, cy + r + 1)):
        row = img[y]
        dy = y - cy
        for x in range(max(0, cx - r), min(W, cx + r + 1)):
            if (x - cx) * (x - cx) + dy * dy <= r2:
                i = x * 3
                row[i : i + 3] = bytes(color)


def draw_triangle(img, pts, color):
    (x1, y1), (x2, y2), (x3, y3) = pts
    minx, maxx = max(0, min(x1, x2, x3)), min(W - 1, max(x1, x2, x3))
    miny, maxy = max(0, min(y1, y2, y3)), min(H - 1, max(y1, y2, y3))

    def sign(px, py, ax, ay, bx, by):
        return (px - bx) * (ay - by) - (ax - bx) * (py - by)

    for y in range(miny, maxy + 1):
        row = img[y]
        for x in range(minx, maxx + 1):
            b1 = sign(x, y, x1, y1, x2, y2) < 0
            b2 = sign(x, y, x2, y2, x3, y3) < 0
            b3 = sign(x, y, x3, y3, x1, y1) < 0
            if b1 == b2 == b3:
                i = x * 3
                row[i : i + 3] = bytes(color)


def base_image(scene):
    img = []
    for y in range(H):
        t = y / (H - 1)
        horizon = 0.58
        if t < horizon:
            c = blend(scene["sky"], (255, 245, 220), max(0, (t - 0.18) / horizon) * 0.25)
        else:
            c = blend(scene["ground"], (18, 26, 34), (t - horizon) / (1 - horizon) * 0.18)
        row = bytearray()
        for x in range(W):
            wave = math.sin((x / W) * math.pi * 4 + t * 2) * 5
            row.extend((clamp(c[0] + wave), clamp(c[1] + wave), clamp(c[2] + wave)))
        img.append(row)
    return img


def render(scene):
    img = base_image(scene)
    a = scene["accent"]
    mode = scene["mode"]

    if mode == "swing":
        draw_rect(img, 250, 140, 270, 440, (92, 58, 38))
        draw_rect(img, 520, 140, 540, 440, (92, 58, 38))
        draw_rect(img, 230, 120, 560, 145, (92, 58, 38))
        draw_rect(img, 365, 330, 430, 350, a)
        for x in range(375, 428, 42):
            draw_rect(img, x, 145, x + 4, 330, (45, 45, 45))
        for x in range(80, 1150, 95):
            draw_circle(img, x, 565 + (x % 3) * 18, 18, (226, 120 + x % 80, 80))
    elif mode == "city":
        for i, x in enumerate(range(80, 1180, 130)):
            draw_rect(img, x, 190 - (i % 3) * 45, x + 90, 585, (42, 50, 68))
            for wx in range(x + 15, x + 78, 25):
                for wy in range(220, 540, 45):
                    draw_rect(img, wx, wy, wx + 11, wy + 18, (245, 206, 92))
        draw_circle(img, 630, 250, 55, a)
        draw_rect(img, 627, 305, 633, 395, (160, 32, 42))
    elif mode == "sailboat":
        draw_triangle(img, [(580, 430), (675, 250), (675, 430)], (248, 248, 238))
        draw_triangle(img, [(685, 430), (685, 285), (780, 430)], (232, 238, 245))
        draw_rect(img, 555, 430, 805, 462, (108, 72, 48))
        draw_rect(img, 674, 255, 684, 462, (74, 54, 42))
        for x in range(0, W, 42):
            draw_rect(img, x, 530 + (x % 5), x + 26, 533 + (x % 5), (210, 230, 245))
    elif mode == "cabin":
        draw_rect(img, 485, 360, 795, 555, (126, 78, 46))
        draw_triangle(img, [(445, 360), (640, 230), (835, 360)], (82, 55, 45))
        draw_rect(img, 690, 260, 730, 340, (70, 50, 42))
        draw_rect(img, 565, 430, 635, 555, (76, 52, 38))
        draw_rect(img, 675, 410, 745, 470, (242, 178, 78))
        for x in [170, 300, 930, 1050, 1160]:
            draw_triangle(img, [(x, 220), (x - 70, 520), (x + 70, 520)], (42, 80, 62))
    elif mode == "neon":
        for x, color in [(190, (44, 216, 232)), (440, a), (820, (255, 206, 70)), (1030, (84, 255, 152))]:
            draw_rect(img, x, 180, x + 120, 360, (36, 40, 58))
            draw_rect(img, x + 12, 205, x + 108, 250, color)
            draw_rect(img, x + 22, 470, x + 98, 486, color)
        draw_triangle(img, [(520, H), (640, 390), (760, H)], (55, 58, 72))
    elif mode == "waterfall":
        draw_rect(img, 570, 210, 720, 610, (206, 238, 235))
        for x in [180, 310, 925, 1085]:
            draw_rect(img, x, 250, x + 45, 620, (64, 74, 45))
            draw_circle(img, x + 22, 230, 85, (48, 116, 55))
        for x in range(450, 850, 55):
            draw_circle(img, x, 610, 32, (70, 76, 68))
    elif mode == "desert":
        draw_triangle(img, [(615, 165), (500, 600), (760, 600)], a)
        draw_circle(img, 1040, 155, 60, (250, 214, 130))
        for x in range(0, W, 90):
            draw_rect(img, x, 575 + (x % 4) * 8, x + 70, 585 + (x % 4) * 8, (226, 160, 95))
    elif mode == "koi":
        for x, y, c in [(340, 420, a), (620, 500, (245, 222, 82)), (870, 390, (238, 92, 92))]:
            draw_circle(img, x, y, 58, c)
            draw_triangle(img, [(x - 65, y), (x - 125, y - 38), (x - 125, y + 38)], c)
        for x, y in [(230, 260), (510, 300), (900, 250), (1080, 450)]:
            draw_circle(img, x, y, 65, (64, 145, 76))
    elif mode == "train":
        draw_rect(img, 300, 375, 890, 505, a)
        draw_rect(img, 360, 295, 690, 380, (58, 80, 94))
        for x in [380, 520, 660, 810]:
            draw_circle(img, x, 510, 48, (36, 35, 34))
        draw_rect(img, 0, 555, W, 575, (70, 65, 60))
        draw_circle(img, 210, 250, 90, (210, 215, 210))
    elif mode == "observatory":
        for x, y, r in [(160, 120, 2), (330, 80, 2), (520, 150, 3), (770, 100, 2), (1050, 130, 3), (1160, 75, 2)]:
            draw_circle(img, x, y, r, (245, 244, 220))
        draw_rect(img, 545, 390, 735, 540, (196, 196, 185))
        draw_circle(img, 640, 390, 96, a)
        draw_rect(img, 500, 540, 780, 585, (90, 90, 86))

    return img


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prompt_lines = []
    image_lines = []
    for i, scene in enumerate(SCENES):
        path = OUT / f"{i:02d}_{scene['name']}.png"
        write_png(path, render(scene))
        prompt_lines.append(scene["prompt"])
        image_lines.append(str(path))
    (OUT / "prompts.txt").write_text("\n".join(prompt_lines) + "\n")
    (OUT / "images.txt").write_text("\n".join(image_lines) + "\n")
    print(OUT)
    print(f"prompts={len(prompt_lines)} images={len(image_lines)}")


if __name__ == "__main__":
    main()
