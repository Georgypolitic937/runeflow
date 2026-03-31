import base64
import io
import json
import os
import re
from typing import Sequence

import requests
from PIL import Image, ImageGrab

def img_to_data_url(pil_image, fmt="PNG"):
    buf = io.BytesIO()
    pil_image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

def chat_vision(
    system_text: str,
    user_text: str,
    pil_images: Sequence[Image.Image],
    temperature: float = 0.1,
    max_tokens: int = 256,
    timeout: int = 30,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    base_url = (base_url or os.getenv("LM_BASE", "http://localhost:1234/v1")).rstrip("/")
    model = model or os.getenv("LM_MODEL", "qwen2-vl-7b-instruct")
    api_key = api_key or os.getenv("LM_API_KEY", "lm-studio")
    url = f"{base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    content = [{"type": "text", "text": user_text}]
    for im in pil_images:
        content.append({"type": "image_url", "image_url": {"url": img_to_data_url(im)}})
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def grab_screen_pil(bbox=None):
    return ImageGrab.grab(bbox=bbox)

def downscale_max(pil_img, max_side=1024):
    w, h = pil_img.size
    scale = max(w, h) / float(max_side)
    if scale <= 1.0:
        return pil_img
    return pil_img.resize((int(w / scale), int(h / scale)), Image.BICUBIC)

def parse_xy(out):
    if not isinstance(out, str):
        return None
    m = re.search(r'\{.*\}', out, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        x, y = obj.get("x"), obj.get("y")
        if isinstance(x, int) and isinstance(y, int):
            return (x, y)
    except Exception:
        return None
    return None

def ask_vlm_for_click(full_screenshot_pil, template_pil, timeout=25, roi=None):
    try:
        x_off = y_off = 0
        if roi:
            x1, y1, x2, y2 = roi
            full_screenshot_pil = full_screenshot_pil.crop((x1, y1, x2, y2))
            x_off, y_off = x1, y1

        shot = downscale_max(full_screenshot_pil, 1024)
        tmpl = downscale_max(template_pil, 512)

        system = (
            "You are a precise UI assistant. Given a screenshot and a small template image, "
            "locate where the template appears and return pixel coordinates of the template's "
            "top-left corner. If unsure, return NONE."
        )
        user = (
            'Find the exact top-left corner of the template within the screenshot. '
            'Return ONLY JSON like {"x":123, "y":456}. '
            'If not visible, return {"x":null, "y":null}.'
        )

        for t in (timeout, timeout + 10, timeout + 20):
            out = chat_vision(system, user, [shot, tmpl], temperature=0.0, max_tokens=80, timeout=t)
            xy = parse_xy(out)
            if xy is not None:
                x, y = xy
                if isinstance(x, int) and isinstance(y, int):
                    return (x + x_off, y + y_off)
        return None
    except Exception:
        return None
