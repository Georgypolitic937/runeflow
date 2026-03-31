import os
import time
from typing import Optional

import cv2
import numpy as np
from PIL import ImageGrab

from .config import ReplayConfig

def grab_screen_bgr(bbox=None):
    img = ImageGrab.grab(bbox=bbox) if bbox else ImageGrab.grab()
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def imread_bgr(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Anchor image not found: {path}")
    return img

def nms_second_best(res, best_loc, exclude_radius=24):
    res2 = res.copy()
    h, w = res.shape
    bx, by = best_loc
    Y, X = np.ogrid[:h, :w]
    mask = (X - bx) ** 2 + (Y - by) ** 2 <= exclude_radius ** 2
    res2[mask] = -1.0
    _, maxv2, _, maxl2 = cv2.minMaxLoc(res2)
    return maxv2, maxl2

def best_match_for_scale(screen_bgr, tmpl_bgr, scale=1.0, edge_low_high=(50,150)):
    if scale != 1.0:
        th, tw = tmpl_bgr.shape[:2]
        tmpl_bgr = cv2.resize(tmpl_bgr, (int(tw*scale), int(th*scale)), interpolation=cv2.INTER_AREA)
    h, w = tmpl_bgr.shape[:2]
    best = (-1.0, None, w, h, "none", None)

    res = cv2.matchTemplate(screen_bgr, tmpl_bgr, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxl = cv2.minMaxLoc(res)
    if maxv > best[0]:
        best = (maxv, maxl, w, h, "color", res)

    grayS = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    grayT = cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(grayS, grayT, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxl = cv2.minMaxLoc(res)
    if maxv > best[0]:
        best = (maxv, maxl, w, h, "gray", res)

    edgeS = cv2.Canny(grayS, edge_low_high[0], edge_low_high[1])
    edgeT = cv2.Canny(grayT, edge_low_high[0], edge_low_high[1])
    res = cv2.matchTemplate(edgeS, edgeT, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxl = cv2.minMaxLoc(res)
    if maxv > best[0]:
        best = (maxv, maxl, w, h, "edges", res)
    return best

def resolve_relative(json_path: str, relpath: str) -> str:
    if os.path.isabs(relpath):
        return relpath
    return os.path.normpath(os.path.join(os.path.dirname(json_path), relpath))

def clip_bbox(x1, y1, x2, y2, screen_w, screen_h):
    return (max(0, x1), max(0, y1), min(screen_w, x2), min(screen_h, y2))

def screen_size():
    img = ImageGrab.grab()
    return img.size

def verify_match(screen_bgr, tmpl_bgr, tl_abs, threshold=0.85, pad=6):
    x, y = tl_abs
    h, w = tmpl_bgr.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = x + w + pad, y + h + pad
    roi = screen_bgr[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < h // 2 or roi.shape[1] < w // 2:
        return False
    res = cv2.matchTemplate(roi, tmpl_bgr, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, _ = cv2.minMaxLoc(res)
    return maxv >= threshold

def mini_verify_match(screen_bgr, tmpl_bgr, tl_abs, threshold=0.80, pad=8):
    x, y = tl_abs
    th, tw = tmpl_bgr.shape[:2]
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(screen_bgr.shape[1], x + tw + pad), min(screen_bgr.shape[0], y + th + pad)
    roi = screen_bgr[y1:y2, x1:x2]
    if roi.size == 0 or roi.shape[0] < th // 2 or roi.shape[1] < tw // 2:
        return False
    res = cv2.matchTemplate(roi, tmpl_bgr, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, _ = cv2.minMaxLoc(res)
    return maxv >= float(threshold)

def find_anchor(anchor_dict: dict, json_path: str, cfg: ReplayConfig, last_xy=None):
    threshold = float(anchor_dict.get("conf", 0.65))
    timeout = float(anchor_dict.get("timeout", 8.0))
    apath = resolve_relative(json_path, anchor_dict.get("path", ""))
    try:
        tmpl = imread_bgr(apath)
    except FileNotFoundError:
        if cfg.debug_match:
            print(f"[match] missing anchor: {apath}")
        return None

    bbox = None
    if last_xy and cfg.lock_to_last_window > 0:
        sw, sh = screen_size()
        L = cfg.lock_to_last_window
        cx, cy = last_xy
        bbox = clip_bbox(cx - L, cy - L, cx + L, cy + L, sw, sh)

    t_end = time.time() + timeout
    while time.time() < t_end:
        screen = grab_screen_bgr(bbox=bbox)
        best_overall = (-1.0, None, 0, 0, "none", None)
        best_scale = 1.0
        for sc in cfg.scale_sweep:
            score, tl, w, h, mode, res = best_match_for_scale(screen, tmpl, scale=sc, edge_low_high=cfg.edge_low_high)
            if tl is not None and score > best_overall[0]:
                best_overall = (score, tl, w, h, mode, res)
                best_scale = sc

        score, tl, w, h, mode, resmap = best_overall
        if tl is not None:
            second_score, _ = nms_second_best(resmap, tl, exclude_radius=18)
            unique_ok = (score - second_score) >= cfg.uniqueness_margin
            if cfg.debug_match:
                print(
                    f"[match] {os.path.basename(apath)} score={score:.3f} second={second_score:.3f} "
                    f"Δ={score-second_score:.3f} uniq={unique_ok} mode={mode} scale={best_scale} roi={'yes' if bbox else 'no'}"
                )
            if score >= threshold and unique_ok:
                abs_x = tl[0] + (bbox[0] if bbox else 0)
                abs_y = tl[1] + (bbox[1] if bbox else 0)
                th, tw = tmpl.shape[:2]
                tws, ths = int(round(tw * best_scale)), int(round(th * best_scale))
                if tws < 2 or ths < 2:
                    tmpl_to_verify = tmpl
                    tws, ths = tw, th
                else:
                    tmpl_to_verify = cv2.resize(tmpl, (tws, ths), interpolation=cv2.INTER_AREA)
                if cfg.verify_enabled:
                    screen_full = grab_screen_bgr()
                    verify_ok = verify_match(screen_full, tmpl_to_verify, (abs_x, abs_y), threshold=max(0.78, threshold - 0.05), pad=6)
                    if not verify_ok:
                        if cfg.debug_match:
                            print("[match] verify failed; retrying…")
                        time.sleep(cfg.poll_sleep)
                        continue
                return (abs_x, abs_y, tws, ths)
        time.sleep(cfg.poll_sleep)
    if cfg.debug_match:
        print("[match] timeout waiting for anchor")
    return None
