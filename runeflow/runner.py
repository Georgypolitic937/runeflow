import json
import os
import random
import time
from threading import Event

import cv2
import pyautogui
from PIL import Image

from .config import ReplayConfig, VLMConfig
from .vision import find_anchor, grab_screen_bgr, imread_bgr, mini_verify_match, resolve_relative, screen_size
from .vlm import ask_vlm_for_click, grab_screen_pil

pyautogui.FAILSAFE = False

def button_name(b):
    s = str(b).lower() if b is not None else "button.left"
    if "right" in s:
        return "right"
    if "middle" in s:
        return "middle"
    return "left"

def jitter(x, y, px):
    if px <= 0:
        return int(x), int(y)
    return int(x + random.randint(-px, px)), int(y + random.randint(-px, px))

def sleep_with_stop(sec, stop_event):
    end = time.time() + sec
    while time.time() < end:
        if stop_event.is_set():
            return
        time.sleep(min(0.05, end - time.time()))

def install_esc_hotkey(stop_event: Event):
    from pynput import keyboard
    def on_press(key):
        if str(key) == "Key.esc":
            stop_event.set()
            return False
    keyboard.Listener(on_press=on_press, daemon=True).start()

def run_events(
    events,
    json_path,
    jitter_px=0,
    slow_scale=1.0,
    fast_scale=1.0,
    stop_event: Event | None = None,
    recovery_events=None,
    replay_config: ReplayConfig | None = None,
    vlm_config: VLMConfig | None = None,
):
    stop_event = stop_event or Event()
    replay_config = replay_config or ReplayConfig()
    vlm_config = vlm_config or VLMConfig()
    last_click_xy = None

    for idx, ev in enumerate(events):
        if stop_event.is_set():
            break

        click_x, click_y = ev.get("x"), ev.get("y")

        if "anchor" in ev and ev["anchor"]:
            anchor = ev["anchor"]
            click_x, click_y = ev.get("x"), ev.get("y")
            step_max_s = float(anchor.get("max_wait_s", replay_config.max_anchor_stubborn_s))
            step_max_tries = int(anchor.get("max_tries", replay_config.max_anchor_tries))

            got_point_from_any = False
            tlx = tly = w = h = None

            if vlm_config.enabled:
                if replay_config.debug_match:
                    print(f"[vlm] asking model for step={idx} …")
                vlm_roi = None
                if last_click_xy:
                    L = vlm_config.roi_half_width
                    cx, cy = last_click_xy
                    sw, sh = screen_size()
                    x1 = max(0, cx - L); y1 = max(0, cy - L)
                    x2 = min(sw, cx + L); y2 = min(sh, cy + L)
                    vlm_roi = (x1, y1, x2, y2)

                anchor_path = resolve_relative(json_path, anchor.get("path", ""))
                tmpl_bgr = imread_bgr(anchor_path)
                screen_pil = grab_screen_pil()

                vlm_xy = ask_vlm_for_click(
                    screen_pil,
                    Image.fromarray(cv2.cvtColor(tmpl_bgr, cv2.COLOR_BGR2RGB)),
                    timeout=int(vlm_config.timeout_secs),
                    roi=vlm_roi,
                )
                if vlm_xy is not None:
                    vx, vy = vlm_xy
                    screen_bgr = grab_screen_bgr()
                    verified = mini_verify_match(
                        screen_bgr,
                        tmpl_bgr,
                        (vx, vy),
                        threshold=float(vlm_config.min_confidence),
                        pad=int(vlm_config.verify_delta),
                    )
                    if verified:
                        ox = int(anchor.get("ox", 0) or 0)
                        oy = int(anchor.get("oy", 0) or 0)
                        click_x, click_y = vx + ox, vy + oy
                        got_point_from_any = True
                        if replay_config.debug_match:
                            print(f"[vlm] verified step={idx} -> ({click_x},{click_y})")
                    elif replay_config.debug_match:
                        print(f"[vlm] verification failed for step={idx}; falling back to OpenCV")

            if not got_point_from_any:
                start_t = time.time()
                tries = 0
                while not stop_event.is_set():
                    tries += 1
                    found = find_anchor(anchor, json_path, replay_config, last_xy=last_click_xy)
                    if found is not None:
                        tlx, tly, w, h = found
                        ox = int(anchor.get("ox", 0) or 0)
                        oy = int(anchor.get("oy", 0) or 0)
                        click_x, click_y = tlx + ox, tly + oy
                        got_point_from_any = True
                        break

                    if replay_config.debug_match:
                        print(f"[run] waiting for anchor… step={idx} try={tries}")

                    if (time.time() - start_t) >= step_max_s or tries >= step_max_tries:
                        if replay_config.debug_match:
                            print(f"[run] stubborn cap hit (t>={step_max_s}s or tries>={step_max_tries})")
                        break
                    time.sleep(replay_config.poll_sleep)

                if not got_point_from_any:
                    if replay_config.fallback_to_absolute and (click_x is not None and click_y is not None):
                        if replay_config.debug_match:
                            print(f"[run] FAILSAFE → recorded (x,y)=({click_x},{click_y})")
                    else:
                        if replay_config.debug_match:
                            print("[run] anchor unresolved and no failsafe; skipping step")
                        continue

        delay = float(ev.get("delay", 0.0)) * random.uniform(float(slow_scale), float(fast_scale))
        if delay > 0:
            sleep_with_stop(delay, stop_event)
        if stop_event.is_set():
            break

        if ev["type"] == "click":
            x, y = jitter(click_x, click_y, int(jitter_px))
            pyautogui.moveTo(x, y, duration=random.uniform(*replay_config.move_duration_range))
            btn = button_name(ev.get("button", "Button.left"))
            pyautogui.click(button=btn)
            last_click_xy = (x, y)
            if replay_config.debug_match:
                print(f"[click] ({x},{y}) button={btn} step={idx}")
        elif ev["type"] == "key":
            k = ev["key"]
            if str(k).lower() in ("wait", "key.pause"):
                continue
            keyname = k if (isinstance(k, str) and len(k) == 1) else str(k).replace("Key.", "")
            pyautogui.press(keyname)
            if replay_config.debug_match:
                print(f"[key] {keyname} step={idx}")
    return "ok"

def run_json(
    path,
    loops=1,
    jitter_px=0,
    slow_scale=1.0,
    fast_scale=1.0,
    stop_event: Event | None = None,
    recovery_path=None,
    replay_config: ReplayConfig | None = None,
    vlm_config: VLMConfig | None = None,
):
    stop_event = stop_event or Event()
    with open(path, "r", encoding="utf-8") as f:
        events = json.load(f).get("events", [])

    recovery_events = None
    if recovery_path:
        try:
            with open(recovery_path, "r", encoding="utf-8") as rf:
                recovery_events = json.load(rf).get("events", [])
        except Exception:
            recovery_events = None

    install_esc_hotkey(stop_event)
    i, loops = 0, int(loops)
    while not stop_event.is_set() and i < loops:
        result = run_events(
            events,
            path,
            jitter_px=jitter_px,
            slow_scale=slow_scale,
            fast_scale=fast_scale,
            stop_event=stop_event,
            recovery_events=recovery_events,
            replay_config=replay_config,
            vlm_config=vlm_config,
        )
        if result == "restart":
            continue
        i += 1
    return "stopped" if stop_event.is_set() else "ok"
