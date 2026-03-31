import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyautogui
from PIL import ImageGrab, Image
from pynput import keyboard, mouse

from .config import RecorderConfig

def ensure_dir(path: str | os.PathLike[str]) -> None:
    os.makedirs(path, exist_ok=True)

def capture_anchor(x: int, y: int, assets_dir: str, size: int) -> tuple[str, int, int, int]:
    ensure_dir(assets_dir)
    half = size // 2
    left, top = max(int(x) - half, 0), max(int(y) - half, 0)
    img = ImageGrab.grab(bbox=(left, top, left + size, top + size))
    out = os.path.join(assets_dir, f"{uuid.uuid4().hex[:12]}.png")
    img.save(out)
    return os.path.abspath(out), left, top, size

def normalize_recording(events_abs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    prev = 0.0
    for ev in events_abs:
        delay = max(0.0, ev["t"] - prev)
        prev = ev["t"]
        row: dict[str, Any] = {"delay": delay, "type": ev["type"]}
        if ev["type"] == "click":
            row.update({"x": ev["x"], "y": ev["y"], "button": ev["button"]})
            if ev.get("anchor"):
                row["anchor"] = dict(ev["anchor"])
        else:
            row.update({"key": ev["key"]})
        out.append(row)
    return out

def save_recording(events: list[dict[str, Any]], json_path: str) -> str:
    assets_dir = os.path.join(os.path.dirname(json_path), "assets")
    ensure_dir(assets_dir)
    normalized = []
    for idx, step in enumerate(events):
        step = dict(step)
        if step["type"] == "key" and str(step.get("key")).lower() == "wait":
            step["key"] = "Key.pause"
        anc = step.get("anchor")
        if anc and "path" in anc:
            src = anc["path"]
            if not os.path.isabs(src):
                src = os.path.abspath(src)
            dst = os.path.join(assets_dir, f"step_{idx:04d}.png")
            try:
                Image.open(src).convert("RGB").save(dst, "PNG")
                anc = dict(anc)
                anc["path"] = os.path.join("assets", os.path.basename(dst))
                step["anchor"] = anc
            except Exception:
                pass
        normalized.append(step)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"version": "1.0", "events": normalized}, f, indent=2)
    return json_path

@dataclass
class Recorder:
    assets_dir: str
    config: RecorderConfig = RecorderConfig()

    def __post_init__(self) -> None:
        self.events_abs: list[dict[str, Any]] = []
        self.is_recording = False
        self._t0 = 0.0
        self.stop_event = threading.Event()
        self._mouse_listener = None
        self._key_listener = None

    def ts(self) -> float:
        return time.time() - self._t0

    def start(self) -> None:
        self.events_abs = []
        self.is_recording = True
        self.stop_event.clear()
        self._t0 = time.time()
        self._mouse_listener = mouse.Listener(on_click=self._on_click)
        self._key_listener = keyboard.Listener(on_press=self._on_key_press)
        self._mouse_listener.start()
        self._key_listener.start()

    def stop(self) -> list[dict[str, Any]]:
        self.is_recording = False
        try:
            if self._mouse_listener:
                self._mouse_listener.stop()
            if self._key_listener:
                self._key_listener.stop()
        except Exception:
            pass
        return normalize_recording(self.events_abs)

    def _on_click(self, x, y, button, pressed):
        if not self.is_recording or not pressed:
            return
        try:
            apath, left, top, size = capture_anchor(x, y, self.assets_dir, self.config.anchor_size)
            anchor = {
                "path": apath,
                "conf": self.config.default_confidence,
                "timeout": self.config.default_timeout,
                "ox": int(x - left),
                "oy": int(y - top),
                "w": int(size),
                "h": int(size),
            }
        except Exception:
            anchor = None
        ev = {"type": "click", "x": x, "y": y, "button": str(button), "t": self.ts()}
        if anchor:
            ev["anchor"] = anchor
        self.events_abs.append(ev)

    def _on_key_press(self, key):
        if self.is_recording and str(key) == "Key.f8":
            self.stop_event.set()
            self.is_recording = False
            return False
        if not self.is_recording:
            return
        try:
            k = key.char
        except Exception:
            k = str(key)
        self.events_abs.append({"type": "key", "key": k, "t": self.ts()})
