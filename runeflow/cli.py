import argparse
import os
import time
from pathlib import Path

from .config import ReplayConfig, RecorderConfig, VLMConfig
from .recorder import Recorder, save_recording
from .runner import run_json

def cmd_record(args):
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    assets_dir = out.parent / "assets"
    recorder = Recorder(str(assets_dir), RecorderConfig(anchor_size=args.anchor_size, default_confidence=args.confidence, default_timeout=args.timeout))
    print("Recording started. Press F8 to stop.")
    recorder.start()
    while not recorder.stop_event.is_set():
        time.sleep(0.05)
    events = recorder.stop()
    save_recording(events, str(out))
    print(f"Saved {len(events)} events -> {out}")

def cmd_run(args):
    replay_cfg = ReplayConfig(
        fallback_to_absolute=not args.no_failsafe,
        debug_match=not args.quiet,
        verify_enabled=args.verify,
        max_anchor_stubborn_s=args.max_wait_s,
        max_anchor_tries=args.max_tries,
        lock_to_last_window=args.lock_to_last_window,
        uniqueness_margin=args.uniqueness_margin,
    )
    vlm_cfg = VLMConfig(
        enabled=args.vlm,
        timeout_secs=args.vlm_timeout,
        verify_delta=args.vlm_verify_delta,
        min_confidence=args.vlm_min_confidence,
    )
    run_json(
        args.recording,
        loops=args.loops,
        jitter_px=args.jitter,
        slow_scale=args.slow_scale,
        fast_scale=args.fast_scale,
        replay_config=replay_cfg,
        vlm_config=vlm_cfg,
    )

def build_parser():
    p = argparse.ArgumentParser(prog="runeflow", description="Open-source visual UI automation core.")
    sub = p.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Record clicks and keys until F8 is pressed.")
    rec.add_argument("output", help="Output JSON path")
    rec.add_argument("--anchor-size", type=int, default=48)
    rec.add_argument("--confidence", type=float, default=0.90)
    rec.add_argument("--timeout", type=float, default=4.0)
    rec.set_defaults(func=cmd_record)

    run = sub.add_parser("run", help="Replay a saved RuneFlow recording.")
    run.add_argument("recording", help="Path to JSON recording")
    run.add_argument("--loops", type=int, default=1)
    run.add_argument("--jitter", type=int, default=0)
    run.add_argument("--slow-scale", type=float, default=1.0)
    run.add_argument("--fast-scale", type=float, default=1.0)
    run.add_argument("--verify", action="store_true")
    run.add_argument("--vlm", action="store_true")
    run.add_argument("--vlm-timeout", type=float, default=30.0)
    run.add_argument("--vlm-verify-delta", type=int, default=14)
    run.add_argument("--vlm-min-confidence", type=float, default=0.65)
    run.add_argument("--max-wait-s", type=float, default=3.0)
    run.add_argument("--max-tries", type=int, default=3)
    run.add_argument("--lock-to-last-window", type=int, default=0)
    run.add_argument("--uniqueness-margin", type=float, default=0.0)
    run.add_argument("--no-failsafe", action="store_true")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)
    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
