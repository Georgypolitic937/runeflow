# RuneFlow

RuneFlow is an open-source visual automation core for AI agents.

It records clicks and keystrokes, captures small anchor images around clicks, and replays workflows by locating those anchors on-screen instead of trusting raw coordinates. When enabled, a vision-language model can propose click locations as a fallback when deterministic matching fails.

## What this repo includes

- anchor-based recorder
- OpenCV replay engine
- optional VLM fallback
- JSON recording format
- CLI for recording and replay

## What is intentionally not included

This cut keeps the open-source core and leaves out the commercial desktop shell pieces from the original Qt app, including the offline license gate and assistant dock.

## Install

```bash
pip install -e .
```

## Record a workflow

```bash
runeflow record recordings/demo.json
```

Press `F8` to stop recording.

## Replay a workflow

```bash
runeflow run recordings/demo.json --loops 1 --jitter 2
```

## VLM fallback

```bash
runeflow run recordings/demo.json --vlm --vlm-timeout 30
```

The VLM bridge expects an OpenAI-compatible endpoint such as LM Studio:

- `LM_BASE`
- `LM_MODEL`
- `LM_API_KEY`

## JSON format

```json
{
  "version": "1.0",
  "events": [
    {
      "delay": 0.42,
      "type": "click",
      "x": 640,
      "y": 410,
      "button": "Button.left",
      "anchor": {
        "path": "assets/step_0000.png",
        "conf": 0.9,
        "timeout": 4.0,
        "ox": 24,
        "oy": 18,
        "w": 48,
        "h": 48
      }
    }
  ]
}
```

## Repo layout

- `runeflow/recorder.py` — lightweight recorder extracted from the Qt app
- `runeflow/vision.py` — deterministic anchor matching
- `runeflow/vlm.py` — OpenAI-compatible vision fallback
- `runeflow/runner.py` — replay engine
- `runeflow/exporters.py` — standalone runner template
- `runeflow/cli.py` — command line entrypoint

## Launch positioning

RuneFlow is best positioned as the execution layer for AI agents:
LLM or agent framework -> RuneFlow -> real software.
