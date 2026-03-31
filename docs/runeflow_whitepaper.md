# RuneFlow Whitepaper

## RuneFlow: AI-Native UI Automation Infrastructure

### Abstract
RuneFlow is an open-source framework for automating software through visual interfaces instead of APIs. It combines deterministic anchor-based matching with an optional vision-language fallback so agents can act on real software, including apps that expose no API.

### The problem
Most AI agent frameworks depend on tools, APIs, or browser DOM hooks. Real workflows are messier. Desktop apps, legacy systems, internal tools, and thick clients still dominate many high-value workflows. Traditional RPA often breaks because it depends on brittle coordinates, selectors, or rigid scripting.

### The RuneFlow approach
RuneFlow records clicks and keypresses while capturing a small anchor image around each click. During replay, it searches for that anchor on-screen, locates the best match across multiple scales and matching modes, then clicks at the stored relative offset inside the anchor.

### Core architecture
1. Record workflow events.
2. Capture anchor images for click steps.
3. Normalize events into a portable JSON format.
4. Replay using deterministic OpenCV matching.
5. If enabled, ask a VLM to propose a click location when deterministic matching fails.
6. Verify the proposal locally, then execute.

### Technical differentiators
- Anchor-based execution rather than pure XY replay
- Multi-pass matching using color, grayscale, and edges
- Scale sweep for slightly resized UI elements
- Uniqueness scoring to reject ambiguous matches
- Stubborn retry caps per step
- Recorded-XY failsafe when desired
- Optional VLM fallback through an OpenAI-compatible local endpoint

### Use cases
- Legacy enterprise software
- Trading terminals and thick clients
- Internal tools with no automation surface
- QA and regression automation
- Repetitive back-office workflows
- Personal automation on a local machine

### Open-source scope
The public repo should expose the execution core:
- recorder
- replay engine
- vision matcher
- VLM bridge
- JSON format
- CLI

The polished Qt desktop shell, license gate, and bundled assistant can remain private product layers.

### Position in the stack
LLM or agent framework -> RuneFlow -> real software UI.

### Why this matters
Open-source agent tooling has concentrated on planning and tool calling. RuneFlow attacks the execution problem directly. It turns software itself into the interface.

### Conclusion
RuneFlow is a practical bridge from AI reasoning to AI action. By grounding automation in visual anchors and pairing deterministic execution with optional model fallback, it opens a path toward reliable computer-use automation in the real world.
