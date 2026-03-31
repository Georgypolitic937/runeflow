from dataclasses import dataclass, field
from typing import Sequence

@dataclass
class RecorderConfig:
    anchor_size: int = 48
    default_confidence: float = 0.90
    default_timeout: float = 4.0

@dataclass
class VLMConfig:
    enabled: bool = False
    timeout_secs: float = 30.0
    verify_delta: int = 14
    min_confidence: float = 0.65
    roi_half_width: int = 500

@dataclass
class ReplayConfig:
    fallback_to_absolute: bool = True
    move_duration_range: tuple[float, float] = (0.05, 0.15)
    poll_sleep: float = 0.12
    scale_sweep: Sequence[float] = field(default_factory=lambda: [0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
    edge_low_high: tuple[int, int] = (50, 150)
    debug_match: bool = True
    verify_enabled: bool = False
    max_anchor_stubborn_s: float = 3.0
    max_anchor_tries: int = 3
    lock_to_last_window: int = 0
    uniqueness_margin: float = 0.0
