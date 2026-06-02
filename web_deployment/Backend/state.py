"""
state.py — Per-connection session state
=========================================
Each WebSocket connection gets its own SessionState instance.
This makes the system fully multi-user — two drivers on two
browser tabs will never share or corrupt each other's counters.

The session registry (sessions dict) is also used by the HTTP
endpoints (/history, /events, /reset) to look up a session
by the ID that was assigned when the WebSocket connected.
"""

from collections import deque
from dataclasses import dataclass, field

HISTORY_MAX_LEN = 120   # ~60 s at 2 fps


@dataclass
class SessionState:
    """
    Isolated state for a single driver session / WebSocket connection.
    Create one instance per connection; discard it when the client disconnects.
    """

    # Detection counters
    closed_counter:     int   = 0
    yawn_counter:       int   = 0
    no_face_counter:    int   = 0
    distracted_counter: int   = 0

    # Head-pose smoothing (exponential moving average)
    smooth_yaw:   float = 0.0
    smooth_pitch: float = 0.0

    # Alert tracking
    last_alert:       object = None
    last_alert_start: object = None

    # History buffers
    score_history: deque = field(
        default_factory=lambda: deque(maxlen=HISTORY_MAX_LEN)
    )
    event_log: list = field(default_factory=list)

    def reset(self) -> None:
        """Reset counters and history, keeping the session alive."""
        self.closed_counter = self.yawn_counter = 0
        self.no_face_counter = self.distracted_counter = 0
        self.smooth_yaw = self.smooth_pitch = 0.0
        self.last_alert = self.last_alert_start = None
        self.score_history.clear()
        self.event_log.clear()


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------
# Maps session_id (str) -> SessionState.
# Populated by ws_routes.py on connect; cleaned up on disconnect.
# HTTP routes use this to serve history/events for a specific session.
# ---------------------------------------------------------------------------
sessions: dict[str, "SessionState"] = {}
