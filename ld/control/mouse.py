"""Mouse control — stub in offline mode; live in Phase 7."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MouseController:
    """Offline: log targets only. Live: move OS cursor (Phase 7)."""

    enabled: bool = False
    dry_run: bool = True

    def move_to(self, x: float, y: float) -> None:
        if not self.enabled or self.dry_run:
            return
        # Phase 7: pyautogui / pydirectinput
        raise NotImplementedError("Live mouse control is Phase 7")
