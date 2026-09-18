"""Paper trading layer for V1.0.

This module provides a lightweight broker simulator that executes strategy
signals against synthetic cash and positions. It is intentionally independent
of the live/provider network so it can run fully offline.
"""

from backend.paper.engine import PaperTradingEngine
from backend.paper.models import PaperOrder, PaperPosition, PaperSnapshot

__all__ = [
    "PaperOrder",
    "PaperPosition",
    "PaperSnapshot",
    "PaperTradingEngine",
]
