"""AI explanation and tool-based agent layer.

All explanations are grounded in system data. When no LLM key is configured
the agent degrades gracefully to deterministic, data-driven answers instead of
inventing market data or positions.
"""

from backend.ai.agent import AICopilot

__all__ = ["AICopilot"]
