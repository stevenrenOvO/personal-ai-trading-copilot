"""Tool-based AI copilot.

The agent exposes tool functions that read system state, then produces an
answer only from tool results. If an LLM API key is absent, it falls back to
deterministic summarization and clearly reports ``AI unavailable`` for
open-ended questions rather than hallucinating.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

from backend.ai.context import build_context
from backend.data.config import get_settings
from backend.data.providers.base import DataProvider
from backend.strategies.base import Strategy


@dataclass
class ToolResult:
    name: str
    data: Any


class AICopilot:
    """Tool-based agent grounded in system data."""

    def __init__(
        self,
        provider: Optional[DataProvider] = None,
        strategy: Optional[Strategy] = None,
        context_cache: Optional[object] = None,
    ) -> None:
        self.provider = provider
        self.strategy = strategy
        # Building the context runs the whole engine stack; the API layer hands
        # in a short-TTL cache so repeated questions do not rescan the market.
        self._context_cache = context_cache
        self._tools: dict[str, Callable[..., Any]] = {}
        self._register_tools()

    def _register_tools(self) -> None:
        self._tools = {
            "get_market_state": self.get_market_state,
            "get_emotion_state": self.get_emotion_state,
            "get_sector_strength": self.get_sector_strength,
            "get_opportunities": self.get_opportunities,
            "get_board_environment": self.get_board_environment,
            "get_positions": self.get_positions,
            "get_risk_status": self.get_risk_status,
            "get_strategy_health": self.get_strategy_health,
            "get_trade_history": self.get_trade_history,
            "audit_trade": self.audit_trade,
            "get_personal_profile": self.get_personal_profile,
        }

    def _ctx(self, date: str) -> dict[str, Any]:
        if self._context_cache is None:
            return build_context(date, provider=self.provider, strategy=self.strategy)
        return self._context_cache.get_or_compute(
            ("ai_context", date),
            lambda: build_context(
                date, provider=self.provider, strategy=self.strategy
            ),
        )

    def get_market_state(self, date: str) -> ToolResult:
        return ToolResult("get_market_state", self._ctx(date)["market"])

    def get_emotion_state(self, date: str) -> ToolResult:
        return ToolResult("get_emotion_state", self._ctx(date)["emotion"])

    def get_sector_strength(self, date: str) -> ToolResult:
        return ToolResult("get_sector_strength", self._ctx(date)["sectors"])

    def get_opportunities(self, date: str) -> ToolResult:
        return ToolResult("get_opportunities", self._ctx(date)["opportunities"])

    def get_board_environment(self, date: str) -> ToolResult:
        return ToolResult("get_board_environment", self._ctx(date)["board"])

    def get_positions(self, date: str) -> ToolResult:
        return ToolResult("get_positions", self._ctx(date)["positions"])

    def get_risk_status(self, date: str) -> ToolResult:
        return ToolResult("get_risk_status", self._ctx(date)["risk"])

    def get_strategy_health(self, date: str) -> ToolResult:
        return ToolResult("get_strategy_health", self._ctx(date)["strategies"])

    def get_trade_history(self, date: str) -> ToolResult:
        return ToolResult("get_trade_history", self._ctx(date)["trades"])

    def audit_trade(self, date: str) -> ToolResult:
        return ToolResult("audit_trade", self._ctx(date)["audit"])

    def get_personal_profile(self, date: str) -> ToolResult:
        return ToolResult("get_personal_profile", self._ctx(date)["profile"])

    def tools(self) -> dict[str, Callable[..., Any]]:
        return dict(self._tools)

    def call_tool(self, name: str, date: str) -> ToolResult:
        if name not in self._tools:
            return ToolResult(name, {"error": f"unknown tool: {name}"})
        return self._tools[name](date)

    def daily_briefing(self, date: str) -> dict[str, Any]:
        """Data-grounded daily briefing without any LLM dependency."""
        ctx = self._ctx(date)
        return {
            "date": date,
            "can_trade_today": self._can_trade_today(ctx),
            "emotion": ctx["emotion"]["emotion_cycle"],
            "top_sectors": [
                {"name": s["name"], "score": s["score"]} for s in ctx["sectors"][:3]
            ],
            "top_opportunities": [
                {
                    "ts_code": o["ts_code"],
                    "name": o["name"],
                    "score": o["score"],
                    "suggested_action": o["suggested_action"],
                    "reason": o["stock_reason"],
                }
                for o in ctx["opportunities"][:5]
            ],
            "risk": ctx["risk"],
            "board": {
                "grade": ctx["board"]["grade"],
                "limit_up_count": ctx["board"]["limit_up_count"],
            },
            "recommended_exposure": ctx["emotion"]["recommended_exposure"],
        }

    def ask(self, query: str, date: str) -> dict[str, Any]:
        """Answer a natural-language question using tools first."""
        q = query.strip().lower()
        if any(k in q for k in ("市场", "market", "能不能做")):
            result = self.get_market_state(date)
        elif any(k in q for k in ("情绪", "emotion", "周期")):
            result = self.get_emotion_state(date)
        elif any(k in q for k in ("板块", "sector")):
            result = self.get_sector_strength(date)
        elif any(k in q for k in ("机会", "opportunity", "关注")):
            result = self.get_opportunities(date)
        elif any(k in q for k in ("涨停", "连板", "board")):
            result = self.get_board_environment(date)
        elif any(k in q for k in ("风险", "risk")):
            result = self.get_risk_status(date)
        elif any(k in q for k in ("持仓", "position")):
            result = self.get_positions(date)
        elif any(k in q for k in ("交易", "trade", "history")):
            result = self.get_trade_history(date)
        elif any(k in q for k in ("审计", "audit")):
            result = self.audit_trade(date)
        elif any(k in q for k in ("画像", "profile")):
            result = self.get_personal_profile(date)
        else:
            return {
                "answer": self._llm_fallback(query, date),
                "tool_calls": [],
                "ai_available": self._llm_available(),
            }

        return {
            "answer": f"已通过 {result.name} 获取数据",
            "tool_calls": [result.name],
            "data": result.data,
            "ai_available": False,
        }

    def _can_trade_today(self, ctx: dict[str, Any]) -> bool:
        market = ctx["market"]
        emotion = ctx["emotion"]
        if market["state"] in ("数据不足", "冰点"):
            return False
        if emotion["emotion_cycle"] in ("冰点", "退潮"):
            return False
        return market["score"] >= 45

    def _llm_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY"))

    def _llm_fallback(self, query: str, date: str) -> str:
        if not self._llm_available():
            return "AI unavailable: 未配置 LLM API Key，无法进行开放式问答。请使用结构化工具查询市场/情绪/板块/机会/风险/持仓等。"
        return "Data unavailable: LLM 后端未实现或不可用。"
