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
        runtime: Optional[object] = None,
    ) -> None:
        self.provider = provider
        self.strategy = strategy
        # Building the context runs the whole engine stack; the API layer hands
        # in a short-TTL cache so repeated questions do not rescan the market.
        self._context_cache = context_cache
        self.runtime = runtime
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
            return build_context(date, provider=self.provider, strategy=self.strategy, runtime=self.runtime)
        paper = getattr(self.runtime, "paper", None)
        version = paper.snapshot().version if paper is not None else 0
        return self._context_cache.get_or_compute(
            ("ai_context", date, version),
            lambda: build_context(
                date, provider=self.provider, strategy=self.strategy, runtime=self.runtime
            ),
        )

    def get_market_state(self, date: str) -> ToolResult:
        ctx = self._ctx(date)
        market = dict(ctx["market"])
        meta = ctx.get("data_meta", {}).get("market", {})
        plan = ctx.get("trade_plan", {})
        market.update(source=meta.get("source"),
            data_kind="日线统计" if meta.get("kind") in ("historical", "historical_scan") else "行情快照",
            analysis_basis_date=plan.get("basis_date") or plan.get("date"),
            can_attack=self._trade_permission(ctx), attack_note=plan.get("attack_note", ""))
        return ToolResult("get_market_state", market)

    def get_emotion_state(self, date: str) -> ToolResult:
        return ToolResult("get_emotion_state", self._ctx(date)["emotion"])

    def get_sector_strength(self, date: str) -> ToolResult:
        return ToolResult("get_sector_strength", self._ctx(date)["sectors"])

    def get_opportunities(self, date: str) -> ToolResult:
        ctx = self._ctx(date)
        plan = dict(ctx.get("trade_plan") or {})
        plan.update(requested_date=date, basis_date=plan.get("basis_date") or plan.get("date"),
            can_attack=self._trade_permission(ctx), source="机会页交易计划")
        return ToolResult("get_opportunities", plan)

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
        current_basis = ctx["emotion"].get("date") == date
        permission = self._trade_permission(ctx)
        plan = ctx.get("trade_plan", {})
        return {
            "date": date,
            "basis_date": ctx["emotion"].get("date"),
            "basis_note": ctx["emotion"].get("basis_note", ""),
            "can_trade_today": permission,
            "attack_note": plan.get("attack_note", ""),
            "market_source": ctx.get("data_meta", {}).get("market", {}).get("source"),
            "analysis_basis_date": plan.get("basis_date") or plan.get("date"),
            "emotion": ctx["emotion"]["emotion_cycle"],
            "top_sectors": [
                {"name": s["name"], "score": s["score"]} for s in ctx["sectors"][:3]
            ],
            "top_opportunities": [
                {
                    "ts_code": o["ts_code"],
                    "name": o["name"],
                    "score": o["score"],
                    "suggested_action": o.get("state", "观察"),
                    "tier": o.get("tier", ""),
                    "reason": o.get("why_not_buy") or "；".join(o.get("reasons", [])),
                }
                for o in ([item for item in plan.get("plans", []) if item.get("tier") in ("A", "B")][:5]
                          if current_basis and permission is not None else [])
            ],
            "risk": ctx["risk"],
            "board": {
                "grade": ctx["board"]["grade"],
                "limit_up_count": ctx["board"]["limit_up_count"],
            },
            "recommended_exposure": plan.get("total_exposure_cap") if current_basis and permission is not None else None,
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
        elif any(k in q for k in ("审计", "audit")):
            result = self.audit_trade(date)
        elif any(k in q for k in ("策略", "健康", "strategy")):
            result = self.get_strategy_health(date)
        elif any(k in q for k in ("交易", "trade", "history")):
            result = self.get_trade_history(date)
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
        return self._trade_permission(ctx) is True

    def _trade_permission(self, ctx: dict[str, Any]) -> Optional[bool]:
        """Unknown current structures must not be represented as a negative signal."""
        market = ctx["market"]
        emotion = ctx["emotion"]
        if emotion.get("date") != ctx["date"]:
            return None
        if "trade_plan" in ctx:
            plan = ctx["trade_plan"]
            meta = ctx.get("data_meta", {}).get("trade_plan", {})
            if (plan.get("date") != ctx["date"] or plan.get("available") is False or "can_attack" not in plan
                    or plan.get("historical_only") or meta.get("status") in ("no_data", "warming", "error")):
                return None
            return bool(plan.get("can_attack"))
        if market["state"] in ("数据不足", "数据准备中", "当日数据准备中"):
            return None
        if market["state"] == "冰点":
            return False
        if emotion["emotion_cycle"] in ("冰点", "退潮"):
            return False
        return market["score"] >= 45

    def _llm_available(self) -> bool:
        return False  # Availability requires an implemented backend, not just a key.

    def _llm_fallback(self, query: str, date: str) -> str:
        if not self._llm_available():
            return "当前版本尚未接入开放式问答后端。请使用结构化查询：市场、情绪、板块、机会、风险、持仓、交易记录等。"
        return "Data unavailable: LLM 后端未实现或不可用。"
