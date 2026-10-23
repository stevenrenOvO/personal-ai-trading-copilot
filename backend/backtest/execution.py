"""Shared conservative fill constraints for paper and research orders."""
import math


def rejection_reason(action, price, limits=None, *, suspended=False, bar=None):
    if not math.isfinite(price) or price <= 0:
        return "成交价格无效"
    if suspended:
        return "停牌不可成交"
    if limits:
        upper = float(limits.get("up_limit", float("nan")))
        lower = float(limits.get("down_limit", float("nan")))
        if not math.isfinite(upper) or not math.isfinite(lower):
            return "涨跌停价格不可验证"
        if price >= upper - 1e-8 and action == "BUY":
            return "涨停不可买入"
        if price <= lower + 1e-8 and action == "SELL":
            return "跌停不可卖出"
        if price > upper + 1e-8 or price < lower - 1e-8:
            return "价格超出涨跌停范围"
    if bar and action == "BUY" and limits:
        if float(bar.get("low", 0)) >= float(limits["up_limit"]) - 1e-8:
            return "一字涨停不可买入"
    return None
