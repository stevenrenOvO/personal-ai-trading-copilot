# personal-ai-trading-copilot

Personal AI Trading Copilot — 个人 A 股交易决策与审计系统（V1.0）。

核心闭环：

```
A 股行情 → 市场状态 → 情绪周期 → 强势板块 → 个股机会 → 策略信号
→ 风控 → 模拟交易 → 决策日志 → 决策审计 → 个人交易画像 → AI 复盘
```

## 快速开始

```powershell
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 可选：配置真实数据源（token 只从 .env 读取，绝不写入代码/日志/git）
Copy-Item .env.example .env
# 编辑 .env，填入 TUSHARE_TOKEN

# 3. 启动服务
.\.venv\Scripts\python.exe -m backend.api.app

# 4. 打开浏览器
http://127.0.0.1:8000/
```

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

没有 `TUSHARE_TOKEN` 时，真实数据测试会自动 SKIP，但离线 Fixture 测试始终运行。

## 主要接口

- `/health`
- `/market/state` `/emotion/state` `/sectors`
- `/opportunities` `/board/environment` `/risk`
- `/positions` `/strategies` `/strategies/health`
- `/trades`（GET/POST） `/trades/{id}/audit`（POST）
- `/profile` `/backtest/runs`（GET/POST） `/paper/trade`（POST）
- `/ai/copilot` `/board/ladder` `/data/latest` `/data/history/dates`
- `/opportunities/candidates` `/opportunities/setups` `/opportunities/plan`

> 早期骨架的 `/api/*` fixture 路由（`/api/backtest`、`/api/paper/snapshot`、`/api/profile`、
> `/api/opportunity/scan`）已在 V1.0 收尾时下线，现返回 404；它们曾以 `tests/fixtures` 的
> 样例数据作为数据源，与"不把样例当真实数据"的原则冲突。

## 数据说明

- `TushareProvider`：需要 `.env` 中的 `TUSHARE_TOKEN`。
- `BaoStockProvider`：免费、免 token，V1.1 默认数据源，提供 A 股历史日线 / 基础信息 / 行业 / 日历 / 复权因子（盘后数据，非实时）。
- `AkShareProvider`：免费，作为补充/降级，依赖第三方公开接口，稳定性弱于 BaoStock。
- `FallbackProvider`：按顺序自动降级（Tushare → 本地存储 → BaoStock → AkShare → Fixture）。
- `FixtureProvider`：离线测试数据，位于 `tests/fixtures`，仅用于开发/测试，不作为生产真实数据。
- 落地管线：`provider → raw → quality check → normalized → local storage`。

## 真实数据落地

```powershell
# 拉取 CSI300 成分股真实日线到本地 data/ 目录
.\.venv\Scripts\python.exe -m backend.data.ingest --universe hs300 --start 20260701 --end 20260908
```

服务启动时若检测到本地已落地数据，会优先使用本地数据（快速离线），否则回退到免费网络数据源。

## 目录结构

```
backend/
  ai/           AI 解释与 Tool-based Agent
  api/          WSGI API（标准库实现）
  backtest/     回测 + A 股规则 + Walk-Forward
  board/        涨停/连板环境
  data/         数据 schema / provider / 日历 / 管线
  emotion/      情绪周期
  journal/      决策日志 / 审计
  market/       市场状态
  opportunity/  机会引擎（六级过滤）
  paper/        模拟盘
  profile/      个人交易画像
  risk/         风控
  sector/       板块强度
  strategies/   策略（strong_sector_breakout 为正式第一策略，ma_cross 为 baseline）
```
