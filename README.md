# personal-ai-trading-copilot

Personal AI Trading Copilot — 个人 A 股交易决策与审计系统（V1.2.2，统一发布版本 `1.2.2`，Windows 本地运行）。

V1.2.2 为日常可用性更新：按「每天收盘后看一次、第二天按单执行」设计——默认进入「今日 · 账户」，执行单只依据收盘价，给出可直接在券商 App 设置的限价买入与止损条件单，并可勾选执行或放弃；持仓并入观察池（只需成本价和日期）；全部输入可用股票名称；分区为今日 / 观察池 / 计划 / 设置；紧凑的交易日选择。详见 `docs/V1.2.2-发布说明.md`。

V1.2.1 为稳定性补丁：观察池复权数据并发获取、次日早上补写前一天收盘快照、快照来源记录不再被覆盖、交易前检查加入涨跌停判断、前端交互测试覆盖 V1.2 页面。详见 `docs/V1.2.1-发布说明.md`。

V1.2 把系统转向“个人波段交易体系的执行与审计”，在「我的账户」页形成每天可用的闭环：观察池（≤10只）→ 按个人波段规则逐只给出状态与依据 → 交易前检查（规则、可买手数、总仓位余量）→ 记录真实成交与持仓（与100万研究用模拟盘分开）→ 收盘后计划（新候选≤3只）。风控参数存于本地文件，可在页面修改；仓位按止损距离、整手和最低佣金计算。均线与MACD使用观察池股票的前复权日线。收盘后用全市场行情快照一次写入当日日线，不再逐只下载。详见 `docs/V1.2-发布说明.md` 与 `docs/V1.2-验收清单.md`。

V1.1 聚焦交易与数据一致性：批次 T+1、账户原子保存与跨进程保护、成交/决策记录幂等、统一账户估值/AI、补数退避恢复、回测成交与统计修正。新增运行状态和窄屏体验，修正后的引擎已离线重算真实行情样本，详见 `docs/V1.1-release-notes.md` 和 `docs/V1.1-research-report.md`。

本版修正盘前默认空白、半成品历史日期、跨日结论混用、补数缓存延迟和指数缺失。网页默认展示最近覆盖充分的日线，可切换最近行情；模拟交易继续使用真实当前交易日。详见 `docs/V1.1-数据显示修复.md`。

本版同时完成跨日订单恢复、AI查询与机会计划统一、数据状态和首装流程优化；提供锁定依赖、独立离线演示和带校验清单的源码发布包。发布说明见 `docs/V1.1-发布说明.md`，后续路线见 `docs/项目路线图.md`。开发期间的修复统一纳入 `1.1`，Git标签为 `v1.1`；下载附件及发布信息以[V1.1官方发布页](https://github.com/stevenrenOvO/personal-ai-trading-copilot/releases/tag/v1.1)为准。

核心闭环：

```
A 股行情 → 市场状态 → 情绪周期 → 强势板块 → 个股机会 → 策略信号
→ 风控 → 模拟交易 → 决策日志 → 决策审计 → 个人交易画像 → AI 复盘
```

## 快速开始

已安装依赖时，双击 `start.cmd` 启动；双击 `stop.cmd` 停止。首次解压请先安装 Python（本次验证 Windows 64 位 Python 3.13，最低 3.11），双击 `setup.cmd` 安装锁定的35个运行依赖，再双击 `start-demo.cmd` 离线体验。演示使用8001端口、固定样例日期及独立 `data/demo/` 账户，页面明确标记测试数据；`stop-demo.cmd` 停止演示。

真实模式使用8000端口；演示数据不迁入真实账户。首次初始化真实数据请参照下方命令，之后后台会补日线。源码包不附带用户账户、密钥和真实行情库。

```powershell
# 开发者安装：创建环境并安装运行+测试依赖
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

手机验收时，将手机和电脑连接同一 Wi-Fi，先停止已有服务，再在项目目录运行：

```powershell
.\.venv\Scripts\python.exe -m backend.api.app --host 0.0.0.0 --port 8000
```

手机访问 `http://电脑的局域网IP:8000/`。此方式在终端前台运行，按 Ctrl+C 停止；默认一键启动仍使用本机地址。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

联网测试要求显式启用 `RUN_LIVE_DATA_TESTS=1` 并具备相应凭证。前端交互回归需要 Node.js；没有 Node 时这些交互测试会跳过（也可借用 VS Code 自带运行时：设置 `ELECTRON_RUN_AS_NODE=1` 后用 `Code.exe` 代替 `node`）。日常运行无需 Node。

## 主要接口

- `/health`（服务存活） `/ready`（账户、行情、历史和补数状态）
- `/market/state` `/emotion/state` `/sectors`
- `/opportunities` `/board/environment` `/risk`
- `/positions` `/strategies` `/strategies/health`
- `/trades`（GET/POST） `/trades/{id}/audit`（POST）
- `/profile` `/backtest/runs`（GET/POST） `/paper/trade`（POST）
- `/ai/copilot` `/board/ladder` `/data/latest` `/data/history/dates`
- `/opportunities/candidates` `/opportunities/setups` `/opportunities/plan`
- `/data/ingest/status`
- V1.2 我的账户：`/personal/today`（执行单） `/personal/today/mark`（POST） `/personal/stocks`（名称查找） `/personal/watchlist/hold` `/personal/watchlist/sell`（POST） `/personal/usage`（本机使用记录） `/personal/risk-profile`（GET/POST） `/personal/sizing` `/personal/watchlist`（GET/POST）
  `/personal/watchlist/remove`（POST） `/personal/watchlist/status` `/personal/pretrade`
  `/personal/account` `/personal/trades`（GET/POST） `/personal/trades/void`（POST） `/personal/plan`（GET/POST）

> 早期骨架的 `/api/*` fixture 路由（`/api/backtest`、`/api/paper/snapshot`、`/api/profile`、
> `/api/opportunity/scan`）已在 V1.0 收尾时下线，现返回 404；它们曾以 `tests/fixtures` 的
> 样例数据作为数据源，与"不把样例当真实数据"的原则冲突。

## 数据说明

- `TushareProvider`：需要 `.env` 中的 `TUSHARE_TOKEN`。
- `BaoStockProvider`：免费、免 token，V1.1 默认数据源，提供 A 股历史日线 / 基础信息 / 行业 / 日历 / 复权因子（盘后数据，非实时）。
- `AkShareProvider`：免费，作为补充/降级，依赖第三方公开接口，稳定性弱于 BaoStock。
- 生产启动优先选择本地全市场历史库，再选择已有 CSV；无本地数据时使用延迟连接的网络降级链（已配置的 Tushare → BaoStock → AkShare），不会自动使用 Fixture。
- `FixtureProvider`：离线测试数据，位于 `tests/fixtures`，仅用于开发/测试，不作为生产真实数据。
- 演示模式必须显式设置 `COPILOT_DEMO_MODE=1`，界面标记为测试样例。
- 落地管线：`provider → raw → quality check → normalized → local storage`。

## 真实数据落地

首次使用真实模式，可先初始化全市场证券名单和日历（日期区间按需要修改）：

```powershell
.\.venv\Scripts\python.exe -m backend.data.ingest --universe all --start 20260701 --end 20261023 --metadata-only
.\start.ps1
```

该命令需要真实数据源联网成功；之后在顶部查看后台日线补数进度。初次下载可能需要较长时间，先展示达到覆盖门槛的最新日期。失败会保留实际缺口并重试。需要直接落地整个全市场区间时，去掉 `--metadata-only`，结果也会保存到可恢复的历史库。

```powershell
# 拉取 CSI300 成分股真实日线到本地 data/ 目录
.\.venv\Scripts\python.exe -m backend.data.ingest --universe hs300 --start 20260701 --end 20260908
```

服务启动时若检测到本地已落地数据，会优先使用本地数据（快速离线），否则回退到免费网络数据源。

## 构建发布包

```powershell
.\.venv\Scripts\python.exe scripts/build_release.py --smoke
```

输出源码 ZIP、SHA256和逐文件清单，流程沿用 `docs/V1.1-发布流程.md`（版本号以 `VERSION` 为准）。构建不会推送Git或上传Release。本项目发布目标是个人本地工具；公网服务的身份验证、账户隔离和生产部署属于后续独立工作。

模拟交易必须填写正价格与正整数数量。API 使用当前交易日并检查标的和涨跌停约束；当前日的验证数据缺失时会明确拒绝，历史重放请使用独立账户。账户仍使用兼容的 `data/paper/state.json`，新格式增加批次、幂等请求和可恢复成交事件，不能将新格式回写给旧服务。

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
