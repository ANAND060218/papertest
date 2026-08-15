# V8 Frozen Forward-Testing Scientific Protocol

**Document Status:** IMMUTABLE / FROZEN  
**Effective Date:** 2026-08-14  
**Stage:** V8 — Frozen Forward Validation  
**Platform Path:** `c:\Users\anand\Desktop\trade\intra\`

---

## 1. Scientific Objective

Determine whether the frozen Machine Learning model (`xgb_intraday_5m.json`) maintains a positive mathematical expectancy and a statistically reliable edge on **genuinely unseen forward market data** under realistic Indian intraday execution conditions (including statutory costs, spread, and slippage).

---

## 2. Core Rule of the Forward Test

> **NO INTERVENTION. NO DAILY RETRAINING. NO THRESHOLD TUNING.**
>
> The model, threshold, risk limits, and setup rules are locked in [`data/production_config.json`](file:///c:/Users/anand/Desktop/trade/intra/data/production_config.json).
> If the system experiences a string of losing trades, the threshold will **NOT** be adjusted, features will **NOT** be added, and symbols will **NOT** be removed.
> The purpose of V8 is measurement and validation, not optimization.

---

## 3. Frozen Configuration & System Parameters

| Parameter | Locked Value | Rationale |
|:---|:---|:---|
| **Model Artifact** | `results/xgb_intraday_5m.json` | Trained on isolated pre-V8 dataset |
| **Probability Threshold** | **`P(win) >= 0.40`** | Selected strictly on validation partition |
| **Minimum Expected Value** | **`EV > 0.0%`** | Enforces positive economic expectancy |
| **Active Universe** | `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `HDFCBANK.NS`, `ICICIBANK.NS`, `SBIN.NS` | High liquidity Nifty large caps |
| **Active Setups** | `ORB` (15m), `VWAP_BREAKOUT`, `PREV_DAY_HIGH_BREAKOUT` | Intraday momentum & range breakouts |
| **Risk Allocation** | **1.0%** of equity per trade | Maximum ₹1,000 risk on ₹1,00,000 capital |
| **Position Sizing Cap** | **10.0%** of equity per trade | Maximum ₹10,000 position exposure |
| **Max Concurrent Positions** | **3** active positions | Limits portfolio correlation risk |
| **Daily Drawdown Limit** | **2.0%** of capital | Auto kill-switch activates on ₹2,000 day loss |
| **Compulsory Square-Off** | **15:15:00 IST** | Avoids broker penalty and overnight risk |

---

## 4. Execution Pipeline & Dual-Stream Logging

```
                          INCOMING 5M TICK / BAR
                                    │
                                    ▼
                         SETUP DETECTION ENGINE
                  (ORB / VWAP / Prev-Day High Breakout)
                                    │
                                    ▼
                          FEATURE ENGINE (11 FEATS)
                                    │
                                    ▼
                        FROZEN XGBOOST INFERENCE
                                    │
                                    ▼
                          P(win) >= 0.40 & EV > 0?
                                   / \
                                  /   \
                             YES /     \ NO
                                /       \
                               ▼         ▼
                     PAPER ORDER EXEC     REJECTED SETUP
                     (Slippage + Costs)   (Counterfactual)
                               │                 │
                               ▼                 ▼
                       `journal_trades`   `journal_rejected_setups`
                               └────────┬────────┘
                                        ▼
                           `v8_forward_report.py`
                           (Unbiased Performance Audit)
```

### Table 1: `journal_trades` (Executed Trades)
- `trade_id`, `created_at`, `symbol`, `setup_type`, `direction`, `regime`
- `entry_price` (planned), `fill_entry_price` (actual with slippage)
- `stop_price`, `target_price`, `quantity`
- `xgb_probability`, `ev_score`, `features_json`
- `exit_timestamp`, `exit_price` (planned), `fill_exit_price` (actual)
- `gross_pnl`, `total_costs` (brokerage + STT + stamp + GST + exchange + SEBI + slippage)
- `net_pnl`, `return_pct`, `exit_reason`, `bars_held`, `status`

### Table 2: `journal_rejected_setups` (Counterfactual Audit)
- `setup_id`, `created_at`, `symbol`, `setup_type`, `regime`
- `current_ltp`, `xgb_probability`, `ev_score`
- `rejection_reason` (e.g. `P(win) 0.32 < 0.40`, `Negative EV`, `Risk Blocked`)
- `features_json`

---

## 5. Minimum Evidence Requirements (Success Gates)

To make a scientifically defensible evaluation before any micro-capital live pilot (V10), the forward test must fulfill:

1. **Duration Gate**: Minimum **8 to 12 calendar weeks** of uninterrupted paper execution.
2. **Observation Gate**: Minimum **100 executed trade setups** (or all generated setups across 12 weeks if low frequency).
3. **Regime Coverage Gate**: Forward period must span at least **2 distinct market regimes** (e.g. Trending Bull + Sideways/High-Vol event).

---

## 6. Evaluation Metrics & Counterfactual Proof

The automated forward report generator (`v8_forward_report.py`) will evaluate:

### A. Strategy Quality
- **Win Rate (%)**, **Profit Factor (PF)**, **Expectancy per Trade (₹)**, **Max Drawdown (%)**, **Sharpe Ratio**.

### B. Execution Divergence (Backtest vs Reality)
- **Modeled Slippage vs Actual Realized Slippage**.
- **Modeled Transaction Costs vs Actual Broker-Equivalent Costs**.
- **Fill Quality**: `abs(fill_price - planned_price)`.

### C. Model Calibration Curve
- Probability buckets: `0.40–0.50`, `0.50–0.60`, `0.60–0.70`, `0.70+`.
- Predicted $P(\text{win})$ vs Actual Realized Win Rate in each bucket.

### D. The Ultimate Test: Counterfactual Analysis
- **Accepted Setups ($P \ge 0.40$) PF** vs **Rejected Setups ($P < 0.40$) PF**.
- *Hypothesis*: If the model is genuinely adding value, Accepted PF must significantly exceed Rejected PF ($PF_{\text{accepted}} > 1.0 > PF_{\text{rejected}}$).
