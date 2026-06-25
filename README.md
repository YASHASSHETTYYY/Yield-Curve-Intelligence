# 📈 Yield Curve Intelligence Dashboard

> **A fixed-income analytics dashboard built on FRED's T10Y2Y data — tracking yield curve dynamics, bond price sensitivity, rate trends, and credit stress from 2021 to 2026.**

---

## What This Dashboard Does (And Why It Matters)

The **10Y−2Y Treasury spread** (T10Y2Y) is probably the single most-watched macro signal on Wall Street. When the 10-year yield is *higher* than the 2-year, markets expect economic growth — that's the normal, upward-sloping yield curve. When it *inverts* (2Y > 10Y), it tells you that short-term borrowing costs are punishing relative to long-term expectations, and historically, that's preceded every U.S. recession in the last 50 years, with a 12–18 month lag.

This dashboard makes that signal — and a full suite of related fixed-income metrics — interactive and explorable.

---

## Dataset

| Field | Detail |
|---|---|
| **Source** | FRED (Federal Reserve Economic Data), St. Louis Fed |
| **Series** | `T10Y2Y` — 10-Year Treasury minus 2-Year Treasury Constant Maturity Rate |
| **Frequency** | Daily (business days) |
| **Date Range** | June 2021 → June 2026 |
| **Missing Values** | ~20 rows (federal holidays) — dropped during load, not imputed |

The raw file is one column: `T10Y2Y` (in percentage points, not basis points). A spread of `1.23` means the 10Y yields 123bps more than the 2Y. A spread of `-1.07` means the curve is deeply inverted by 107bps.

---

## Feature Engineering

All derived columns are computed in `load_data()` and cached:

| Column | Logic |
|---|---|
| `spread_30d_ma` | 30-day rolling mean of spread |
| `spread_90d_ma` | 90-day rolling mean of spread |
| `yoy_change` | `.diff(252)` — change vs ~1 trading year ago |
| `volatility` | 30-day rolling standard deviation |
| `regime` | Rule-based bucketing: `< −0.5` = Deeply Inverted → `≥ 0.25` = Normal |
| `rate_2y` | **Synthetic proxy** — linear interpolation across known Fed cycle waypoints + Gaussian noise |
| `rate_10y` | `rate_2y + spread` (exact by definition) |
| `ig_spread` | **Model-derived** — IG proxy inversely correlated with spread |
| `hy_spread` | **Model-derived** — HY proxy with higher sensitivity to inversion depth |
| `oas_spread` | Simple average of IG and HY proxies |

> ⚠️ **Analyst Note:** `rate_2y`, `ig_spread`, and `hy_spread` are **synthetic approximations** built from the T10Y2Y spread alone. In a production environment you'd pull `DGS2`, `DGS10`, `BAMLC0A0CM` (IG OAS), and `BAMLH0A0HYM2` (HY OAS) separately from FRED. The proxies here preserve directional accuracy and correlation structure, not absolute level accuracy.

---

## Dashboard Structure

### Tab 1 — Yield Curve Spread

**What you see:**
- The full daily T10Y2Y spread as a filled area chart, color-coded blue (positive) with red-shaded inversion zones
- 30-day and 90-day moving averages overlaid (toggleable)
- Rolling 30-day volatility as a sub-chart
- Regime distribution donut chart (Deeply Inverted / Inverted / Flat / Normal)

**Key moments in the data:**
- **Jun 2021**: Spread near +1.2% — normal post-COVID steepness, market pricing in recovery
- **Mar 2022 → Jul 2022**: Rapid flattening as the Fed began its aggressive rate hike cycle
- **Jul 2022 → Oct 2023**: Deep and sustained inversion, peaking at **−1.07%** on March 7, 2023 — the most extreme since 1981
- **Late 2023 → 2025**: Gradual re-steepening as markets priced Fed cuts
- **2025–2026**: Spread stabilizes in the +0.4–0.7% range (flat-to-normal regime)

**Sidebar toggle — Shade Regime Zones:** Adds red vrect fills wherever the spread was negative, making inversion episodes immediately visible.

---

### Tab 2 — Bond Price vs Yield

**What you see:**
- A bond price–yield curve for a configurable 10-year coupon bond (par value and coupon adjustable from sidebar)
- A diamond marker showing where the current 10Y yield sits on that curve
- A green dashed line at the coupon rate (where price = par)
- Modified Duration chart across the yield range
- A price sensitivity table: what happens to bond price if yields shift ±0.5%, ±1%, ±2%

**The core fixed-income concept:**
When you buy a bond, you lock in a coupon. If new bonds come out paying more (because yields rose), your bond's price falls so its effective yield matches the market. This inverse relationship is **non-linear** — it's convex. That means:
- A 1% yield *rise* hurts price less than a 1% yield *fall* helps it
- Duration stretches at low yields (more sensitive)
- Duration compresses at high yields (less sensitive)

**Analyst use case:** Use the sidebar coupon slider to model a specific bond you hold. Change par value to reflect your position size. The sensitivity table gives you instant P&L scenarios for FOMC surprise moves.

---

### Tab 3 — Interest Rate Trend Tracking

**What you see:**
- Dual-axis line chart: 2Y and 10Y yields on the left axis, spread overlaid on the right
- Year-over-year spread change heatmap (months × years, green = steepening, red = flattening)
- Spread distribution histogram with mean and zero-line markers
- Key statistics table: current level, 90D high/low, all-time extremes, % of days inverted

**The heatmap insight:**  
Reading across a row (one year) tells you seasonality. Reading down a column (one month across years) shows the macro trajectory. The deep red cluster in 2022–2023 shows how rapidly the curve inverted during the Fed's tightening campaign — a move that took months rather than years.

**Regime classification logic:**

```python
if spread < -0.5:   "Deeply Inverted"   # Recession signal
elif spread < 0:    "Inverted"           # Warning zone
elif spread < 0.25: "Flat"              # Neutral / transitional
else:               "Normal"            # Healthy growth expectations
```

---

### Tab 4 — Credit Spread Monitoring

**What you see:**
- IG and HY credit spread time series with a blended OAS overlay
- Scatter plot: HY spread vs T10Y2Y spread (shows the inverse relationship + linear trend)
- Grouped bar chart: quarterly average IG vs HY spreads (last 12 quarters)
- Composite credit risk gauge (0–100)

**Why yield curve + credit spreads go together:**
Yield curve inversion tightens net interest margins for banks (they borrow short, lend long), reduces lending appetite, and signals that growth is decelerating. That stress feeds into credit markets — when corporates are more likely to default, investors demand higher spreads above Treasuries to hold their bonds. The scatter plot in this tab makes that inverse relationship visible: deeper inversion → wider credit spreads.

**The risk gauge:**  
Composite score derived from current HY spread level and inversion depth. Thresholds:
- `< 30`: Low risk (green)
- `30–60`: Elevated risk (amber)  
- `> 60`: High risk (red)

---

## How to Run

### Prerequisites

```bash
pip install streamlit plotly pandas numpy
```

### Launch

```bash
streamlit run yield_dashboard.py
```

The app opens at `http://localhost:8501`. Ensure `T10Y2Y.csv` is in the same directory as the script.

### File Structure

```
.
├── yield_dashboard.py    # Main Streamlit app
├── T10Y2Y.csv            # FRED T10Y2Y dataset
└── README.md             # This file
```

---

## Sidebar Controls

| Control | Effect |
|---|---|
| **Date Range (Quick Select)** | Filter all charts to Last 1Y / 2Y / 3Y or All Data |
| **Custom Date Range** | Fine-grained from/to date pickers |
| **Show Moving Averages** | Toggle 30D and 90D MA lines on spread chart |
| **Shade Regime Zones** | Toggle red vrect shading for inversion periods |
| **Bond Coupon Rate** | Slider (1–8%) — recalculates all bond pricing charts |
| **Par Value** | Dollar amount — scales bond price and P&L table |

---

## Known Limitations & Production Gaps

1. **Synthetic 2Y/10Y rates** — derived from the spread alone, not actual DGS2/DGS10. Directionally correct, not quote-accurate.
2. **Proxy credit spreads** — IG/HY modeled from spread dynamics. For real credit monitoring, source BAMLC0A0CM and BAMLH0A0HYM2 from FRED.
3. **Bond price model** — assumes annual coupon payments, flat yield curve, no accrued interest. A production model would use day-count conventions and a full discount curve.
4. **No real-time feed** — data is a static CSV snapshot. For live dashboards, connect to FRED API (`fredapi` Python package) or a Bloomberg/Refinitiv feed.
5. **Single-tenor view** — this only tracks 10Y−2Y. A full yield curve dashboard would add 3M, 1Y, 5Y, 7Y, 20Y, 30Y tenors and plot the full curve shape over time.

---

## Technologies Used

| Library | Role |
|---|---|
| `streamlit` | Web app framework, sidebar, tabs, layout |
| `pandas` | Data loading, cleaning, feature engineering, rolling stats |
| `numpy` | Vectorized bond math, interpolation, synthetic rate generation |
| `plotly` | All interactive charts (area, scatter, bar, heatmap, gauge, table) |

---

*Built on FRED T10Y2Y data · Yashas Shetty · RAIT DY Patil University*
