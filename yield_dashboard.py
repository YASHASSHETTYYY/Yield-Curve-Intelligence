import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yield Curve Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
DARK_BG      = "#060B14"  # Deeper terminal black
CARD_BG      = "#0F1623"
BORDER       = "#1E293B"
ACCENT_BLUE  = "#2563EB"  # Institutional blue
ACCENT_AMBER = "#D97706"
ACCENT_RED   = "#DC2626"
ACCENT_GREEN = "#059669"
MUTED        = "#64748B"
TEXT_PRIMARY = "#E2E8F0"
TEXT_SEC     = "#94A3B8"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=JetBrains+Mono:wght@400&family=Space+Grotesk:wght@500;600&display=swap');

html, body, [class*="css"] {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Inter', sans-serif;
}}

/* sidebar */
section[data-testid="stSidebar"] {{
    background-color: {DARK_BG} !important;
    border-right: 1px solid {BORDER} !important;
}}

/* cards */
.kpi-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-left: 2px solid var(--accent-color, {BORDER});
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 4px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
}}
.kpi-label {{
    font-size: 11px;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: {MUTED};
    font-family: 'Inter', sans-serif;
}}
.kpi-value {{
    font-size: 28px;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    line-height: 1.2;
    margin-top: 4px;
}}
.kpi-delta {{
    font-size: 12px;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 4px;
    color: {TEXT_SEC};
}}
.section-header {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    font-size: 14px;
    color: {TEXT_SEC};
    margin: 32px 0 16px 0;
}}
.insight-box {{
    background: {CARD_BG};
    border-left: 3px solid {ACCENT_BLUE};
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 13px;
    color: {TEXT_SEC};
    margin: 8px 0 18px 0;
    line-height: 1.6;
}}
.warning-box {{
    background: {CARD_BG};
    border-left: 3px solid {ACCENT_RED};
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 13px;
    color: {TEXT_SEC};
    margin: 8px 0 18px 0;
}}
/* plotly container border */
.stPlotlyChart > div {{
    border-radius: 10px;
    border: 1px solid {BORDER};
    overflow: hidden;
}}
/* tab bar font */
[data-testid="stTab"] button {{
    font-size: 13px;
}}
/* hide streamlit menu */
#MainMenu, footer {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, JetBrains Mono, sans-serif", color=TEXT_PRIMARY, size=12),
    margin=dict(l=20, r=20, t=40, b=30),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, showspikes=True,
               spikecolor=MUTED, spikethickness=1),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    hoverlabel=dict(bgcolor="#0F1623", bordercolor=ACCENT_BLUE, font_color=TEXT_PRIMARY),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["observation_date"])
    df.rename(columns={"observation_date": "date", "T10Y2Y": "spread"}, inplace=True)
    df = df.dropna(subset=["spread"]).sort_values("date").reset_index(drop=True)

    # ── Derived columns
    df["spread_30d_ma"] = df["spread"].rolling(30).mean()
    df["spread_90d_ma"] = df["spread"].rolling(90).mean()
    df["yoy_change"]    = df["spread"].diff(252)          # ~1 trading year
    df["volatility"]    = df["spread"].rolling(30).std()
    df["regime"] = df["spread"].apply(
        lambda x: "Deeply Inverted" if x < -0.5
        else "Inverted" if x < 0
        else "Flat" if x < 0.25
        else "Normal"
    )

    # ── Synthetic 2Y & 10Y from spread + reasonable anchors
    # Rough approximation: use spread alone to back-compute.
    # We anchor 2Y to a plausible Fed path proxy and derive 10Y.
    # (In a real setup you'd load DGS2 + DGS10 separately.)
    np.random.seed(42)
    # Approximate 2Y from a linear declining hike cycle shape
    n = len(df)
    base_2y = np.interp(np.arange(n),
                        [0, 300, 600, 800, n-1],
                        [0.25, 1.0, 4.5, 4.8, 4.1])
    noise = np.random.normal(0, 0.04, n)
    df["rate_2y"] = np.round(np.clip(base_2y + noise, 0.05, 5.5), 2)
    df["rate_10y"] = np.round(df["rate_2y"] + df["spread"], 2)

    # ── Bond price model (10Y par bond, annual coupon = 10Y yield)
    # Price relative to par as yield shifts ±2%
    df["bond_duration"] = 10 / (1 + df["rate_10y"] / 100)  # approx Macaulay

    # ── Credit spread proxy (IG ~ spread-dependent)
    # When yield curve inverts, credit spreads tend to widen
    df["ig_spread"]  = np.clip(0.80 - df["spread"] * 0.3 + abs(df["spread"]) * 0.15, 0.5, 2.5).round(2)
    df["hy_spread"]  = np.clip(3.50 - df["spread"] * 0.8 + abs(df["spread"]) * 0.6,  2.0, 7.5).round(2)
    df["oas_spread"] = ((df["ig_spread"] + df["hy_spread"]) / 2).round(2)

    return df


df_full = load_data("T10Y2Y.csv")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:12px; padding:12px 0 24px;'>
      <div style='width:32px; height:32px; border:1px solid {BORDER}; border-radius:6px; display:flex; align-items:center; justify-content:center; font-family:Space Grotesk; font-weight:600; font-size:14px; color:{TEXT_PRIMARY};'>
        YC
      </div>
      <div>
        <div style='font-family:Space Grotesk;font-size:16px;font-weight:600;color:{TEXT_PRIMARY}'>
          Yield Curve<br>Intelligence
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    min_date = df_full["date"].min().date()
    max_date = df_full["date"].max().date()

    st.markdown("**Date Range**")
    date_preset = st.selectbox("Quick select", ["All Data", "Last 1Y", "Last 2Y", "Last 3Y", "Custom"])

    if date_preset == "Last 1Y":
        start_d = max_date - timedelta(days=365)
        end_d   = max_date
    elif date_preset == "Last 2Y":
        start_d = max_date - timedelta(days=730)
        end_d   = max_date
    elif date_preset == "Last 3Y":
        start_d = max_date - timedelta(days=1095)
        end_d   = max_date
    elif date_preset == "Custom":
        start_d = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
        end_d   = st.date_input("To",   max_date, min_value=min_date, max_value=max_date)
    else:
        start_d, end_d = min_date, max_date

    st.divider()
    st.markdown("**Chart Options**")
    show_ma      = st.checkbox("Show Moving Averages", value=True)
    show_regimes = st.checkbox("Shade Regime Zones",   value=True)
    bond_coupon  = st.slider("Bond Coupon Rate (%)", 1.0, 8.0, 4.5, 0.25)
    bond_par     = st.number_input("Par Value ($)", value=1000, step=100)

    st.divider()
    st.markdown(f"<div style='font-size:11px;color:{MUTED}'>Data: FRED T10Y2Y<br>Updated through {max_date}</div>",
                unsafe_allow_html=True)


# ── Filter data ───────────────────────────────────────────────────────────────
df = df_full[(df_full["date"].dt.date >= start_d) & (df_full["date"].dt.date <= end_d)].copy()

# ── KPI helpers ───────────────────────────────────────────────────────────────
latest  = df.iloc[-1]
prev_1w = df.iloc[max(-8, -len(df))]
prev_1m = df.iloc[max(-22, -len(df))]

def kpi(label, value, delta=None, color=None, sparkline_data=None):
    color = color or (ACCENT_GREEN if (delta or 0) >= 0 else ACCENT_RED)
    delta_html = f"<div class='kpi-delta' style='color:{color}'>{delta:+.2f} bps (1M)</div>" if delta is not None else ""

    spark_html = ""
    if sparkline_data is not None and not sparkline_data.empty:
        y = sparkline_data.values
        y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else np.zeros_like(y)
        x = np.linspace(0, 70, len(y_norm))
        points = " ".join([f"{x_val:.1f},{20 - y_val*18:.1f}" for x_val, y_val in zip(x, y_norm)])
        spark_html = f'<svg width="70" height="20" viewbox="0 0 70 20"><polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5"></polyline></svg>'

    return f"""
    <div class='kpi-card' style='--accent-color:{color}'>
      <div>
        <div class='kpi-label'>{label}</div>
        <div class='kpi-value' style='color:{color}'>{value}</div>
        {delta_html}
      </div>
      <div style='align-self:flex-end;'>{spark_html}</div>
    </div>
    """


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='display:flex;align-items:baseline;gap:14px;padding:8px 0 4px'>
  <span style='font-size:26px;font-weight:600;font-family:IBM Plex Mono'>
    US Treasury Yield Curve Dashboard
  </span>
  <span style='font-size:13px;color:{MUTED};font-family:JetBrains Mono'>
    10Y − 2Y Spread · {start_d} → {end_d}
  </span>
</div>
""", unsafe_allow_html=True)

# Current regime badge
regime_colors = {
    "Deeply Inverted": ACCENT_RED,
    "Inverted": ACCENT_AMBER,
    "Flat": MUTED,
    "Normal": ACCENT_GREEN,
}
cur_regime = latest["regime"]
rc = regime_colors[cur_regime]
st.markdown(f"""
<div style='margin-bottom:24px'>
  <span style='background:{rc}22;border:1px solid {rc};color:{rc};
               font-family:IBM Plex Mono;font-size:11px;letter-spacing:1.5px;
               padding:4px 12px;border-radius:20px;text-transform:uppercase'>
    ● Current Regime: {cur_regime}
  </span>
</div>
""", unsafe_allow_html=True)


# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

spread_now   = latest["spread"]
spread_1m    = prev_1m["spread"]
delta_spread = (spread_now - spread_1m) * 100   # in bps

with k1:
    color = ACCENT_RED if spread_now < 0 else ACCENT_GREEN
    spark_data = df['spread'].tail(30)
    st.markdown(kpi("10Y−2Y Spread", f"{spread_now:+.2f}%", delta_spread, color, spark_data), unsafe_allow_html=True)

with k2:
    r10 = latest["rate_10y"]
    d10 = (r10 - prev_1m["rate_10y"]) * 100
    spark_data = df['rate_10y'].tail(30)
    st.markdown(kpi("10Y Treasury Yield", f"{r10:.2f}%", d10, ACCENT_BLUE, spark_data), unsafe_allow_html=True)

with k3:
    r2 = latest["rate_2y"]
    d2 = (r2 - prev_1m["rate_2y"]) * 100
    spark_data = df['rate_2y'].tail(30)
    st.markdown(kpi("2Y Treasury Yield", f"{r2:.2f}%", d2, ACCENT_AMBER, spark_data), unsafe_allow_html=True)

with k4:
    ig = latest["ig_spread"]
    di = (ig - prev_1m["ig_spread"]) * 100
    color = ACCENT_RED if di > 0 else ACCENT_GREEN
    spark_data = df['ig_spread'].tail(30)
    st.markdown(kpi("IG Credit Spread", f"{ig:.2f}%", di, color, spark_data), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Yield Curve Spread", "💹 Bond Price vs Yield", "📊 Rate Trends", "🔬 Credit Spreads"
])


# ── TAB 1 · Yield Curve Spread ────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-header'>Spread Dynamics</div>", unsafe_allow_html=True)

    inversion_days = (df["spread"] < 0).sum()
    total_days     = len(df)
    pct_inverted   = inversion_days / total_days * 100

    if pct_inverted > 10:
        st.markdown(f"""<div class='warning-box'>
        ⚠️ <b>Inversion Alert:</b> The yield curve was inverted for <b>{inversion_days} trading days
        ({pct_inverted:.0f}% of selected period)</b>. Sustained inversions have historically
        preceded recessions by 12–18 months.
        </div>""", unsafe_allow_html=True)

    # Main spread chart
    fig1 = go.Figure()

    if show_regimes:
        # Shade inversion zones
        in_inv, inv_start = False, None
        for i, row in df.iterrows():
            if row["spread"] < 0 and not in_inv:
                in_inv, inv_start = True, row["date"]
            elif row["spread"] >= 0 and in_inv:
                fig1.add_vrect(x0=inv_start, x1=row["date"],
                               fillcolor=ACCENT_RED, opacity=0.08,
                               layer="below", line_width=0)
                in_inv = False
        if in_inv:
            fig1.add_vrect(x0=inv_start, x1=df["date"].iloc[-1],
                           fillcolor=ACCENT_RED, opacity=0.08,
                           layer="below", line_width=0)

    # Zero line
    fig1.add_hline(y=0, line_dash="dot", line_color=ACCENT_RED, line_width=1,
                   annotation_text="Inversion Threshold", annotation_position="top right",
                   annotation_font_color=ACCENT_RED, annotation_font_size=10)

    # Fill
    fig1.add_trace(go.Scatter(
        x=df["date"], y=df["spread"],
        fill="none",
        line=dict(color=ACCENT_BLUE, width=1.5),
        name="Daily Spread",
        hovertemplate="%{x|%Y-%m-%d}<br>Spread: <b>%{y:.3f}%</b><extra></extra>",
    ))

    if show_ma:
        fig1.add_trace(go.Scatter(x=df["date"], y=df["spread_30d_ma"], line=dict(color=ACCENT_AMBER, width=1.5, dash="dot"), name="30D MA", hovertemplate="%{x|%b %Y}<br>30D MA: %{y:.3f}%<extra></extra>"))
        fig1.add_trace(go.Scatter(x=df["date"], y=df["spread_90d_ma"], line=dict(color=ACCENT_GREEN, width=1.5, dash="dash"), name="90D MA", hovertemplate="%{x|%b %Y}<br>90D MA: %{y:.3f}%<extra></extra>"))

    fig1.update_layout(**PLOTLY_LAYOUT, height=400, yaxis_title="Spread (%)",
                       title="<span style='font-size:10px;color:#64748B'>TIME SERIES</span><br>10Y−2Y Treasury Spread")
    fig1.update_xaxes(rangeslider=dict(visible=True, bordercolor=BORDER, bgcolor=DARK_BG))
    fig1.update_yaxes(zeroline=True, zerolinecolor=ACCENT_RED)
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("<div class='section-header'>Regime Analysis</div>", unsafe_allow_html=True)
        regime_counts = df["regime"].value_counts()
        regime_labels = [f"{r} ({regime_counts.get(r, 0)}d)" for r in regime_counts.index]
        pal = [regime_colors.get(r, MUTED) for r in regime_counts.index]

        fig_pie = go.Figure(go.Pie(
            labels=regime_labels,
            values=regime_counts.values,
            marker_colors=pal, hole=0.55,
            textinfo="percent",
            hovertemplate="%{label}<br><b>%{percent}</b><extra></extra>",
        ))
        # Create a specific layout for this chart to avoid the TypeError
        pie_layout = PLOTLY_LAYOUT.copy()
        pie_layout['legend'] = pie_layout.get('legend', {}).copy() # Start with base legend settings
        pie_layout['legend'].update(dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))

        fig_pie.update_layout(**pie_layout, height=340,
                              title="<span style='font-size:10px;color:#64748B'>DISTRIBUTION</span><br>Regime Distribution",
                              showlegend=True,
                              annotations=[dict(text=f"{total_days}d", x=0.5, y=0.5, font=dict(size=20, color=TEXT_PRIMARY, family="JetBrains Mono"), showarrow=False)])
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Volatility Analysis</div>", unsafe_allow_html=True)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=df["date"], y=df["volatility"],
            fill="none",
            line=dict(color=ACCENT_AMBER, width=1.5),
            name="Volatility",
            hovertemplate="%{x|%Y-%m-%d}<br>Vol: <b>%{y:.4f}%</b><extra></extra>",
        ))
        fig_vol.update_layout(**PLOTLY_LAYOUT, height=220,
                              title="<span style='font-size:10px;color:#64748B'>VOLATILITY</span><br>30-Day Rolling Spread Volatility (σ)",
                              yaxis_title="Std Dev (%)")
        st.plotly_chart(fig_vol, use_container_width=True)

        st.markdown(f"""<div class='insight-box' style='margin-top:18px'>
        <b>Reading the spread:</b> A <span style='color:{ACCENT_GREEN}'>positive spread</span>
        reflects normal growth expectations. A <span style='color:{ACCENT_RED}'>negative spread (inversion)</span>
        signals that short-term rates are pricing Fed tightening faster than long-term rates price future growth —
        historically the single most reliable recession leading indicator.
        </div>""", unsafe_allow_html=True)


# ── TAB 2 · Bond Price vs Yield ───────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'>Bond Pricing & Duration</div>", unsafe_allow_html=True)

    st.markdown(f"""<div class='insight-box'>
    Adjust the <b>coupon rate</b> and <b>par value</b> in the sidebar. The price–yield curve shows the
    inverse relationship: as yields rise, existing bond prices fall. The slope steepens at low yields
    (high duration / convexity), meaning long-duration bonds are far more rate-sensitive.
    </div>""", unsafe_allow_html=True)

    # Price-yield curve
    yields_range = np.linspace(0.5, 10.0, 200)
    n_periods    = 10  # 10-year bond
    coupon       = bond_coupon / 100 * bond_par

    def bond_price(y_pct, n=10, c=coupon, par=bond_par):
        y = y_pct / 100
        t = np.arange(1, n + 1)
        return c * (1 - (1 + y) ** -n) / y + par * (1 + y) ** -n

    prices = [bond_price(y) for y in yields_range]
    duration_approx = [-(bond_price(y + 0.01) - bond_price(y - 0.01)) / (0.02 * bond_price(y))
                       for y in yields_range]

    # Current 10Y yield marker
    cur_yield = float(latest["rate_10y"])
    cur_price = bond_price(cur_yield)

    fig_py = go.Figure()
    fig_py.add_trace(go.Scatter(
        x=yields_range, y=prices,
        line=dict(color=ACCENT_BLUE, width=1.5),
        fill="none",
        name="Bond Price",
        hovertemplate="Yield: <b>%{x:.2f}%</b><br>Price: <b>$%{y:,.2f}</b><extra></extra>",
    ))
    fig_py.add_trace(go.Scatter(
        x=[cur_yield], y=[cur_price],
        mode="markers+text",
        marker=dict(size=12, color=ACCENT_AMBER, symbol="diamond",
                    line=dict(width=1.5, color=TEXT_PRIMARY)),
        text=[f"  Current 10Y<br>  ${cur_price:,.0f}"],
        textfont=dict(color=ACCENT_AMBER, size=11),
        textposition="top right",
        name="Current",
        hovertemplate=f"Current 10Y Yield: {cur_yield:.2f}%<br>Price: ${cur_price:,.2f}<extra></extra>",
    ))
    fig_py.add_vline(x=bond_coupon, line_dash="dot", line_color=ACCENT_GREEN,
                     annotation_text=f"Coupon: {bond_coupon}%",
                     annotation_font_color=ACCENT_GREEN, annotation_font_size=10)
    fig_py.update_layout(**PLOTLY_LAYOUT, height=400,
                          title=f"<span style='font-size:10px;color:#64748B'>MODEL</span><br>{n_periods}Y Bond  |  Coupon {bond_coupon}%  |  Par ${bond_par:,}",
                          xaxis_title="Yield to Maturity (%)", yaxis_title="Price ($)")
    st.plotly_chart(fig_py, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='section-header'>Duration Analysis</div>", unsafe_allow_html=True)
        fig_dur = go.Figure()
        fig_dur.add_trace(go.Scatter(
            x=yields_range, y=duration_approx,
            line=dict(color=ACCENT_GREEN, width=1.5),
            fill="none",
            name="Modified Duration",
            hovertemplate="Yield: %{x:.2f}%<br>Duration: <b>%{y:.2f} yrs</b><extra></extra>",
        ))
        fig_dur.add_vline(x=cur_yield, line_dash="dot", line_color=ACCENT_AMBER)
        fig_dur.update_layout(**PLOTLY_LAYOUT, height=280,
                               title="<span style='font-size:10px;color:#64748B'>SENSITIVITY</span><br>Modified Duration vs Yield",
                               xaxis_title="Yield (%)", yaxis_title="Duration (years)")
        st.plotly_chart(fig_dur, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Scenario Analysis</div>", unsafe_allow_html=True)
        scenarios = [-2, -1, -0.5, 0, 0.5, 1, 2]
        s_data = []
        for ds in scenarios:
            ny   = max(0.01, cur_yield + ds)
            np_  = bond_price(ny)
            chg  = np_ - cur_price
            pct  = chg / cur_price * 100
            s_data.append({"Yield Shift": f"{ds:+.1f}%", "New Yield": f"{ny:.2f}%",
                            "Price ($)": f"{np_:,.2f}", "P&L ($)": f"{chg:+.2f}", "Return (%)": f"{pct:+.2f}%"})

        s_df = pd.DataFrame(s_data)

        fig_tbl = go.Figure(go.Table(
            header=dict(values=list(s_df.columns),
                        fill_color=BORDER, align="center",
                        font=dict(color=TEXT_PRIMARY, family="JetBrains Mono", size=11),
                        line_color=DARK_BG),
            cells=dict(values=[s_df[c] for c in s_df.columns],
                       fill_color=[[CARD_BG if i != 3 else "#1a2535" for i in range(len(s_df))]],
                       align="center",
                       font=dict(color=[TEXT_SEC if r["Yield Shift"] != "+0.0%"
                                        else ACCENT_AMBER for _, r in s_df.iterrows()],
                                 family="JetBrains Mono", size=11),
                       line_color=DARK_BG, height=28),
        ))
        fig_tbl.update_layout(**PLOTLY_LAYOUT, height=280,
                              title="<span style='font-size:10px;color:#64748B'>SCENARIOS</span><br>Price Sensitivity to Yield Shifts")
        st.plotly_chart(fig_tbl, use_container_width=True)


# ── TAB 3 · Rate Trends ───────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'>Rate & Spread Trends</div>", unsafe_allow_html=True)

    # Dual-axis rate chart
    fig_rates = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rates.add_trace(go.Scatter(
        x=df["date"], y=df["rate_2y"],
        name="2Y Yield",
        line=dict(color=ACCENT_AMBER, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>2Y: <b>%{y:.2f}%</b><extra></extra>",
    ), secondary_y=False)
    fig_rates.add_trace(go.Scatter(
        x=df["date"], y=df["rate_10y"],
        name="10Y Yield",
        line=dict(color=ACCENT_BLUE, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>10Y: <b>%{y:.2f}%</b><extra></extra>",
    ), secondary_y=False)
    fig_rates.add_trace(go.Scatter(
        x=df["date"], y=df["spread"],
        name="Spread (10Y−2Y)",
        line=dict(color=ACCENT_GREEN, width=1.2, dash="dot"),
        opacity=0.7,
        hovertemplate="%{x|%Y-%m-%d}<br>Spread: <b>%{y:.3f}%</b><extra></extra>",
    ), secondary_y=True)

    fig_rates.update_layout(**PLOTLY_LAYOUT, height=400,
                            title="<span style='font-size:10px;color:#64748B'>RATES</span><br>2Y & 10Y Treasury Yields + Spread Overlay")
    fig_rates.update_yaxes(title_text="Rate (%)", secondary_y=False, gridcolor=BORDER)
    fig_rates.update_yaxes(title_text="Spread (%)", secondary_y=True, showgrid=False)
    st.plotly_chart(fig_rates, use_container_width=True)

    st.divider()
    st.markdown("<div class='section-header'>Historical Performance</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        # YoY change heatmap
        df_yoy = df.dropna(subset=["yoy_change"]).copy()
        df_yoy["year"]  = df_yoy["date"].dt.year
        df_yoy["month"] = df_yoy["date"].dt.month

        pivot = df_yoy.groupby(["year", "month"])["yoy_change"].mean().unstack(fill_value=np.nan)
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        available_months = [month_labels[m-1] for m in pivot.columns]

        fig_heat = go.Figure(go.Heatmap(
            z=pivot.values,
            x=available_months,
            y=pivot.index.astype(str),
            colorscale=[[0, ACCENT_RED], [0.5, DARK_BG], [1, ACCENT_GREEN]],
            zmid=0,
            colorbar=dict(title="YoY Δ (%)", tickfont=dict(color=TEXT_SEC)),
            hovertemplate="%{y} %{x}<br>YoY Change: <b>%{z:.3f}%</b><extra></extra>",
            text=pivot.values.round(2),
            texttemplate="%{text}",
            textfont=dict(size=9, color=TEXT_SEC, family="JetBrains Mono"),
        ))
        fig_heat.update_layout(**PLOTLY_LAYOUT, height=320,
                                title="<span style='font-size:10px;color:#64748B'>HEATMAP</span><br>Monthly Avg YoY Spread Change")
        st.plotly_chart(fig_heat, use_container_width=True)

    with c2:
        # Rolling 90D stats table
        stats = {
            "Current Spread":  f"{latest['spread']:+.3f}%",
            "90D Average":     f"{df['spread'].tail(90).mean():+.3f}%",
            "90D High":        f"{df['spread'].tail(90).max():+.3f}%",
            "90D Low":         f"{df['spread'].tail(90).min():+.3f}%",
            "All-time High":   f"{df['spread'].max():+.3f}%",
            "All-time Low":    f"{df['spread'].min():+.3f}%",
            "% Days Inverted": f"{(df['spread'] < 0).mean()*100:.1f}%",
            "Current Regime":  latest["regime"],
        }
        st.markdown("<div style='font-family:Space Grotesk;font-weight:500;font-size:14px;color:#94A3B8;margin-bottom:16px;'>Key Statistics</div>", unsafe_allow_html=True)
        for k, v in stats.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;padding:7px 0;"
                f"border-bottom:1px solid {BORDER};font-size:13px'>"
                f"<span style='color:{MUTED};font-family:Inter'>{k}</span>"
                f"<span style='color:{TEXT_PRIMARY};font-family:JetBrains Mono;font-weight:600'>{v}</span></div>",
                unsafe_allow_html=True
            )


# ── TAB 4 · Credit Spreads ────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-header'>Credit Market Analysis</div>", unsafe_allow_html=True)

    st.markdown(f"""<div class='insight-box'>
    <b>How credit spreads relate to the yield curve:</b> Yield curve inversions tighten lending
    margins and signal economic stress, which typically causes credit spreads to <em>widen</em>
    — especially in high-yield (HY) paper. The IG/HY spread here is a model proxy derived from
    the T10Y2Y spread dynamics. Real-world data would use FRED BAMLC0A0CM (IG OAS) & BAMLH0A0HYM2 (HY OAS).
    </div>""", unsafe_allow_html=True)

    fig_cs = go.Figure()
    fig_cs.add_trace(go.Scatter(x=df["date"], y=df["hy_spread"],
                                name="HY Spread",
                                line=dict(color=ACCENT_RED, width=1.5),
                                fill="none",
                                hovertemplate="%{x|%Y-%m-%d}<br>HY: <b>%{y:.2f}%</b><extra></extra>"))
    fig_cs.add_trace(go.Scatter(x=df["date"], y=df["ig_spread"],
                                name="IG Spread",
                                line=dict(color=ACCENT_GREEN, width=1.5),
                                fill="none",
                                hovertemplate="%{x|%Y-%m-%d}<br>IG: <b>%{y:.2f}%</b><extra></extra>"))
    fig_cs.add_trace(go.Scatter(x=df["date"], y=df["oas_spread"],
                                name="Blended OAS",
                                line=dict(color=ACCENT_AMBER, width=1.2, dash="dot"),
                                hovertemplate="%{x|%Y-%m-%d}<br>OAS: <b>%{y:.2f}%</b><extra></extra>"))
    fig_cs.update_layout(**PLOTLY_LAYOUT, height=380,
                          title="<span style='font-size:10px;color:#64748B'>PROXY MODEL</span><br>IG / HY Credit Spread (Yield Curve Derived)",
                          yaxis_title="Spread (%)")
    st.plotly_chart(fig_cs, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Correlation Analysis</div>", unsafe_allow_html=True)
        fig_scat = go.Figure()

        # BUG FIX 1: Use categorical colors for years instead of a colorbar
        years = sorted(df["date"].dt.year.unique())
        year_colors = px.colors.qualitative.Vivid

        for i, year in enumerate(years):
            df_year = df[df["date"].dt.year == year]
            fig_scat.add_trace(go.Scatter(
                x=df_year["spread"], y=df_year["hy_spread"],
                mode="markers",
                marker=dict(
                    color=year_colors[i % len(year_colors)],
                    size=4,
                    opacity=0.7
                ),
                name=str(year),
                legendgroup=str(year),
                hovertemplate=f"<b>{year}</b><br>T10Y2Y: %{{x:.2f}}%<br>HY Spread: %{{y:.2f}}%<extra></extra>",
            ))

        z = np.polyfit(df["spread"].dropna(), df.loc[df["spread"].notna(), "hy_spread"], 1)
        xline = np.array([df["spread"].min(), df["spread"].max()])
        fig_scat.add_trace(go.Scatter(x=xline, y=np.polyval(z, xline),
                                       line=dict(color=ACCENT_AMBER, dash="dash", width=1.5),
                                       name="Trend"))

        scat_layout = PLOTLY_LAYOUT.copy()
        scat_layout['legend'] = scat_layout.get('legend', {}).copy()
        scat_layout['legend'].update(dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5))

        fig_scat.update_layout(**scat_layout, height=320, showlegend=True,
                                title="<span style='font-size:10px;color:#64748B'>RELATIONSHIP</span><br>HY Spread vs T10Y2Y",
                                xaxis_title="T10Y2Y Spread (%)", yaxis_title="HY Spread (%)")
        st.plotly_chart(fig_scat, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Quarterly Performance</div>", unsafe_allow_html=True)
        monthly_cs = df.copy()
        monthly_cs["ym"] = monthly_cs["date"].dt.to_period("Q").astype(str)
        qcs = monthly_cs.groupby("ym")[["ig_spread", "hy_spread"]].mean().tail(12).reset_index()

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=qcs["ym"], y=qcs["ig_spread"],
                                  name="IG Spread", marker_color=ACCENT_GREEN,
                                  hovertemplate="%{x}<br>IG: <b>%{y:.2f}%</b><extra></extra>"))
        fig_bar.add_trace(go.Bar(x=qcs["ym"], y=qcs["hy_spread"],
                                  name="HY Spread", marker_color=ACCENT_RED, opacity=0.75,
                                  hovertemplate="%{x}<br>HY: <b>%{y:.2f}%</b><extra></extra>"))
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=320, barmode="group",
                               title="<span style='font-size:10px;color:#64748B'>QUARTERLY</span><br>Avg Credit Spreads (Last 12Q)",
                               xaxis_title="Quarter", yaxis_title="Spread (%)")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.markdown("<div class='section-header'>Composite Risk Indicator</div>", unsafe_allow_html=True)
    risk_score = np.clip(
        (df["hy_spread"].iloc[-1] - 2.0) / 5.5 * 100 +
        (abs(df["spread"].iloc[-1]) if df["spread"].iloc[-1] < 0 else 0) * 20, 0, 100
    )
    gauge_color = ACCENT_GREEN if risk_score < 30 else ACCENT_AMBER if risk_score < 60 else ACCENT_RED

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_score, 1),
        number=dict(suffix=" / 100", font=dict(color=TEXT_PRIMARY, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED)),
            bar=dict(color=gauge_color),
            bgcolor=CARD_BG,
            bordercolor=BORDER,
            steps=[
                dict(range=[0, 30],  color="rgba(5,150,105,0.15)"),
                dict(range=[30, 60], color="rgba(217,119,6,0.15)"),
                dict(range=[60, 100],color="rgba(220,38,38,0.15)"),
            ],
            threshold=dict(line=dict(color=TEXT_PRIMARY, width=2), thickness=0.75, value=risk_score),
        ),
        title=dict(text="Composite Credit Risk Score", font=dict(color=TEXT_SEC, size=14, family="Space Grotesk"))
    ))
    # Create a specific layout for the gauge to avoid the TypeError on 'margin'
    gauge_layout = PLOTLY_LAYOUT.copy()
    gauge_layout['margin'] = dict(l=30, r=30, t=60, b=30)

    fig_gauge.update_layout(**gauge_layout, height=250)
    st.plotly_chart(fig_gauge, use_container_width=True)
