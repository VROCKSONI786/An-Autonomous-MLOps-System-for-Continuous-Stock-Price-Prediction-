import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_ROOT, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from monitoring.performance_monitor import PerformanceMonitor
from monitoring.drift_detector import DriftDetector


def _load_config() -> dict:
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_history(logs_dir: str) -> pd.DataFrame:
    path = os.path.join(logs_dir, "performance_history.jsonl")
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _load_latest_drift_summary(reports_dir: str) -> list[dict]:
    if not os.path.exists(reports_dir):
        return []
    files = sorted(
        [f for f in os.listdir(reports_dir) if f.startswith("drift_summary_")],
        reverse=True,
    )
    if not files:
        return []
    with open(os.path.join(reports_dir, files[0])) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────

def show():
    st.header("📊 Performance & Drift Monitoring")

    config = _load_config()
    symbols = config["data_collection"]["stock_symbols"]
    logs_dir = os.path.join(PROJECT_ROOT, "logs")
    reports_dir = os.path.join(PROJECT_ROOT, "data", "reports")

    # ── Top action bar ────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)

    run_perf = col_a.button("🔍 Check Performance", width='stretch', type="primary")
    run_drift = col_b.button("🌊 Check Data Drift", width='stretch')
    run_both = col_c.button("🚀 Run Full Check", width='stretch')

    # ── Performance check ─────────────────────────────────────────────────────
    if run_perf or run_both:
        with st.spinner("Evaluating models on real INR prices…"):
            try:
                monitor = PerformanceMonitor()
                metrics, degraded = monitor.monitor_all_models(symbols)
                st.session_state["perf_metrics"] = metrics
                st.session_state["perf_degraded"] = degraded
                if metrics:
                    st.success(f"✅ Evaluated {len(metrics)} models.")
                else:
                    st.warning("⚠ No models found. Train models first.")
            except Exception as e:
                st.error(f"Performance check error: {e}")

    # ── Drift check ───────────────────────────────────────────────────────────
    if run_drift or run_both:
        with st.spinner("Running drift detection…"):
            try:
                ref_days = config.get("drift", {}).get("reference_period_days", 90)
                detector = DriftDetector(reference_period_days=ref_days)
                drift_results = detector.check_all_symbols(symbols)
                st.session_state["drift_results"] = drift_results
                if drift_results:
                    n_drifted = sum(1 for r in drift_results if r["drift_detected"])
                    st.success(
                        f"✅ Drift check complete. {n_drifted}/{len(drift_results)} symbols drifted."
                    )
                else:
                    st.warning("⚠ No drift data — preprocessed data may be missing.")
            except Exception as e:
                st.error(f"Drift detection error: {e}")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # PERFORMANCE SECTION
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📈 Model Performance (INR Scale)")

    # Add explanation
    with st.expander("ℹ️ Understanding Performance Metrics"):
        st.markdown("""
        ### Key Metrics Explained

        **MAE (Mean Absolute Error)** - ₹X
        - Average amount (in rupees) predictions are off by
        - Example: MAE = ₹45 means on average prediction is ±₹45 from actual
        - **When to act**: If MAE increases by >20%, retrain model
        
        **MAPE (Mean Absolute Percentage Error)** - X%
        - Percentage error across all predictions
        - More useful than MAE because it scales with stock price
        - Stock at ₹500: 2% error = ±₹10 | Stock at ₹2000: 2% error = ±₹40
        - **Threshold**: < 3% is GOOD | 3-5% is ACCEPTABLE | > 5% needs attention
        
        **Direction Accuracy** - X%
        - "Did we correctly predict UP or DOWN?" (Most important for traders)
        - Baseline: 50% (random coin flip)
        - Your model: 72% = 44% BETTER than random
        - **Threshold**: > 60% is GOOD | > 70% is EXCELLENT
        
        **R² Score** - 0.0 to 1.0
        - How well model explains price variance (0 = useless, 1 = perfect)
        - 0.8+ = Strong | 0.5-0.8 = Moderate | <0.5 = Weak signal
        
        **Hit Rate** - (Direction_Acc + Within_2%) / 2
        - Combined metric for trading reliability
        - **Production threshold**: > 65%
        """)

    metrics = st.session_state.get("perf_metrics", None)
    degraded = st.session_state.get("perf_degraded", [])

    if not metrics:
        # Try loading from latest snapshot
        if os.path.exists(logs_dir):
            snapshots = sorted(
                [f for f in os.listdir(logs_dir) if f.startswith("performance_metrics_")],
                reverse=True,
            )
            if snapshots:
                with open(os.path.join(logs_dir, snapshots[0])) as f:
                    metrics = json.load(f)
                if metrics:
                    st.info(f"Showing cached results from {snapshots[0]}")

    if metrics:
        # Degradation banner
        if degraded:
            st.error(f"⚠ Models needing retraining: **{', '.join(degraded)}**")
        else:
            st.success("✅ All models within acceptable performance thresholds.")

        # KPI cards
        cols = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            with cols[i]:
                sym = m["symbol"].replace(".NS", "")
                color = "🔴" if m["symbol"] in degraded else "🟢"
                st.metric(f"{color} {sym}", f"₹{m.get('mae_inr', 0):.2f}", "MAE")
                st.metric("MAPE", f"{m.get('mape_pct', 0):.2f}%")
                st.metric("R²", f"{m.get('r2_score', 0):.4f}")
                st.metric("Dir Acc", f"{m.get('direction_accuracy_pct', 0):.1f}%")

        st.divider()

        # Bar chart comparison
        df_metrics = pd.DataFrame(metrics)
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                df_metrics,
                x="symbol",
                y="mae_inr",
                title="MAE per Stock (₹)",
                color="symbol",
                labels={"mae_inr": "MAE (₹)", "symbol": "Stock"},
                text_auto=".2f",
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, width='stretch')

        with col2:
            fig = px.bar(
                df_metrics,
                x="symbol",
                y="mape_pct",
                title="MAPE per Stock (%)",
                color="symbol",
                labels={"mape_pct": "MAPE (%)", "symbol": "Stock"},
                text_auto=".2f",
            )
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, width='stretch')

        # Full metrics table
        with st.expander("📋 Full Metrics Table"):
            display_cols = [
                "symbol", "mae_inr", "rmse_inr", "mape_pct",
                "r2_score", "direction_accuracy_pct",
                "last_actual_price_inr", "last_predicted_price_inr",
                "test_samples", "timestamp",
            ]
            display_df = df_metrics[[c for c in display_cols if c in df_metrics.columns]]
            st.dataframe(display_df, width='stretch', hide_index=True)
    else:
        st.info("No performance data available. Click 'Check Performance' to evaluate models.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # DRIFT SECTION
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("🌊 Data Drift Detection")

    # Add drift explanation
    with st.expander("ℹ️ What is Data Drift? Why Does It Matter?"):
        st.markdown("""
        ### Understanding Data Drift

        **What is it?**
        - Changes in real-world data patterns compared to training data
        - "Your model learned from 2023 data, but now it's 2026 — market has changed"
        
        **Why it matters?**
        - LSTM models assume: "Future patterns ≈ Past patterns"
        - When drift occurs, this assumption breaks down
        - Predictions become unreliable → Need to retrain
        
        ### Common Drift Causes
        1. **Market Volatility Change** — Volume_MA, Volatility_ATR increase
        2. **Sector Rotation** — Beta changes (stock moves differently vs benchmark)
        3. **Company Events** — Earnings, splits, dividend changes
        4. **Macro Shifts** — Interest rate changes, inflation, policy
        5. **Seasonal Patterns** — New trading patterns emerge
        
        ### What To Do?
        - 🟢 **Green (No Drift)**: Continue using model as is
        - 🟡 **Yellow (Low-Medium Drift)**: Monitor closely, plan retraining
        - 🔴 **Red (High Drift)**: RETRAIN IMMEDIATELY
        
        ### Retraining Process
        Typically takes 5-10 minutes per stock (new data + training)
        """)

    drift_results = st.session_state.get("drift_results", None)

    if not drift_results:
        drift_results = _load_latest_drift_summary(reports_dir)
        if drift_results:
            st.info("Showing cached drift results from last run.")

    if drift_results:
        # Summary row
        n_drifted = sum(1 for r in drift_results if r.get("drift_detected", False))
        d_cols = st.columns(4)
        d_cols[0].metric("Symbols Checked", len(drift_results))
        d_cols[1].metric("Drifted", n_drifted, delta=None if n_drifted == 0 else f"{n_drifted} ⚠")
        d_cols[2].metric("Clean", len(drift_results) - n_drifted)
        avg_score = round(
            sum(r.get("drift_score", 0) for r in drift_results) / len(drift_results), 4
        )
        d_cols[3].metric("Avg Drift Score", avg_score)

        # Per-symbol cards
        for dr in drift_results:
            sym = dr["symbol"]
            detected = dr.get("drift_detected", False)
            score = dr.get("drift_score", 0.0)
            n_cols_drifted = dr.get("n_drifted_columns", 0)
            method = dr.get("method", "N/A")
            icon = "🔴" if detected else "🟢"

            with st.expander(f"{icon} {sym} — Drift score: {score:.4f} | Method: {method}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Drift Detected", "Yes" if detected else "No")
                c2.metric("Drift Score", f"{score:.4f}")
                c3.metric("Columns Drifted", n_cols_drifted)
                c4.metric("Reference Rows", dr.get("reference_size", "N/A"))

                col_drifts = dr.get("column_drifts", [])
                if col_drifts:
                    cdf = pd.DataFrame(col_drifts[:15])  # top 15
                    fig = px.bar(
                        cdf,
                        x="column",
                        y="drift_score",
                        color="drift_detected",
                        title=f"{sym} — Feature Drift Scores",
                        labels={"drift_score": "Drift Score", "column": "Feature"},
                        color_discrete_map={True: "#EF4444", False: "#22C55E"},
                    )
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, width='stretch')

                report_path = dr.get("report_path", "")
                if report_path and os.path.exists(report_path):
                    st.caption(f"Full report: `{report_path}`")
    else:
        st.info("No drift data available. Click 'Check Data Drift' to run analysis.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # PERFORMANCE HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("📜 Performance History")

    history_df = _load_history(logs_dir)

    if not history_df.empty:
        # Symbol selector
        available_syms = sorted(history_df["symbol"].unique().tolist())
        selected = st.multiselect("Filter by symbol", available_syms, default=available_syms)
        fdf = history_df[history_df["symbol"].isin(selected)].copy()
        fdf["timestamp"] = pd.to_datetime(fdf["timestamp"])
        fdf = fdf.sort_values("timestamp")

        if "mae_inr" in fdf.columns:
            fig = px.line(
                fdf,
                x="timestamp",
                y="mae_inr",
                color="symbol",
                title="MAE Over Time (₹)",
                markers=True,
                labels={"mae_inr": "MAE (₹)", "timestamp": "Date"},
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, width='stretch')

        with st.expander("📋 Raw History"):
            st.dataframe(
                fdf.tail(50).sort_values("timestamp", ascending=False),
                width='stretch',
                hide_index=True,
            )
    else:
        st.info("No performance history yet. History is built up over repeated monitoring runs.")