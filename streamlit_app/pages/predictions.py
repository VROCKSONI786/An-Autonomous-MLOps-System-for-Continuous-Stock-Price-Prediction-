import os
import sys
import yaml
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_ROOT, PROJECT_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model.lstm_model import LSTMStockModel
from preprocessing.data_preprocessor import DataPreprocessor

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "saved_models")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "scaler.pkl")
CLOSE_IDX = 3  # Close is the 4th feature


def _load_config() -> dict:
    with open(os.path.join(PROJECT_ROOT, "config", "config.yaml")) as f:
        return yaml.safe_load(f)


def _available_models() -> list[str]:
    if not os.path.exists(MODELS_DIR):
        return []
    return [
        f.replace("_lstm_model.keras", "")
        for f in os.listdir(MODELS_DIR)
        if f.endswith("_lstm_model.keras")
    ]


def _inverse_close(scaler, scaled: np.ndarray) -> np.ndarray:
    """Inverse-transform scaled Close values to INR."""
    n = len(scaled)
    n_feat = scaler.n_features_in_
    dummy = np.zeros((n, n_feat))
    dummy[:, CLOSE_IDX] = scaled.flatten()
    return scaler.inverse_transform(dummy)[:, CLOSE_IDX]


def _forecast_future(
    lstm_model: LSTMStockModel,
    last_sequence: np.ndarray,
    n_days: int,
) -> np.ndarray:
    """
    Iteratively predict n_days into the future.
    last_sequence shape: (seq_len, n_features)
    Returns array of n_days scaled Close predictions.
    """
    seq = last_sequence.copy()  # (60, 21)
    predictions = []

    for _ in range(n_days):
        x = seq[np.newaxis, :, :]          # (1, 60, 21)
        pred_scaled = lstm_model.predict(x).flatten()[0]  # scalar
        predictions.append(pred_scaled)

        # Slide window: drop oldest, append new step
        new_step = seq[-1].copy()
        new_step[CLOSE_IDX] = pred_scaled  # update Close feature
        seq = np.vstack([seq[1:], new_step[np.newaxis, :]])

    return np.array(predictions)


# ─────────────────────────────────────────────────────────────────────────────

def show():
    st.header("📈 Stock Price Predictions")

    available = _available_models()
    if not available:
        st.warning(
            f"⚠ No trained models found in `{MODELS_DIR}`. "
            "Please go to **Model Training** and train models first."
        )
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.selectbox("Select Stock", available)
    with col2:
        forecast_days = st.slider("Forecast Days", 1, 30, 7)
    with col3:
        show_ci = st.checkbox("Show ±5% band", value=True)

    if st.button("🔮 Generate Predictions", width='stretch', type="primary"):
        _run_predictions(symbol, forecast_days, show_ci)


def _run_predictions(symbol: str, forecast_days: int, show_ci: bool):
    # ── Load scaler ───────────────────────────────────────────────────────────
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler not found at `{SCALER_PATH}`. Run preprocessing first.")
        return
    scaler = joblib.load(SCALER_PATH)

    with st.spinner(f"Loading model for {symbol}…"):
        model_path = os.path.join(MODELS_DIR, f"{symbol}_lstm_model.keras")
        lstm_model = LSTMStockModel(sequence_length=60, n_features=21)
        try:
            lstm_model.load_model(model_path)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            return

    with st.spinner("Preparing data and generating predictions…"):
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = lstm_model.prepare_data(
                symbol=symbol, test_size=0.2, validation_split=0.1
            )
        except Exception as e:
            st.error(f"Data preparation failed: {e}")
            return

        if len(X_test) == 0:
            st.error("No test data available for this symbol.")
            return

        # ── Historical predictions ────────────────────────────────────────────
        y_pred_scaled = lstm_model.predict(X_test).flatten()
        y_actual_inr = _inverse_close(scaler, y_test)
        y_pred_inr = _inverse_close(scaler, y_pred_scaled)

        # ── Future forecast ───────────────────────────────────────────────────
        last_seq = X_test[-1]  # (60, 21)
        future_scaled = _forecast_future(lstm_model, last_seq, forecast_days)
        future_inr = _inverse_close(scaler, future_scaled)

        # ── Metrics ───────────────────────────────────────────────────────────
        mae = mean_absolute_error(y_actual_inr, y_pred_inr)
        rmse = np.sqrt(mean_squared_error(y_actual_inr, y_pred_inr))
        r2 = r2_score(y_actual_inr, y_pred_inr)
        mape = np.mean(
            np.abs((y_actual_inr - y_pred_inr) / np.where(y_actual_inr == 0, 1e-10, y_actual_inr))
        ) * 100
        if len(y_actual_inr) > 1:
            dir_acc = np.mean(
                np.sign(np.diff(y_actual_inr)) == np.sign(np.diff(y_pred_inr))
            ) * 100
        else:
            dir_acc = 0.0

    # ── KPI cards ──────────────────────────────────────────────────────────
    st.subheader("📊 Model Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MAE", f"₹{mae:.2f}")
    c2.metric("RMSE", f"₹{rmse:.2f}")
    c3.metric("R²", f"{r2:.4f}")
    c4.metric("MAPE", f"{mape:.2f}%")
    c5.metric("Dir Accuracy", f"{dir_acc:.1f}%")

    # ── Chart ─────────────────────────────────────────────────────────────
    st.subheader("📈 Historical Predictions vs Actual")

    # Show last 90 points to keep chart readable
    view_n = min(90, len(y_actual_inr))
    x_hist = list(range(view_n))
    act_view = y_actual_inr[-view_n:]
    pred_view = y_pred_inr[-view_n:]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x_hist, y=act_view, name="Actual", line=dict(color="#3B82F6", width=2))
    )
    fig.add_trace(
        go.Scatter(
            x=x_hist, y=pred_view, name="Predicted",
            line=dict(color="#F97316", width=2, dash="dot")
        )
    )
    if show_ci:
        upper = pred_view * 1.05
        lower = pred_view * 0.95
        fig.add_trace(
            go.Scatter(
                x=x_hist + x_hist[::-1],
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(249,115,22,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="±5% band",
            )
        )
    fig.update_layout(
        title=f"{symbol} — Last {view_n} Test Days",
        xaxis_title="Time Step",
        yaxis_title="Price (₹)",
        hovermode="x unified",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width='stretch')

    # ── Future forecast chart ──────────────────────────────────────────────
    st.subheader(f"🔮 {forecast_days}-Day Future Forecast")

    last_price = float(y_actual_inr[-1])
    x_fut = list(range(forecast_days))

    fig2 = go.Figure()
    fig2.add_hline(
        y=last_price, line_dash="dash",
        line_color="gray", annotation_text=f"Last Close ₹{last_price:.2f}"
    )
    fig2.add_trace(
        go.Scatter(
            x=x_fut, y=future_inr,
            name="Forecast",
            mode="lines+markers",
            line=dict(color="#10B981", width=2.5),
            marker=dict(size=6),
        )
    )
    if show_ci:
        upper_f = future_inr * 1.05
        lower_f = future_inr * 0.95
        fig2.add_trace(
            go.Scatter(
                x=x_fut + x_fut[::-1],
                y=np.concatenate([upper_f, lower_f[::-1]]),
                fill="toself",
                fillcolor="rgba(16,185,129,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="±5% band",
            )
        )
    fig2.update_layout(
        title=f"{symbol} — {forecast_days}-Day Forecast",
        xaxis_title="Days Ahead",
        yaxis_title="Forecasted Price (₹)",
        hovermode="x unified",
        height=380,
    )
    st.plotly_chart(fig2, width='stretch')

    # ── Forecast table ─────────────────────────────────────────────────────
    trend = "📈 Bullish" if future_inr[-1] > last_price else "📉 Bearish"
    pct_chg = (future_inr[-1] - last_price) / last_price * 100

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Last Close", f"₹{last_price:.2f}")
    col_b.metric(f"Day +{forecast_days} Forecast", f"₹{future_inr[-1]:.2f}", f"{pct_chg:+.2f}%")
    col_c.metric("Trend", trend)

    with st.expander("📋 Day-by-Day Forecast"):
        fdf = pd.DataFrame(
            {
                "Day": [f"+{i+1}" for i in range(forecast_days)],
                "Forecasted Price (₹)": [f"₹{p:.2f}" for p in future_inr],
                "Change from Last (₹)": [f"{p - last_price:+.2f}" for p in future_inr],
                "Change (%)": [f"{(p - last_price)/last_price*100:+.2f}%" for p in future_inr],
            }
        )
        st.dataframe(fdf, width='stretch', hide_index=True)

    # ── Detailed comparison table ───────────────────────────────────────────
    with st.expander("📋 Actual vs Predicted (last 20 test days)"):
        tail = min(20, len(y_actual_inr))
        cdf = pd.DataFrame(
            {
                "Actual (₹)": [f"₹{v:.2f}" for v in y_actual_inr[-tail:]],
                "Predicted (₹)": [f"₹{v:.2f}" for v in y_pred_inr[-tail:]],
                "Error (₹)": [f"{a - p:+.2f}" for a, p in zip(y_actual_inr[-tail:], y_pred_inr[-tail:])],
                "Error (%)": [
                    f"{(a-p)/a*100:+.2f}%" for a, p in zip(y_actual_inr[-tail:], y_pred_inr[-tail:])
                ],
            }
        )
        st.dataframe(cdf, width='stretch', hide_index=True)