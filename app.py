from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from frp_predictor import FRPPredictor, PredictorLoadError


APP_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="FRP Strength Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_predictor() -> FRPPredictor:
    return FRPPredictor(APP_DIR)


@st.cache_data
def load_base_dataset() -> pd.DataFrame:
    return get_predictor().dataset.copy()


@st.cache_data
def load_sheet(sheet_name: str) -> pd.DataFrame:
    return get_predictor().get_results_sheet(sheet_name)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Instrument+Serif:ital@0;1&display=swap');

        :root {
            --bg: #f4efe7;
            --paper: rgba(255, 252, 247, 0.72);
            --ink: #1b1a18;
            --muted: #5c5852;
            --line: rgba(27, 26, 24, 0.08);
            --amber: #ce7f31;
            --forest: #174c43;
            --rose: #c5574f;
            --shadow: 0 18px 60px rgba(36, 29, 20, 0.12);
        }

        html, body, [class*="css"]  {
            font-family: "Space Grotesk", "Segoe UI", sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at top left, rgba(206, 127, 49, 0.14), transparent 28%),
                radial-gradient(circle at top right, rgba(23, 76, 67, 0.14), transparent 22%),
                linear-gradient(180deg, #f8f4ee 0%, #f1eadf 100%);
        }

        .stApp {
            background: transparent;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.5rem 2.8rem;
            border: 1px solid rgba(255, 255, 255, 0.55);
            border-radius: 30px;
            background:
                linear-gradient(135deg, rgba(255, 248, 237, 0.95), rgba(244, 239, 231, 0.84)),
                linear-gradient(125deg, rgba(23, 76, 67, 0.08), rgba(206, 127, 49, 0.08));
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        .hero:before {
            content: "";
            position: absolute;
            inset: auto -10% -70% auto;
            width: 380px;
            height: 380px;
            background: radial-gradient(circle, rgba(206, 127, 49, 0.22), transparent 60%);
            animation: drift 10s ease-in-out infinite alternate;
        }

        @keyframes drift {
            from { transform: translate(0, 0) scale(1); }
            to { transform: translate(-20px, -10px) scale(1.08); }
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.76rem;
            color: var(--forest);
            margin-bottom: 0.9rem;
            font-weight: 700;
        }

        .hero-title {
            font-family: "Instrument Serif", Georgia, serif;
            font-size: clamp(2.6rem, 5vw, 4.7rem);
            line-height: 0.98;
            margin: 0;
            max-width: 780px;
        }

        .hero-copy {
            margin-top: 1rem;
            color: var(--muted);
            font-size: 1rem;
            max-width: 720px;
            line-height: 1.7;
        }

        .panel, .stat-card, .info-card {
            background: var(--paper);
            border: 1px solid rgba(255, 255, 255, 0.65);
            border-radius: 24px;
            padding: 1.35rem 1.4rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        .panel-title {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: var(--muted);
            margin-bottom: 0.6rem;
            font-weight: 700;
        }

        .stat-card {
            min-height: 170px;
        }

        .stat-label {
            font-size: 0.88rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.14em;
        }

        .stat-value {
            font-size: clamp(2rem, 3vw, 3rem);
            line-height: 1;
            margin-top: 1rem;
            font-weight: 700;
            color: var(--ink);
        }

        .stat-foot {
            margin-top: 0.9rem;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
        }

        .mini-metric {
            background: rgba(255, 255, 255, 0.62);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem;
        }

        .mini-metric .name {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }

        .mini-metric .value {
            margin-top: 0.55rem;
            font-size: 1.5rem;
            font-weight: 700;
        }

        div[data-testid="stNumberInput"], div[data-testid="stSlider"] {
            background: rgba(255, 255, 255, 0.55);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.55rem 0.8rem 0.4rem 0.8rem;
        }

        div[data-testid="stForm"] {
            border: none;
        }

        .stButton > button, div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            border-radius: 999px;
            border: none;
            min-height: 3.2rem;
            background: linear-gradient(135deg, var(--forest), #0f332d);
            color: #f9f4ec;
            font-weight: 700;
            font-size: 0.98rem;
            box-shadow: 0 12px 30px rgba(23, 76, 67, 0.22);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.7rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.5);
            border-radius: 999px;
            padding: 0.5rem 1rem;
            border: 1px solid var(--line);
        }

        .note {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.65;
        }

        .footer-note {
            padding: 1rem 0 0.5rem 0;
            color: var(--muted);
            font-size: 0.88rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(dataset: pd.DataFrame) -> None:
    min_ns = dataset["Nano Silica %"].min()
    max_ns = dataset["Nano Silica %"].max()
    st.markdown(
        f"""
        <section class="hero">
            <div class="eyebrow">FRP Prediction Suite</div>
            <h1 class="hero-title">Premium material intelligence for nano-silica reinforced FRP.</h1>
            <p class="hero-copy">
                Enter a single <strong>Nano Silica %</strong> value and get instant predicted
                <strong>tensile</strong> and <strong>flexural</strong> strength outputs, backed by the
                training dataset and the saved GRU model. Interactive charts below expose model fit,
                residual behavior, and the strongest composition windows across the dataset range
                of <strong>{min_ns:.2f}%</strong> to <strong>{max_ns:.2f}%</strong>.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_panel(predictor: FRPPredictor, default_value: float) -> None:
    left, right = st.columns([1.05, 1.25], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Prediction Console</div>', unsafe_allow_html=True)
        lower, upper = predictor.get_feature_bounds()
        with st.form("prediction_form", clear_on_submit=False):
            nano_silica = st.number_input(
                "Nano Silica (%)",
                min_value=float(lower),
                max_value=float(upper),
                value=float(default_value),
                step=0.1,
                help="Single material composition input used to estimate both strength outputs.",
            )
            submitted = st.form_submit_button("Generate Strength Prediction")

        st.markdown(
            """
            <p class="note">
                The deployed GRU was trained on 5-step sequences. For a single-value UI,
                the entered composition is expanded into a constant 5-step sequence so the
                saved model can still be used without retraining.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        if submitted or "latest_prediction" not in st.session_state:
            try:
                st.session_state["latest_prediction"] = predictor.predict(float(nano_silica))
                st.session_state["prediction_error"] = None
            except PredictorLoadError as exc:
                st.session_state["prediction_error"] = str(exc)
            except Exception as exc:  # pragma: no cover
                st.session_state["prediction_error"] = f"Prediction failed: {exc}"

        error = st.session_state.get("prediction_error")
        if error:
            st.error(error)
            st.info("Install the required dependencies, then rerun the app to enable live prediction.")
            return

        result = st.session_state["latest_prediction"]
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Predicted Tensile Strength</div>
                    <div class="stat-value">{result.tensile_mpa:.2f} MPa</div>
                    <div class="stat-foot">
                        Estimated for <strong>{result.nano_silica:.2f}%</strong> nano silica.
                        Nearest observed sample: {result.nearest_sample_tensile:.2f} MPa at
                        {result.nearest_sample_nano:.2f}%.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
                <div class="stat-card">
                    <div class="stat-label">Predicted Flexural Strength</div>
                    <div class="stat-value">{result.flexural_mpa:.2f} MPa</div>
                    <div class="stat-foot">
                        Dataset-backed comparison point: {result.nearest_sample_flexural:.2f} MPa at
                        {result.nearest_sample_nano:.2f}% nano silica.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metrics_tab() -> None:
    metrics = load_sheet("Evaluation_Metrics")
    metrics_map = dict(zip(metrics["Metric"], metrics["Value"]))
    st.markdown('<div class="panel-title">Model Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="mini-metric">
                <div class="name">R2 Tensile</div>
                <div class="value">{metrics_map.get('R2_Tensile', 0):.3f}</div>
            </div>
            <div class="mini-metric">
                <div class="name">R2 Flexural</div>
                <div class="value">{metrics_map.get('R2_Flexural', 0):.3f}</div>
            </div>
            <div class="mini-metric">
                <div class="name">RMSE Tensile</div>
                <div class="value">{metrics_map.get('RMSE_Tensile', 0):.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="mini-metric">
                <div class="name">RMSE Flexural</div>
                <div class="value">{metrics_map.get('RMSE_Flexural', 0):.2f}</div>
            </div>
            <div class="mini-metric">
                <div class="name">MAE Tensile</div>
                <div class="value">{metrics_map.get('MAE_Tensile', 0):.2f}</div>
            </div>
            <div class="mini-metric">
                <div class="name">MAE Flexural</div>
                <div class="value">{metrics_map.get('MAE_Flexural', 0):.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bar = px.bar(
        metrics,
        x="Value",
        y="Metric",
        orientation="h",
        color="Value",
        color_continuous_scale=["#174c43", "#ce7f31", "#c5574f"],
    )
    bar.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        coloraxis_showscale=False,
        font=dict(family="Space Grotesk"),
    )
    st.plotly_chart(bar, use_container_width=True)


def render_visualization_tab() -> None:
    predictions = load_sheet("Full_Test_Predictions").sort_values("Nano_Silica_%")
    optimal = load_sheet("Optimal_Results")

    multi = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Tensile Response vs Nano Silica", "Actual vs Predicted Fit"),
        horizontal_spacing=0.12,
    )
    multi.add_trace(
        go.Scatter(
            x=predictions["Nano_Silica_%"],
            y=predictions["Actual_Tensile_Stress"],
            mode="lines+markers",
            name="Actual Tensile",
            line=dict(color="#174c43", width=3),
            marker=dict(size=6),
        ),
        row=1,
        col=1,
    )
    multi.add_trace(
        go.Scatter(
            x=predictions["Nano_Silica_%"],
            y=predictions["Predicted_Tensile_Stress"],
            mode="lines",
            name="Predicted Tensile",
            line=dict(color="#ce7f31", width=3, dash="dot"),
        ),
        row=1,
        col=1,
    )
    multi.add_trace(
        go.Scatter(
            x=predictions["Actual_Tensile_Stress"],
            y=predictions["Predicted_Tensile_Stress"],
            mode="markers",
            name="Regression Fit",
            marker=dict(
                size=10,
                color=predictions["Nano_Silica_%"],
                colorscale=["#174c43", "#ce7f31", "#c5574f"],
                showscale=True,
                colorbar=dict(title="Nano Silica %"),
                line=dict(width=0.5, color="rgba(255,255,255,0.7)"),
            ),
        ),
        row=1,
        col=2,
    )
    ref_line = [predictions["Actual_Tensile_Stress"].min(), predictions["Actual_Tensile_Stress"].max()]
    multi.add_trace(
        go.Scatter(
            x=ref_line,
            y=ref_line,
            mode="lines",
            name="Ideal Fit",
            line=dict(color="#1b1a18", width=2, dash="dash"),
        ),
        row=1,
        col=2,
    )
    multi.update_layout(
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Space Grotesk"),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    st.plotly_chart(multi, use_container_width=True)

    residuals = predictions["Actual_Tensile_Stress"] - predictions["Predicted_Tensile_Stress"]
    lower, upper = st.columns([1.15, 1], gap="large")
    with lower:
        histogram = px.histogram(
            x=residuals,
            nbins=20,
            color_discrete_sequence=["#174c43"],
            opacity=0.86,
            labels={"x": "Residual (Actual - Predicted)"},
        )
        histogram.update_layout(
            height=360,
            title="Residual Distribution",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.45)",
            margin=dict(l=10, r=10, t=50, b=10),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(histogram, use_container_width=True)

    with upper:
        bubble = px.scatter(
            optimal,
            x="Nano_Silica_%",
            y="Predicted_Tensile_Stress",
            color="Predicted_Flexural_Stress",
            size="Predicted_Flexural_Stress",
            color_continuous_scale=["#174c43", "#ce7f31", "#c5574f"],
            labels={"Nano_Silica_%": "Nano Silica (%)"},
            title="Optimal Composition Window",
        )
        bubble.update_layout(
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.45)",
            margin=dict(l=10, r=10, t=50, b=10),
            font=dict(family="Space Grotesk"),
        )
        st.plotly_chart(bubble, use_container_width=True)


def render_data_tab(dataset: pd.DataFrame) -> None:
    view = dataset[["Nano Silica %", "Tensile Stress (MPa)", "Flexural Stress (MPa)"]].copy()
    view = view.sort_values("Nano Silica %")

    st.markdown(
        """
        <div class="info-card">
            <div class="panel-title">Training Data</div>
            <p class="note">
                The app uses the local CSV dataset as the source of feature bounds, nearest-sample context,
                and training-time scaling behavior. You can inspect the cleaned prediction inputs directly below.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(view, use_container_width=True, hide_index=True, height=420)

    curve = px.scatter(
        view,
        x="Nano Silica %",
        y=["Tensile Stress (MPa)", "Flexural Stress (MPa)"],
        trendline="ols",
        color_discrete_sequence=["#174c43", "#ce7f31"],
        title="Observed Material Response",
    )
    curve.update_layout(
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Space Grotesk"),
    )
    st.plotly_chart(curve, use_container_width=True)


def main() -> None:
    inject_styles()
    predictor = get_predictor()
    dataset = load_base_dataset()

    render_hero(dataset)
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    render_prediction_panel(predictor, default_value=float(dataset["Nano Silica %"].median()))
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

    metrics_tab, viz_tab, data_tab = st.tabs(["Metrics", "Result Visualizations", "Training Data"])
    with metrics_tab:
        render_metrics_tab()
    with viz_tab:
        render_visualization_tab()
    with data_tab:
        render_data_tab(dataset)

    st.markdown(
        """
        <div class="footer-note">
            Live prediction depends on the local <code>frp_rnn_model_optimized.h5</code> model and TensorFlow runtime.
            Charts and metric panels are loaded from the saved training artifacts in this workspace.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
