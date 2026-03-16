from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from advanced_estimator import SCENARIOS, rank_all_scenarios, scenario_options, simulate_application
from frp_predictor import FRPPredictor, PredictorLoadError, PredictionResult


APP_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="FRP Composite Command Center",
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


@st.cache_data(show_spinner=False)
def build_sweep(
    scenario_key: str,
    thickness_mm: float,
    area_m2: float,
    operating_temp_c: float,
    annual_volume: int,
) -> pd.DataFrame:
    predictor = get_predictor()
    lower, upper = predictor.get_feature_bounds()
    nano_values = np.linspace(lower, upper, 64)
    predictions = predictor.predict_many(nano_values.tolist())
    rows = []
    for prediction in predictions:
        simulation = simulate_application(
            prediction=prediction,
            scenario_key=scenario_key,
            thickness_mm=thickness_mm,
            area_m2=area_m2,
            operating_temp_c=operating_temp_c,
            annual_volume=annual_volume,
        )
        rows.append(
            {
                "Nano Silica %": prediction.nano_silica,
                "Predicted Tensile (MPa)": prediction.tensile_mpa,
                "Predicted Flexural (MPa)": prediction.flexural_mpa,
                "Pressure Capacity (kPa)": simulation.pressure_capacity_kpa,
                "Service Temperature (C)": simulation.service_temperature_c,
                "Suitability Score": simulation.suitability_score,
                "Safety Factor": simulation.governing_safety_factor,
            }
        )
    return pd.DataFrame(rows)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=Cormorant+Garamond:wght@500;600;700&display=swap');

        :root {
            --sand: #f1e9dc;
            --paper: rgba(255, 251, 245, 0.76);
            --ink: #181613;
            --muted: #5f5a53;
            --line: rgba(24, 22, 19, 0.08);
            --forest: #143f39;
            --amber: #c47a28;
            --clay: #a84a3f;
            --mist: rgba(255, 255, 255, 0.55);
            --shadow: 0 24px 70px rgba(34, 28, 22, 0.13);
        }

        html, body, [class*="css"] {
            font-family: "Sora", "Segoe UI", sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at 15% 10%, rgba(196, 122, 40, 0.18), transparent 22%),
                radial-gradient(circle at 82% 12%, rgba(20, 63, 57, 0.16), transparent 18%),
                linear-gradient(180deg, #faf5ee 0%, #efe5d7 100%);
        }

        .stApp {
            background: transparent;
        }

        .block-container {
            max-width: 1380px;
            padding-top: 1.75rem;
            padding-bottom: 2.5rem;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.6rem 2.9rem 2.8rem 2.9rem;
            border-radius: 32px;
            border: 1px solid rgba(255, 255, 255, 0.72);
            background:
                linear-gradient(135deg, rgba(255, 248, 238, 0.92), rgba(242, 233, 220, 0.86)),
                linear-gradient(115deg, rgba(20, 63, 57, 0.08), rgba(196, 122, 40, 0.08));
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        .hero::before {
            content: "";
            position: absolute;
            right: -8%;
            top: -20%;
            width: 360px;
            height: 360px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(20, 63, 57, 0.2), transparent 62%);
            animation: pulse 10s ease-in-out infinite alternate;
        }

        .hero::after {
            content: "";
            position: absolute;
            left: 54%;
            bottom: -38%;
            width: 420px;
            height: 420px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(196, 122, 40, 0.17), transparent 62%);
        }

        @keyframes pulse {
            from { transform: translateY(0px) scale(1); }
            to { transform: translateY(16px) scale(1.08); }
        }

        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.18em;
            font-size: 0.76rem;
            font-weight: 700;
            color: var(--forest);
            margin-bottom: 0.9rem;
        }

        .hero-title {
            margin: 0;
            max-width: 900px;
            font-family: "Cormorant Garamond", serif;
            font-size: clamp(3rem, 5vw, 5rem);
            line-height: 0.95;
            font-weight: 700;
        }

        .hero-copy {
            max-width: 780px;
            margin-top: 1rem;
            color: var(--muted);
            line-height: 1.7;
            font-size: 1rem;
        }

        .hero-band {
            display: inline-flex;
            align-items: center;
            margin-top: 1.2rem;
            gap: 0.75rem;
            padding: 0.7rem 1rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.45);
            border: 1px solid rgba(255,255,255,0.7);
            font-size: 0.88rem;
            color: var(--muted);
        }

        .panel, .glass-card, .note-card {
            background: var(--paper);
            border: 1px solid rgba(255,255,255,0.72);
            border-radius: 26px;
            padding: 1.35rem 1.45rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
        }

        .panel-title {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.65rem;
        }

        .metric-card {
            background: var(--paper);
            border-radius: 24px;
            padding: 1.25rem 1.35rem;
            border: 1px solid rgba(255,255,255,0.7);
            box-shadow: var(--shadow);
            min-height: 168px;
        }

        .metric-label {
            font-size: 0.8rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .metric-value {
            margin-top: 0.85rem;
            font-size: clamp(1.9rem, 3vw, 3rem);
            line-height: 1;
            font-weight: 700;
        }

        .metric-foot {
            margin-top: 0.85rem;
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.6;
        }

        .callout {
            padding: 1.1rem 1.2rem;
            border-radius: 20px;
            background: rgba(255,255,255,0.6);
            border: 1px solid var(--line);
            color: var(--muted);
            line-height: 1.65;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.42rem 0.88rem;
            font-size: 0.8rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 700;
            border: 1px solid transparent;
        }

        .badge.good {
            color: #0b4638;
            background: rgba(20, 63, 57, 0.12);
            border-color: rgba(20, 63, 57, 0.16);
        }

        .badge.mid {
            color: #8a4f08;
            background: rgba(196, 122, 40, 0.12);
            border-color: rgba(196, 122, 40, 0.18);
        }

        .badge.bad {
            color: #8b3126;
            background: rgba(168, 74, 63, 0.12);
            border-color: rgba(168, 74, 63, 0.18);
        }

        div[data-testid="stNumberInput"],
        div[data-testid="stSelectbox"],
        div[data-testid="stSlider"] {
            background: rgba(255,255,255,0.56);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.48rem 0.8rem 0.3rem 0.8rem;
        }

        div[data-testid="stForm"] {
            border: none;
        }

        .stButton > button, div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            min-height: 3.15rem;
            border: none;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--forest), #0f312c);
            color: #fffaf2;
            font-weight: 700;
            box-shadow: 0 16px 34px rgba(20, 63, 57, 0.22);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.65rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.42);
            border-radius: 999px;
            padding: 0.52rem 1rem;
            border: 1px solid var(--line);
        }

        .small-note, .footer-note {
            color: var(--muted);
            line-height: 1.65;
            font-size: 0.9rem;
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
            <div class="eyebrow">Advanced FRP Estimator</div>
            <h1 class="hero-title">From nano-silica percentage to structural behavior, thermal envelope, cost, and application fit.</h1>
            <p class="hero-copy">
                This interface keeps your GRU model for strength prediction and layers an engineering simulation on top of it.
                Enter the composition, choose a real deployment scenario, and the app estimates tensile and flexural response,
                pressure resistance, service temperature capability, manufacturing cost, and recommended end-use sectors.
            </p>
            <div class="hero-band">
                Training range: {min_ns:.2f}% to {max_ns:.2f}% nano silica
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def label_class(label: str) -> str:
    if label == "Excellent":
        return "good"
    if label == "Viable":
        return "mid"
    if label == "Conditional":
        return "mid"
    return "bad"


def run_estimation(
    nano_silica: float,
    scenario_key: str,
    thickness_mm: float,
    area_m2: float,
    operating_temp_c: float,
    annual_volume: int,
) -> tuple[PredictionResult, object]:
    predictor = get_predictor()
    prediction = predictor.predict(nano_silica)
    simulation = simulate_application(
        prediction=prediction,
        scenario_key=scenario_key,
        thickness_mm=thickness_mm,
        area_m2=area_m2,
        operating_temp_c=operating_temp_c,
        annual_volume=annual_volume,
    )
    return prediction, simulation


def render_control_deck(dataset: pd.DataFrame) -> None:
    predictor = get_predictor()
    lower, upper = predictor.get_feature_bounds()
    scenario_names = scenario_options()
    scenario_labels = list(scenario_names.keys())

    default_name = "Wind Turbine Shell Skin"
    default_index = scenario_labels.index(default_name) if default_name in scenario_labels else 0

    left, right = st.columns([0.95, 1.35], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Simulation Controls</div>', unsafe_allow_html=True)
        with st.form("advanced_estimator_form", clear_on_submit=False):
            selected_scenario_name = st.selectbox("Application Scenario", scenario_labels, index=default_index)
            nano_silica = st.slider(
                "Nano Silica (%)",
                min_value=float(lower),
                max_value=float(upper),
                value=float(round(float(dataset["Nano Silica %"].median()), 2)),
                step=0.1,
            )
            thickness_mm = st.slider("Panel Thickness (mm)", min_value=2.0, max_value=8.0, value=4.5, step=0.1)
            area_m2 = st.number_input("Component Area (m²)", min_value=0.2, max_value=8.0, value=1.5, step=0.1)
            operating_temp_c = st.slider("Operating Temperature (°C)", min_value=20, max_value=140, value=65, step=1)
            annual_volume = st.number_input("Production Volume (units/year)", min_value=1, max_value=100000, value=250, step=25)
            submitted = st.form_submit_button("Run Advanced Estimator")

        st.markdown(
            """
            <p class="small-note">
                Strength comes from the trained GRU model. Pressure, thermal, cost, and use-case outputs are simulation-based
                estimates derived from the predicted strengths plus explicit engineering assumptions for each scenario.
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if submitted or "estimation_state" not in st.session_state:
        scenario_key = scenario_names[selected_scenario_name]
        try:
            prediction, simulation = run_estimation(
                nano_silica=float(nano_silica),
                scenario_key=scenario_key,
                thickness_mm=float(thickness_mm),
                area_m2=float(area_m2),
                operating_temp_c=float(operating_temp_c),
                annual_volume=int(annual_volume),
            )
            st.session_state["estimation_state"] = {
                "prediction": prediction,
                "simulation": simulation,
                "inputs": {
                    "scenario_key": scenario_key,
                    "scenario_name": selected_scenario_name,
                    "thickness_mm": float(thickness_mm),
                    "area_m2": float(area_m2),
                    "operating_temp_c": float(operating_temp_c),
                    "annual_volume": int(annual_volume),
                },
            }
            st.session_state["estimation_error"] = None
        except PredictorLoadError as exc:
            st.session_state["estimation_error"] = str(exc)
        except Exception as exc:  # pragma: no cover
            st.session_state["estimation_error"] = f"Estimator failed: {exc}"

    with right:
        error = st.session_state.get("estimation_error")
        if error:
            st.error(error)
            return

        state = st.session_state["estimation_state"]
        simulation = state["simulation"]
        badge_class = label_class(simulation.suitability_label)
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="panel-title">Executive Summary</div>
                <div class="badge {badge_class}">{simulation.suitability_label}</div>
                <div style="height: 12px;"></div>
                <div class="callout">{simulation.summary}</div>
                <div style="height: 12px;"></div>
                <div class="callout">
                    <strong>Recommendation:</strong> {simulation.recommendation}<br>
                    <strong>Primary limitation:</strong> {simulation.dominant_limit}<br>
                    <strong>Target use cases:</strong> {", ".join(simulation.use_cases)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_metric_cards() -> None:
    state = st.session_state["estimation_state"]
    prediction = state["prediction"]
    simulation = state["simulation"]

    cols = st.columns(6, gap="medium")
    cards = [
        (
            "Predicted Tensile",
            f"{prediction.tensile_mpa:.1f} MPa",
            f"Nearest sample: {prediction.nearest_sample_tensile:.1f} MPa at {prediction.nearest_sample_nano:.2f}% nano silica.",
        ),
        (
            "Predicted Flexural",
            f"{prediction.flexural_mpa:.1f} MPa",
            f"Nearest sample flexural response: {prediction.nearest_sample_flexural:.1f} MPa.",
        ),
        (
            "Pressure Capacity",
            f"{simulation.pressure_capacity_kpa:.1f} kPa",
            f"Scenario demand: {simulation.demand_pressure_kpa:.1f} kPa. Safety factor: {simulation.pressure_safety_factor:.2f}.",
        ),
        (
            "Service Temperature",
            f"{simulation.service_temperature_c:.1f} C",
            f"Thermal margin above operating condition: {simulation.thermal_margin_c:.1f} C.",
        ),
        (
            "Manufacturing Cost",
            f"${simulation.estimated_cost_usd:.2f}",
            f"Estimated component mass {simulation.component_mass_kg:.2f} kg. Cost intensity ${simulation.cost_per_m2_usd:.2f}/m².",
        ),
        (
            "Overall Suitability",
            f"{simulation.suitability_score:.1f}/100",
            f"Governing safety factor: {simulation.governing_safety_factor:.2f}. Dominant limit: {simulation.dominant_limit}.",
        ),
    ]

    for col, card in zip(cols, cards):
        label, value, foot = card
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-foot">{foot}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_simulation_tab() -> None:
    state = st.session_state["estimation_state"]
    prediction = state["prediction"]
    simulation = state["simulation"]
    inputs = state["inputs"]

    sweep = build_sweep(
        scenario_key=inputs["scenario_key"],
        thickness_mm=inputs["thickness_mm"],
        area_m2=inputs["area_m2"],
        operating_temp_c=inputs["operating_temp_c"],
        annual_volume=inputs["annual_volume"],
    )

    trend = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Strength vs Nano Silica", "Scenario Capacity vs Nano Silica"),
        horizontal_spacing=0.12,
    )
    trend.add_trace(
        go.Scatter(
            x=sweep["Nano Silica %"],
            y=sweep["Predicted Tensile (MPa)"],
            mode="lines",
            name="Tensile",
            line=dict(color="#143f39", width=3),
        ),
        row=1,
        col=1,
    )
    trend.add_trace(
        go.Scatter(
            x=sweep["Nano Silica %"],
            y=sweep["Predicted Flexural (MPa)"],
            mode="lines",
            name="Flexural",
            line=dict(color="#c47a28", width=3),
        ),
        row=1,
        col=1,
    )
    trend.add_vline(x=prediction.nano_silica, line_dash="dash", line_color="#a84a3f", row=1, col=1)
    trend.add_trace(
        go.Scatter(
            x=sweep["Nano Silica %"],
            y=sweep["Pressure Capacity (kPa)"],
            mode="lines",
            name="Pressure Capacity",
            line=dict(color="#143f39", width=3),
        ),
        row=1,
        col=2,
    )
    trend.add_trace(
        go.Scatter(
            x=sweep["Nano Silica %"],
            y=sweep["Service Temperature (C)"],
            mode="lines",
            name="Service Temperature",
            line=dict(color="#a84a3f", width=3, dash="dot"),
        ),
        row=1,
        col=2,
    )
    trend.add_hline(
        y=simulation.demand_pressure_kpa,
        line_dash="dash",
        line_color="#c47a28",
        annotation_text="Pressure demand",
        row=1,
        col=2,
    )
    trend.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        font=dict(family="Sora"),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    st.plotly_chart(trend, use_container_width=True)

    lower, upper = st.columns([1.05, 1], gap="large")
    with lower:
        radar = go.Figure()
        radar.add_trace(
            go.Scatterpolar(
                r=[
                    min(100, prediction.tensile_mpa / 4.2),
                    min(100, prediction.flexural_mpa / 3.3),
                    min(100, simulation.pressure_capacity_kpa / 4.0),
                    min(100, simulation.service_temperature_c),
                    min(100, simulation.suitability_score),
                ],
                theta=["Tensile", "Flexural", "Pressure", "Thermal", "Suitability"],
                fill="toself",
                name="Performance Envelope",
                line=dict(color="#143f39", width=3),
                fillcolor="rgba(20,63,57,0.18)",
            )
        )
        radar.update_layout(
            height=400,
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(255,255,255,0.45)",
                radialaxis=dict(visible=True, range=[0, 100]),
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            title="Composite Performance Envelope",
            font=dict(family="Sora"),
        )
        st.plotly_chart(radar, use_container_width=True)

    with upper:
        indicator = make_subplots(
            rows=2,
            cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}], [{"type": "indicator"}, {"type": "indicator"}]],
            horizontal_spacing=0.1,
            vertical_spacing=0.12,
        )
        indicator.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=simulation.governing_safety_factor,
                title={"text": "Safety Factor"},
                gauge={"axis": {"range": [0, 3]}, "bar": {"color": "#143f39"}, "steps": [{"range": [0, 1], "color": "#f1c7c1"}, {"range": [1, 1.35], "color": "#f1dfc4"}, {"range": [1.35, 3], "color": "#d7ece7"}]},
            ),
            row=1,
            col=1,
        )
        indicator.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=simulation.pressure_capacity_kpa,
                title={"text": "Pressure (kPa)"},
                gauge={"axis": {"range": [0, max(simulation.demand_pressure_kpa * 2.2, simulation.pressure_capacity_kpa * 1.2)]}, "bar": {"color": "#c47a28"}},
            ),
            row=1,
            col=2,
        )
        indicator.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=simulation.service_temperature_c,
                title={"text": "Temp (C)"},
                gauge={"axis": {"range": [0, 160]}, "bar": {"color": "#a84a3f"}},
            ),
            row=2,
            col=1,
        )
        indicator.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=simulation.estimated_cost_usd,
                title={"text": "Cost (USD)"},
                gauge={"axis": {"range": [0, max(250, simulation.estimated_cost_usd * 1.35)]}, "bar": {"color": "#143f39"}},
            ),
            row=2,
            col=2,
        )
        indicator.update_layout(height=400, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Sora"))
        st.plotly_chart(indicator, use_container_width=True)

    st.markdown(
        f"""
        <div class="note-card">
            <div class="panel-title">Scenario Reading</div>
            <p class="small-note">
                <strong>Load case:</strong> {simulation.load_case}<br>
                <strong>Primary risk:</strong> {simulation.failure_mode}<br>
                <strong>Use it for:</strong> {", ".join(simulation.use_cases)}<br>
                <strong>Line load capacity:</strong> {simulation.line_load_kn_per_m:.1f} kN/m<br>
                <strong>Caution:</strong> {simulation.caution}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ranking_tab() -> None:
    state = st.session_state["estimation_state"]
    prediction = state["prediction"]
    inputs = state["inputs"]
    ranking = rank_all_scenarios(
        prediction=prediction,
        thickness_mm=inputs["thickness_mm"],
        area_m2=inputs["area_m2"],
        operating_temp_c=inputs["operating_temp_c"],
        annual_volume=inputs["annual_volume"],
    )

    bar = px.bar(
        ranking,
        x="Suitability Score",
        y="Scenario",
        color="Safety Factor",
        orientation="h",
        color_continuous_scale=["#a84a3f", "#c47a28", "#143f39"],
    )
    bar.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        font=dict(family="Sora"),
    )
    st.plotly_chart(bar, use_container_width=True)
    st.dataframe(ranking.round(2), use_container_width=True, hide_index=True, height=320)


def render_metrics_tab() -> None:
    metrics = load_sheet("Evaluation_Metrics")
    metrics_map = dict(zip(metrics["Metric"], metrics["Value"]))

    top = st.columns(6, gap="medium")
    metric_items = [
        ("R2 Tensile", metrics_map.get("R2_Tensile", 0.0)),
        ("R2 Flexural", metrics_map.get("R2_Flexural", 0.0)),
        ("RMSE Tensile", metrics_map.get("RMSE_Tensile", 0.0)),
        ("RMSE Flexural", metrics_map.get("RMSE_Flexural", 0.0)),
        ("MAE Tensile", metrics_map.get("MAE_Tensile", 0.0)),
        ("MAE Flexural", metrics_map.get("MAE_Flexural", 0.0)),
    ]
    for col, item in zip(top, metric_items):
        label, value = item
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value:.3f}</div>
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
        color_continuous_scale=["#143f39", "#c47a28", "#a84a3f"],
    )
    bar.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        coloraxis_showscale=False,
        font=dict(family="Sora"),
    )
    st.plotly_chart(bar, use_container_width=True)


def render_data_tab(dataset: pd.DataFrame) -> None:
    view = dataset[["Nano Silica %", "Tensile Stress (MPa)", "Flexural Stress (MPa)"]].copy()
    view = view.sort_values("Nano Silica %")

    curve = px.scatter(
        view,
        x="Nano Silica %",
        y=["Tensile Stress (MPa)", "Flexural Stress (MPa)"],
        trendline="ols",
        color_discrete_sequence=["#143f39", "#c47a28"],
        title="Observed Mechanical Response Across the Training Set",
    )
    curve.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        font=dict(family="Sora"),
    )
    st.plotly_chart(curve, use_container_width=True)
    st.dataframe(view.round(3), use_container_width=True, hide_index=True, height=320)


def render_assumptions_tab() -> None:
    st.markdown(
        """
        <div class="note-card">
            <div class="panel-title">Estimator Assumptions</div>
            <p class="small-note">
                1. Tensile and flexural strength come directly from the trained GRU model using the saved <code>.h5</code> file.<br>
                2. Because the deployed model expects a 5-step sequence, a single nano-silica input is expanded into a constant 5-step sequence during inference.<br>
                3. Pressure capacity is estimated from the governing allowable stress, panel thickness, component area, and a scenario-specific transfer coefficient.<br>
                4. Service temperature is a simulation-based thermal envelope derived from composition and scenario offsets, not from a separately trained thermal dataset.<br>
                5. Cost is estimated from density, part volume, nano-silica loading premium, fabrication difficulty, waste factor, and production-volume discount.<br>
                6. Suitability score combines safety reserve, thermal margin, and cost efficiency for comparative project-level analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    assumptions_df = pd.DataFrame(
        [
            {"Scenario": profile.name, "Sector": profile.sector, "Pressure Demand (kPa)": profile.demand_pressure_kpa, "Target Temp (C)": profile.service_temp_target_c, "Reference Thickness (mm)": profile.reference_thickness_mm}
            for profile in SCENARIOS.values()
        ]
    )
    st.dataframe(assumptions_df, use_container_width=True, hide_index=True)


def main() -> None:
    inject_styles()
    dataset = load_base_dataset()

    render_hero(dataset)
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    render_control_deck(dataset)

    if st.session_state.get("estimation_error"):
        return

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    render_metric_cards()
    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    sim_tab, rank_tab, metrics_tab, data_tab, assumptions_tab = st.tabs(
        ["Scenario Simulation", "Scenario Ranking", "Model Metrics", "Training Data", "Assumptions"]
    )
    with sim_tab:
        render_simulation_tab()
    with rank_tab:
        render_ranking_tab()
    with metrics_tab:
        render_metrics_tab()
    with data_tab:
        render_data_tab(dataset)
    with assumptions_tab:
        render_assumptions_tab()

    st.markdown(
        """
        <div class="footer-note">
            This dashboard is strongest when presented as a hybrid system: machine learning for strength prediction,
            and engineering simulation for application-level estimates.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
