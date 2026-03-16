from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from frp_predictor import PredictionResult


@dataclass(frozen=True)
class ScenarioProfile:
    key: str
    name: str
    sector: str
    description: str
    demand_tensile_mpa: float
    demand_flexural_mpa: float
    demand_pressure_kpa: float
    service_temp_target_c: float
    reference_thickness_mm: float
    reference_area_m2: float
    tensile_utilization: float
    flexural_utilization: float
    pressure_gain: float
    base_cost_per_kg_usd: float
    process_factor: float
    fabrication_waste: float
    density_kg_m3: float
    thermal_offset_c: float
    load_case: str
    failure_mode: str
    use_cases: tuple[str, ...]


@dataclass
class SimulationResult:
    scenario_key: str
    scenario_name: str
    sector: str
    nano_silica: float
    tensile_mpa: float
    flexural_mpa: float
    allowable_tensile_mpa: float
    allowable_flexural_mpa: float
    line_load_kn_per_m: float
    pressure_capacity_kpa: float
    demand_pressure_kpa: float
    pressure_safety_factor: float
    tensile_safety_factor: float
    flexural_safety_factor: float
    governing_safety_factor: float
    service_temperature_c: float
    thermal_margin_c: float
    component_mass_kg: float
    estimated_cost_usd: float
    cost_per_m2_usd: float
    suitability_label: str
    suitability_score: float
    dominant_limit: str
    summary: str
    recommendation: str
    caution: str
    load_case: str
    failure_mode: str
    use_cases: tuple[str, ...]


SCENARIOS: dict[str, ScenarioProfile] = {
    "auto_panel": ScenarioProfile(
        key="auto_panel",
        name="Automotive Body Panel",
        sector="Mobility",
        description="Exterior or semi-structural panel focused on low mass, moderate impact tolerance, and stable dimensional response.",
        demand_tensile_mpa=92.0,
        demand_flexural_mpa=78.0,
        demand_pressure_kpa=145.0,
        service_temp_target_c=82.0,
        reference_thickness_mm=3.2,
        reference_area_m2=1.4,
        tensile_utilization=0.34,
        flexural_utilization=0.41,
        pressure_gain=5.6,
        base_cost_per_kg_usd=13.2,
        process_factor=1.14,
        fabrication_waste=0.05,
        density_kg_m3=1825.0,
        thermal_offset_c=2.0,
        load_case="Distributed aerodynamic and handling pressure with repeated low-amplitude vibration.",
        failure_mode="Local buckling near joints and fatigue cracking at fastener interfaces.",
        use_cases=("Door skins", "Battery covers", "Lightweight body shells"),
    ),
    "bridge_laminate": ScenarioProfile(
        key="bridge_laminate",
        name="Bridge Strengthening Laminate",
        sector="Civil Infrastructure",
        description="Retrofit laminate bonded to concrete or steel members where sustained load transfer and conservative safety factors dominate.",
        demand_tensile_mpa=120.0,
        demand_flexural_mpa=96.0,
        demand_pressure_kpa=265.0,
        service_temp_target_c=68.0,
        reference_thickness_mm=4.8,
        reference_area_m2=2.0,
        tensile_utilization=0.31,
        flexural_utilization=0.36,
        pressure_gain=4.4,
        base_cost_per_kg_usd=15.4,
        process_factor=1.26,
        fabrication_waste=0.07,
        density_kg_m3=1880.0,
        thermal_offset_c=-3.0,
        load_case="Long-duration load transfer under bending, creep exposure, and conservative design checks.",
        failure_mode="Bond-line debonding and compression-side delamination under cyclic traffic loading.",
        use_cases=("Girder strengthening", "Deck slab retrofit", "Crack-arrest strips"),
    ),
    "wind_shell": ScenarioProfile(
        key="wind_shell",
        name="Wind Turbine Shell Skin",
        sector="Energy",
        description="Large-area shell component balancing flexural stiffness, low density, and resistance to fluctuating wind pressure.",
        demand_tensile_mpa=102.0,
        demand_flexural_mpa=88.0,
        demand_pressure_kpa=205.0,
        service_temp_target_c=74.0,
        reference_thickness_mm=5.5,
        reference_area_m2=3.4,
        tensile_utilization=0.33,
        flexural_utilization=0.38,
        pressure_gain=4.9,
        base_cost_per_kg_usd=14.7,
        process_factor=1.22,
        fabrication_waste=0.08,
        density_kg_m3=1805.0,
        thermal_offset_c=1.5,
        load_case="Variable pressure field, shell bending, and fatigue-sensitive cyclic loading.",
        failure_mode="Stiffness loss from matrix cracking before ultimate fiber failure.",
        use_cases=("Blade shell inserts", "Nacelle covers", "Secondary energy housings"),
    ),
    "marine_overlay": ScenarioProfile(
        key="marine_overlay",
        name="Marine Deck Overlay",
        sector="Marine",
        description="Moisture-exposed deck or topside panel where corrosion resistance, moderate strength, and serviceability matter together.",
        demand_tensile_mpa=95.0,
        demand_flexural_mpa=84.0,
        demand_pressure_kpa=180.0,
        service_temp_target_c=70.0,
        reference_thickness_mm=4.2,
        reference_area_m2=1.8,
        tensile_utilization=0.32,
        flexural_utilization=0.40,
        pressure_gain=5.1,
        base_cost_per_kg_usd=16.1,
        process_factor=1.18,
        fabrication_waste=0.06,
        density_kg_m3=1860.0,
        thermal_offset_c=-1.0,
        load_case="Wet-service distributed loading with localized footfall or equipment pressure peaks.",
        failure_mode="Surface wear, core-interface peeling, and moisture-assisted matrix softening.",
        use_cases=("Walkway panels", "Deck infill plates", "Access hatch covers"),
    ),
    "drone_fairing": ScenarioProfile(
        key="drone_fairing",
        name="Drone Fairing / Radome",
        sector="Aerospace Lite",
        description="Very low-mass enclosure prioritizing stiffness-to-weight, smooth finish, and moderate service temperature.",
        demand_tensile_mpa=76.0,
        demand_flexural_mpa=70.0,
        demand_pressure_kpa=120.0,
        service_temp_target_c=62.0,
        reference_thickness_mm=2.4,
        reference_area_m2=0.65,
        tensile_utilization=0.37,
        flexural_utilization=0.44,
        pressure_gain=6.2,
        base_cost_per_kg_usd=18.3,
        process_factor=1.31,
        fabrication_waste=0.09,
        density_kg_m3=1765.0,
        thermal_offset_c=4.0,
        load_case="Low-area aerodynamic pressure with strict mass allowance and vibration sensitivity.",
        failure_mode="Impact denting and stiffness loss near edges or access cutouts.",
        use_cases=("UAV covers", "Sensor fairings", "Electronics radomes"),
    ),
    "chemical_panel": ScenarioProfile(
        key="chemical_panel",
        name="Chemical Equipment Access Panel",
        sector="Process Industry",
        description="Maintenance cover or guard panel where moderate strength and stable thermal performance are required.",
        demand_tensile_mpa=108.0,
        demand_flexural_mpa=90.0,
        demand_pressure_kpa=235.0,
        service_temp_target_c=88.0,
        reference_thickness_mm=4.0,
        reference_area_m2=1.0,
        tensile_utilization=0.30,
        flexural_utilization=0.37,
        pressure_gain=5.0,
        base_cost_per_kg_usd=17.4,
        process_factor=1.28,
        fabrication_waste=0.07,
        density_kg_m3=1895.0,
        thermal_offset_c=5.0,
        load_case="Panelized guarding exposed to pressure pulses, handling loads, and elevated service temperature.",
        failure_mode="Thermal softening followed by edge cracking around hardware penetrations.",
        use_cases=("Machine guards", "Inspection covers", "Enclosure doors"),
    ),
}


def scenario_options() -> dict[str, str]:
    return {profile.name: key for key, profile in SCENARIOS.items()}


def _composition_reliability(nano_silica: float) -> float:
    distance_from_peak = abs(nano_silica - 15.0)
    reliability = 0.9 - 0.011 * distance_from_peak
    return float(np.clip(reliability, 0.72, 0.92))


def _service_temperature(profile: ScenarioProfile, nano_silica: float) -> float:
    nano_gain = 1.45 * nano_silica
    overload_penalty = max(0.0, nano_silica - 18.0) * 1.55
    return 78.0 + profile.thermal_offset_c + nano_gain - overload_penalty


def _suitability_label(safety_factor: float, thermal_margin_c: float) -> str:
    if safety_factor >= 1.8 and thermal_margin_c >= 20:
        return "Excellent"
    if safety_factor >= 1.35 and thermal_margin_c >= 10:
        return "Viable"
    if safety_factor >= 1.0 and thermal_margin_c >= 0:
        return "Conditional"
    return "Redesign"


def _dominant_limit(
    pressure_safety: float,
    tensile_safety: float,
    flexural_safety: float,
    thermal_margin_c: float,
) -> str:
    factors = {
        "Pressure capacity": pressure_safety,
        "Tensile reserve": tensile_safety,
        "Flexural reserve": flexural_safety,
        "Thermal margin": max(0.01, thermal_margin_c / 10.0),
    }
    return min(factors, key=factors.get)


def simulate_application(
    prediction: PredictionResult,
    scenario_key: str,
    thickness_mm: float,
    area_m2: float,
    operating_temp_c: float,
    annual_volume: int,
) -> SimulationResult:
    profile = SCENARIOS[scenario_key]
    reliability = _composition_reliability(prediction.nano_silica)
    thickness_factor = (thickness_mm / profile.reference_thickness_mm) ** 1.15
    area_factor = (profile.reference_area_m2 / max(area_m2, 0.2)) ** 0.23

    allowable_tensile = prediction.tensile_mpa * profile.tensile_utilization * reliability
    allowable_flexural = prediction.flexural_mpa * profile.flexural_utilization * reliability
    line_load_kn_per_m = allowable_tensile * thickness_mm * 0.92

    pressure_capacity_kpa = min(allowable_tensile, allowable_flexural) * profile.pressure_gain * thickness_factor * area_factor
    tensile_safety = allowable_tensile / profile.demand_tensile_mpa
    flexural_safety = allowable_flexural / profile.demand_flexural_mpa
    pressure_safety = pressure_capacity_kpa / profile.demand_pressure_kpa
    governing_safety = min(pressure_safety, tensile_safety, flexural_safety)

    service_temperature_c = _service_temperature(profile, prediction.nano_silica)
    thermal_margin_c = service_temperature_c - operating_temp_c

    density = profile.density_kg_m3 + prediction.nano_silica * 2.8
    volume_m3 = area_m2 * (thickness_mm / 1000.0)
    mass_kg = density * volume_m3
    additive_cost = 0.22 * prediction.nano_silica + 0.006 * prediction.nano_silica**2
    processing_penalty = 1.0 + max(0.0, prediction.nano_silica - 12.0) * 0.012
    volume_discount = 1.0 - min(0.12, np.log10(max(annual_volume, 1)) * 0.035)
    cost_per_kg = (profile.base_cost_per_kg_usd + additive_cost) * profile.process_factor * processing_penalty
    estimated_cost = mass_kg * cost_per_kg * (1.0 + profile.fabrication_waste) * volume_discount
    cost_per_m2 = estimated_cost / max(area_m2, 0.1)

    safety_component = min(100.0, governing_safety * 48.0)
    thermal_component = float(np.clip((thermal_margin_c + 15.0) * 2.3, 0.0, 100.0))
    cost_component = float(np.clip(100.0 - max(0.0, cost_per_m2 - 70.0) * 0.6, 15.0, 100.0))
    suitability_score = 0.5 * safety_component + 0.25 * thermal_component + 0.25 * cost_component
    label = _suitability_label(governing_safety, thermal_margin_c)
    dominant_limit = _dominant_limit(pressure_safety, tensile_safety, flexural_safety, thermal_margin_c)

    if label == "Excellent":
        recommendation = f"Strong candidate for {profile.name.lower()} service with comfortable reserve."
    elif label == "Viable":
        recommendation = f"Suitable for {profile.name.lower()} duty with standard design conservatism."
    elif label == "Conditional":
        recommendation = f"Can be used for {profile.name.lower()} only with thickness or operating-condition refinement."
    else:
        recommendation = f"Current composition is not recommended for {profile.name.lower()} without redesign."

    summary = (
        f"At {prediction.nano_silica:.2f}% nano silica, the model predicts "
        f"{prediction.tensile_mpa:.1f} MPa tensile and {prediction.flexural_mpa:.1f} MPa flexural strength. "
        f"For the selected {profile.name.lower()} case, the governing safety factor is {governing_safety:.2f} "
        f"with an estimated pressure capacity of {pressure_capacity_kpa:.1f} kPa and service temperature envelope "
        f"up to {service_temperature_c:.1f} C."
    )

    caution = (
        f"Primary risk: {profile.failure_mode} "
        f"The most constrained dimension in this estimate is {dominant_limit.lower()}."
    )

    return SimulationResult(
        scenario_key=profile.key,
        scenario_name=profile.name,
        sector=profile.sector,
        nano_silica=prediction.nano_silica,
        tensile_mpa=prediction.tensile_mpa,
        flexural_mpa=prediction.flexural_mpa,
        allowable_tensile_mpa=allowable_tensile,
        allowable_flexural_mpa=allowable_flexural,
        line_load_kn_per_m=line_load_kn_per_m,
        pressure_capacity_kpa=pressure_capacity_kpa,
        demand_pressure_kpa=profile.demand_pressure_kpa,
        pressure_safety_factor=pressure_safety,
        tensile_safety_factor=tensile_safety,
        flexural_safety_factor=flexural_safety,
        governing_safety_factor=governing_safety,
        service_temperature_c=service_temperature_c,
        thermal_margin_c=thermal_margin_c,
        component_mass_kg=mass_kg,
        estimated_cost_usd=estimated_cost,
        cost_per_m2_usd=cost_per_m2,
        suitability_label=label,
        suitability_score=suitability_score,
        dominant_limit=dominant_limit,
        summary=summary,
        recommendation=recommendation,
        caution=caution,
        load_case=profile.load_case,
        failure_mode=profile.failure_mode,
        use_cases=profile.use_cases,
    )


def rank_all_scenarios(
    prediction: PredictionResult,
    thickness_mm: float,
    area_m2: float,
    operating_temp_c: float,
    annual_volume: int,
) -> pd.DataFrame:
    rows = []
    for scenario_key in SCENARIOS:
        result = simulate_application(
            prediction=prediction,
            scenario_key=scenario_key,
            thickness_mm=thickness_mm,
            area_m2=area_m2,
            operating_temp_c=operating_temp_c,
            annual_volume=annual_volume,
        )
        rows.append(
            {
                "Scenario": result.scenario_name,
                "Sector": result.sector,
                "Suitability Score": result.suitability_score,
                "Suitability": result.suitability_label,
                "Safety Factor": result.governing_safety_factor,
                "Pressure Capacity (kPa)": result.pressure_capacity_kpa,
                "Service Temp (C)": result.service_temperature_c,
                "Estimated Cost (USD)": result.estimated_cost_usd,
                "Dominant Limit": result.dominant_limit,
            }
        )
    return pd.DataFrame(rows).sort_values(["Suitability Score", "Safety Factor"], ascending=[False, False])
