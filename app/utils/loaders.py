from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st


# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "app"

MANIFEST_PATH = ARTIFACTS_DIR / "app_manifest.json"

PREDICTIONS_PATH = (
    ARTIFACTS_DIR
    / "lightgbm_core_test_predictions.parquet"
)

ALERTS_PATH = (
    ARTIFACTS_DIR
    / "lightgbm_core_test_alerts.parquet"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lightgbm_core_baseline.txt"
)

ML_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paysim_ml_features.parquet"
)

GRAPH_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "paysim_graph_ready.parquet"
)

DYNAMIC_TEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "dynamic_gnn"
    / "dynamic_gnn_final_test_predictions.npz"
)


@st.cache_data
def load_manifest():
    # Charger le manifeste
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_predictions():
    # Charger les prédictions Test
    return pd.read_parquet(PREDICTIONS_PATH)


@st.cache_data
def load_alerts():
    # Charger les alertes
    return pd.read_parquet(ALERTS_PATH)


@st.cache_resource
def load_core_model():
    # Charger LightGBM Core
    return lgb.Booster(
        model_file=str(MODEL_PATH)
    )


@st.cache_data
def load_known_status():
    # Charger le statut Known/New du Test
    with np.load(
        DYNAMIC_TEST_PATH,
        allow_pickle=False
    ) as data:
        return {
            "src_known": data["src_known"].copy(),
            "dst_known": data["dst_known"].copy()
        }

GNN_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "gnn"
    / "gnn_baseline_summary.json"
)

DYNAMIC_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "dynamic_gnn"
    / "dynamic_gnn_final_summary.json"
)

GRAPH_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "graph"
    / "graph_construction_metadata.json"
)


@st.cache_data
def load_gnn_summary():
    # Charger les résultats GraphSAGE
    with open(
        GNN_SUMMARY_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


@st.cache_data
def load_dynamic_summary():
    # Charger les résultats dynamiques
    with open(
        DYNAMIC_SUMMARY_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


@st.cache_data
def load_graph_metadata():
    # Charger les métadonnées du graphe
    with open(
        GRAPH_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)
    
def load_global_style():
    # Charger le style global
    style_path = (
        PROJECT_ROOT
        / "app"
        / "assets"
        / "style.css"
    )

    theme_path = (
        PROJECT_ROOT
        / "app"
        / "assets"
        / "theme-light.css"
    )

    styles = []
    if style_path.exists():
        styles.append(style_path.read_text(encoding="utf-8"))

    if (
        st.session_state.get("ui_theme", "dark") == "light"
        and theme_path.exists()
    ):
        styles.append(theme_path.read_text(encoding="utf-8"))

    if styles:
        st.markdown(
            f"<style>{'\n'.join(styles)}</style>",
            unsafe_allow_html=True
        )
