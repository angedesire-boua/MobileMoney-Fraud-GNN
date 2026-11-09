from html import escape

import streamlit as st


METRIC_ICONS = {
    "Transactions": "swap_horiz",
    "Alertes": "notifications_active",
    "Alertes actives": "notifications_active",
    "Alertes affichées": "notifications_active",
    "Fraudes réelles": "gpp_bad",
    "Fraudes détectées": "gpp_bad",
    "Vrais positifs": "verified_user",
    "Faux positifs": "rule",
    "Précision": "target",
    "Précision filtre": "target",
    "Volume alerté": "account_balance_wallet",
    "Montant affiché": "account_balance_wallet",
    "Montant": "account_balance_wallet",
    "PR-AUC Test": "monitoring",
    "GraphSAGE": "hub",
    "Dynamic GNN": "history",
    "NoMemory": "memory",
    "LightGBM Core": "account_tree",
    "Modèle": "account_tree",
    "Score": "speed",
    "Score brut": "speed",
    "Risque": "shield",
    "Step": "schedule",
    "Split": "dataset",
    "Contexte": "history",
    "Seuil Validation": "tune",
    "Seuil app": "tune",
    "Seuil choisi sur": "tune",
    "Décision modèle": "fact_check",
    "Décision": "fact_check",
    "Vrai label PaySim": "label",
    "Classification": "category",
    "Mesure finale sur": "analytics",
    "Source": "north_east",
    "Destination": "south_east",
    "Probabilité recalculée": "calculate",
    "Probabilité sauvegardée": "save",
    "Somme contributions": "functions",
    "Meilleur graph-based": "hub",
    "Delta Validation": "compare_arrows",
    "Delta Test": "compare_arrows",
    "Delta": "compare_arrows",
    "Nœuds": "hub",
    "Arêtes": "share",
    "Degré = 1": "linear_scale",
    "Overlap": "join_inner",
    "Source et destination": "join_inner",
    "Nouvelles sources": "north_east",
    "Nouvelles destinations": "south_east",
    "Nouvelles sources Test": "north_east",
    "Nouvelles destinations Test": "south_east",
    "N→K Dynamic": "history",
    "N→K NoMemory": "memory",
}


def render_metric(label, value):
    """Keep native metrics and pair repeated labels with the same icon."""
    icon = METRIC_ICONS.get(label)
    display_label = f":material/{icon}: {label}" if icon else label
    return st.metric(display_label, value)


def get_plot_theme():
    if st.session_state.get("ui_theme", "dark") == "light":
        return {
            "font": "#52667f",
            "strong": "#102743",
            "grid": "rgba(36, 73, 112, 0.11)",
            "edge": "rgba(82, 102, 127, 0.58)",
            "node_outline": "#ffffff"
        }

    return {
        "font": "#9fb0c5",
        "strong": "#f8fafc",
        "grid": "rgba(148, 163, 184, 0.09)",
        "edge": "rgba(125, 145, 167, 0.58)",
        "node_outline": "#dbeafe"
    }


def render_page_header(title, subtitle, context):
    st.markdown(
        (
            '<div class="page-heading">'
            '<div class="page-heading-row">'
            f'<h1>{escape(title)}</h1>'
            f'<div class="page-eyebrow">{escape(context)}</div>'
            '</div>'
            f'<p>{escape(subtitle)}</p>'
            '</div>'
        ),
        unsafe_allow_html=True
    )


def render_panel_header(title, meta=None):
    meta_html = (
        f'<span class="panel-meta">{escape(str(meta))}</span>'
        if meta is not None
        else ""
    )
    st.markdown(
        (
            '<div class="panel-heading">'
            f'<span>{escape(title)}</span>'
            f'{meta_html}'
            '</div>'
        ),
        unsafe_allow_html=True
    )


def render_badge(text, tone="neutral"):
    allowed_tones = {"neutral", "info", "success", "risk", "warning"}
    safe_tone = tone if tone in allowed_tones else "neutral"
    st.markdown(
        f'<span class="status-badge status-badge--{safe_tone}">{escape(str(text))}</span>',
        unsafe_allow_html=True
    )


def format_compact_amount(value):
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f} Md"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} M"
    if value >= 1_000:
        return f"{value / 1_000:.2f} k"
    return f"{value:,.0f}"
