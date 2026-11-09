import pandas as pd
import plotly.express as px
import streamlit as st

from utils.auth import require_authentication
from utils.research_network import render_alert_network
from utils.loaders import (
    load_dynamic_summary,
    load_global_style,
    load_gnn_summary,
    load_graph_metadata,
    load_manifest
)
from utils.ui import (
    get_plot_theme,
    render_badge,
    render_metric,
    render_page_header,
    render_panel_header
)


require_authentication()
load_global_style()

manifest = load_manifest()
gnn_summary = load_gnn_summary()
dynamic_summary = load_dynamic_summary()
graph_metadata = load_graph_metadata()


def apply_plot_theme(fig, height, top=18):
    theme = get_plot_theme()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=top, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme["font"], size=10),
        legend=dict(
            font=dict(color=theme["font"]),
            title=dict(font=dict(color=theme["font"])),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(
        gridcolor=theme["grid"],
        zeroline=False,
        tickfont=dict(color=theme["font"]),
        title_font=dict(color=theme["font"])
    )
    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        tickfont=dict(color=theme["font"]),
        title_font=dict(color=theme["font"])
    )
    return fig


dynamic_validation = float(
    dynamic_summary["best_dynamic_model"]["validation_pr_auc"]
)
dynamic_test = float(
    dynamic_summary["best_dynamic_model"]["test_pr_auc"]
)
control_validation = float(
    dynamic_summary["no_memory_control"]["validation_pr_auc"]
)
control_test = float(
    dynamic_summary["no_memory_control"]["test_pr_auc"]
)
lightgbm_test = float(manifest["test_metrics"]["pr_auc"])
graphsage_test = float(gnn_summary["test"]["pr_auc"])
threshold = float(manifest["model"]["threshold"])
delta_test = dynamic_test - control_test
delta_validation = dynamic_validation - control_validation

model_results = pd.DataFrame(
    {
        "Modèle": [
            "GraphSAGE",
            "Dynamic GNN",
            "NoMemory",
            "LightGBM Core"
        ],
        "PR-AUC Test": [
            graphsage_test,
            dynamic_test,
            control_test,
            lightgbm_test
        ]
    }
)

render_page_header(
    "Recherche",
    "Résultats expérimentaux, limites structurelles et lecture scientifique.",
    "PaySim · comparaison des familles de modèles"
)

kpi_cols = st.columns(4, gap="small")
research_kpis = [
    (
        "graphsage",
        "GraphSAGE",
        f"{graphsage_test:.4f}",
        "PR-AUC Test"
    ),
    (
        "dynamic",
        "Dynamic GNN",
        f"{dynamic_test:.4f}",
        "PR-AUC Test"
    ),
    (
        "nomemory",
        "NoMemory",
        f"{control_test:.4f}",
        "PR-AUC Test"
    ),
    (
        "lightgbm",
        "LightGBM Core",
        f"{lightgbm_test:.4f}",
        "PR-AUC Test · modèle retenu"
    )
]

for col, (key, label, value, note) in zip(kpi_cols, research_kpis):
    with col:
        with st.container(border=True, key=f"kpi_research_{key}"):
            render_metric(label, value)
            st.caption(note)

st.session_state.setdefault("research_tabs", st.session_state.get("research_active_tab", "Comparaison"))
comparison_tab, memory_tab, structure_tab, network_tab, conclusion_tab = st.tabs(
    [
        "Comparaison",
        "Mémoire temporelle",
        "Structure PaySim",
        "Réseau des alertes",
        "Conclusion"
    ],
    key="research_tabs",
    on_change="rerun",
)
st.session_state["research_active_tab"] = st.session_state["research_tabs"]

with comparison_tab:
    chart_col, insight_col = st.columns([1.55, 1], gap="small")

    with chart_col:
        with st.container(
            border=True,
            key="panel_research_models",
            height="stretch"
        ):
            render_panel_header(
                "Comparaison des modèles",
                "PR-AUC sur le split Test"
            )
            ordered_results = model_results.sort_values("PR-AUC Test")
            fig_models = px.bar(
                ordered_results,
                x="PR-AUC Test",
                y="Modèle",
                orientation="h",
                text="PR-AUC Test",
                color="Modèle",
                color_discrete_map={
                    "GraphSAGE": "#64748b",
                    "Dynamic GNN": "#38bdf8",
                    "NoMemory": "#3b82f6",
                    "LightGBM Core": "#22c55e"
                }
            )
            fig_models.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside",
                cliponaxis=False
            )
            fig_models.update_layout(
                xaxis_range=[0, 1.05],
                showlegend=False
            )
            apply_plot_theme(fig_models, 254)
            st.plotly_chart(
                fig_models,
                width="stretch",
                config={"displayModeBar": False}
            )

    with insight_col:
        with st.container(
            border=True,
            key="panel_research_reading",
            height="stretch"
        ):
            render_panel_header("Lecture scientifique", "Hiérarchie observée")

            # Bandeau de décision : modèle retenu
            st.markdown(
                (
                    '<div class="verdict-banner verdict-banner--success">'
                    '<span class="verdict-dot"></span>'
                    '<div class="verdict-copy">'
                    '<strong>LightGBM Core retenu</strong>'
                    f'<small>PR-AUC Test {lightgbm_test:.4f} · modèle opérationnel</small>'
                    '</div>'
                    '</div>'
                ),
                unsafe_allow_html=True
            )

            r1, r2 = st.columns(2)
            with r1:
                render_metric("Meilleur graph-based", f"{control_test:.4f}")
            with r2:
                render_metric("Seuil app", f"{threshold:.4f}")

            # Chaîne de hiérarchie visuelle
            st.markdown(
                (
                    '<div class="hierarchy-chain">'
                    '<span>GraphSAGE</span><i>&lt;</i>'
                    '<span>Dynamic GNN</span><i>&lt;</i>'
                    '<span>NoMemory</span><i>&lt;</i>'
                    '<span class="hierarchy-chain__win">LightGBM Core</span>'
                    '</div>'
                ),
                unsafe_allow_html=True
            )

            st.markdown(
                (
                    '<div class="protocol-note">'
                    'Le seuil est sélectionné sur <strong>Validation</strong>. '
                    'Le split Test sert uniquement à l\'évaluation finale.'
                    '</div>'
                ),
                unsafe_allow_html=True
            )

with memory_tab:
    dynamic_col, reading_col = st.columns([1.52, 1], gap="small")

    with dynamic_col:
        with st.container(
            border=True,
            key="panel_research_dynamic",
            height="stretch"
        ):
            render_panel_header(
                "Dynamic GNN vs NoMemory",
                "Validation et Test"
            )
            comparison_df = pd.DataFrame(
                {
                    "Split": ["Validation", "Validation", "Test", "Test"],
                    "Modèle": [
                        "Dynamic GNN",
                        "NoMemory",
                        "Dynamic GNN",
                        "NoMemory"
                    ],
                    "PR-AUC": [
                        dynamic_validation,
                        control_validation,
                        dynamic_test,
                        control_test
                    ]
                }
            )
            fig_dynamic = px.bar(
                comparison_df,
                x="Split",
                y="PR-AUC",
                color="Modèle",
                barmode="group",
                text="PR-AUC",
                color_discrete_map={
                    "Dynamic GNN": "#38bdf8",
                    "NoMemory": "#3b82f6"
                }
            )
            fig_dynamic.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside",
                cliponaxis=False
            )
            fig_dynamic.update_layout(yaxis_range=[0, 1.0])
            apply_plot_theme(fig_dynamic, 254)
            fig_dynamic.update_xaxes(showgrid=False)
            fig_dynamic.update_yaxes(
                gridcolor=get_plot_theme()["grid"]
            )
            st.plotly_chart(
                fig_dynamic,
                width="stretch",
                config={"displayModeBar": False}
            )

    with reading_col:
        with st.container(
            border=True,
            key="panel_research_memory_reading",
            height="stretch"
        ):
            render_panel_header("Apport de la mémoire", "Delta Dynamic - NoMemory")
            d1, d2 = st.columns(2)
            with d1:
                render_metric("Delta Validation", f"{delta_validation:+.4f}")
            with d2:
                render_metric("Delta Test", f"{delta_test:+.4f}")

            render_badge("Pas de gain mesuré", "warning")
            st.markdown(
                "Le Dynamic GNN améliore GraphSAGE, mais reste inférieur au "
                "contrôle **NoMemory** construit avec le même prédicteur sans "
                "mémoire temporelle."
            )
            st.caption(
                "Le résultat ne justifie pas de remplacer LightGBM Core dans "
                "l'application opérationnelle."
            )

with structure_tab:
    graph_col, cold_col = st.columns([1.12, 1], gap="small")

    with graph_col:
        with st.container(
            border=True,
            key="panel_research_structure",
            height="stretch"
        ):
            render_panel_header("Structure du graphe PaySim", "Faible profondeur relationnelle")
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                render_metric("Nœuds", "9.07 M")
            with g2:
                render_metric("Arêtes", "6.36 M")
            with g3:
                render_metric("Degré = 1", "94.83 %")
            with g4:
                render_metric("Overlap", "0.0195 %")

            structure_df = pd.DataFrame(
                {
                    "Caractéristique": [
                        "Degré = 1",
                        "Degré ≤ 2",
                        "Source & destination"
                    ],
                    "Pourcentage": [94.83, 95.78, 0.0195]
                }
            )
            fig_structure = px.bar(
                structure_df,
                x="Caractéristique",
                y="Pourcentage",
                text="Pourcentage",
                color_discrete_sequence=["#38bdf8"]
            )
            fig_structure.update_traces(
                texttemplate="%{text:.4g} %",
                textposition="outside",
                cliponaxis=False
            )
            apply_plot_theme(fig_structure, 216)
            fig_structure.update_xaxes(showgrid=False)
            fig_structure.update_yaxes(
                gridcolor=get_plot_theme()["grid"]
            )
            st.plotly_chart(
                fig_structure,
                width="stretch",
                config={"displayModeBar": False}
            )

            st.caption(
                "Aucun couple dirigé répété, aucune réciprocité, aucun "
                "self-loop et aucun cycle observé."
            )

    with cold_col:
        with st.container(
            border=True,
            key="panel_research_cold_start",
            height="stretch"
        ):
            render_panel_header("Cold-start Test", "N = nouveau · K = connu")
            cs1, cs2 = st.columns(2)
            with cs1:
                render_metric("Nouvelles sources", "99.7 %")
            with cs2:
                render_metric("Nouvelles destinations", "43.4 %")

            cold_start_df = pd.DataFrame(
                {
                    "Catégorie": ["K→K", "N→K", "K→N", "N→N"],
                    "Transactions": [97, 451490, 76, 347164],
                    "Fraudes": [0, 1402, 0, 2428]
                }
            )
            st.dataframe(
                cold_start_df,
                hide_index=True,
                width="stretch",
                height=168,
                row_height=28,
                column_config={
                    "Transactions": st.column_config.NumberColumn(format="%d"),
                    "Fraudes": st.column_config.NumberColumn(format="%d")
                }
            )

            n1, n2, n3 = st.columns(3)
            with n1:
                render_metric("N→K Dynamic", "0.5167")
            with n2:
                render_metric("N→K NoMemory", "0.5980")
            with n3:
                render_metric("Delta", "-0.0813")

    with st.popover(
        "Métadonnées graphe chargées",
        icon=":material/database:",
        width="stretch"
    ):
        st.json(graph_metadata, expanded=False)

if network_tab.open:
    with network_tab:
        render_alert_network(manifest)

with conclusion_tab:
    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        with st.container(
            border=True,
            key="panel_research_conclusion_model",
            height="stretch"
        ):
            render_panel_header("01 · Référence prédictive")
            render_badge("LightGBM Core", "success")
            st.markdown(
                "Avec une PR-AUC Test proche de **0.980**, le modèle tabulaire "
                "surpasse nettement les modèles Graph Learning évalués."
            )

    with c2:
        with st.container(
            border=True,
            key="panel_research_conclusion_graph",
            height="stretch"
        ):
            render_panel_header("02 · Limite structurelle")
            render_badge("Graphe peu informatif", "info")
            st.markdown(
                "Le graphe est dominé par des comptes peu récurrents, avec une "
                "faible profondeur relationnelle et un cold-start massif."
            )

    with c3:
        with st.container(
            border=True,
            key="panel_research_conclusion_memory",
            height="stretch"
        ):
            render_panel_header("03 · Mémoire temporelle")
            render_badge("Gain non démontré", "warning")
            st.markdown(
                "Le Dynamic GNN améliore GraphSAGE, mais reste inférieur au "
                "contrôle NoMemory sur Validation et Test."
            )

    st.success(
        "Enseignement principal : la sophistication du modèle doit rester "
        "cohérente avec la structure informationnelle réellement présente "
        "dans les données."
    )

if not network_tab.open:
    with st.container(border=True, key="panel_research_landmarks"):
        render_panel_header(
            "Repères structurels PaySim",
            "Graphe complet et cold-start du split Test"
        )
        landmark_cols = st.columns(6, gap="small")
        landmarks = [
            (
                "Nœuds",
                f"{int(graph_metadata['dataset']['nodes']):,}"
            ),
            (
                "Arêtes",
                f"{int(graph_metadata['dataset']['transactions']):,}"
            ),
            ("Degré = 1", "94.83 %"),
            ("Source et destination", "0.0195 %"),
            ("Nouvelles sources Test", "99.7 %"),
            ("Nouvelles destinations Test", "43.4 %")
        ]
        for col, (label, value) in zip(landmark_cols, landmarks):
            with col:
                render_metric(label, value)
