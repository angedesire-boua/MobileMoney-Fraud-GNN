import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.auth import require_authentication
from utils.investigation import (
    CORE_FEATURES,
    compute_local_contributions,
    get_local_history,
    load_transaction_rows
)
from utils.loaders import (
    load_alerts,
    load_core_model,
    load_global_style,
    load_known_status,
    load_manifest
)
from utils.prediction import get_risk_level
from utils.ui import (
    get_plot_theme,
    render_badge,
    render_metric,
    render_page_header,
    render_panel_header
)


require_authentication()
load_global_style()

alerts = load_alerts()
manifest = load_manifest()
model = load_core_model()
known_status = load_known_status()
threshold = float(manifest["model"]["threshold"])

feature_order = CORE_FEATURES


def format_time_since(value):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):,.0f}"


def build_graph_figure(history, source, destination):
    theme = get_plot_theme()
    graph = nx.DiGraph()

    if not history.empty:
        for _, row in history.iterrows():
            graph.add_edge(
                str(row["nameOrig"]),
                str(row["nameDest"]),
                step=int(row["step"]),
                amount=float(row["amount"]),
                transaction_type=str(row["type"]),
                current=False
            )

    graph.add_edge(source, destination, current=True)
    # A horizontal focus edge and two history rows avoid the compressed star
    # layout. Only geometry changes; every account and causal edge is retained.
    positions = {source: (-0.8, 0.0), destination: (0.8, 0.0)}
    historical_nodes = sorted(set(graph.nodes()) - {source, destination})
    for offset, nodes in ((0.55, historical_nodes[::2]), (-0.55, historical_nodes[1::2])):
        for index, node in enumerate(nodes):
            x = -0.85 + 1.7 * (index + 1) / (len(nodes) + 1)
            positions[node] = (x, offset)
    figure = go.Figure()

    for u, v, data in graph.edges(data=True):
        x0, y0 = positions[u]
        x1, y1 = positions[v]
        is_current = data.get("current", False)
        edge_color = "#ef4444" if is_current else theme["edge"]
        edge_width = 3.6 if is_current else 1.25
        hover_text = (
            "Transaction étudiée"
            if is_current
            else (
                f"Step : {data.get('step')}<br>"
                f"Type : {data.get('transaction_type')}<br>"
                f"Montant : {data.get('amount', 0):,.2f}"
            )
        )

        figure.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=edge_width, color=edge_color),
                hoverinfo="text",
                text=hover_text,
                showlegend=False
            )
        )
        figure.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.05,
            arrowwidth=edge_width,
            arrowcolor=edge_color,
            opacity=0.88
        )

    for node in graph.nodes():
        x, y = positions[node]
        if node == source:
            node_color = "#38bdf8"
            node_size = 24
            label = "Source"
        elif node == destination:
            node_color = "#ef4444"
            node_size = 26
            label = "Destination"
        else:
            node_color = "#64748b"
            node_size = 16
            label = "Historique"

        figure.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode="markers+text" if node in {source, destination} else "markers",
                marker=dict(
                    size=node_size,
                    color=node_color,
                    line=dict(width=1, color=theme["node_outline"])
                ),
                text=[label],
                textfont=dict(size=11, color=theme["strong"]),
                textposition="top center",
                cliponaxis=False,
                customdata=[[node]],
                hovertemplate=(
                    "<b>%{text}</b><br>Compte : %{customdata[0]}<extra></extra>"
                ),
                showlegend=False
            )
        )

    figure.update_layout(
        height=140 if history.empty else 180,
        margin=dict(l=18, r=18, t=22, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme["font"], size=10),
        xaxis=dict(visible=False, range=[-1.1, 1.1]),
        yaxis=dict(visible=False, range=[-0.85, 0.95]),
        hovermode="closest",
        dragmode="pan"
    )
    return figure


render_page_header(
    "Enquête",
    "Contexte causal, explication locale et graphe de proximité d'une alerte.",
    "Investigation · historique strictement antérieur"
)

alert_ids = alerts["transaction_id"].astype(int).tolist()
default_id = int(
    st.session_state.get("investigation_transaction_id", alert_ids[0])
)
if default_id not in alert_ids:
    default_id = alert_ids[0]

with st.container(border=True, key="command_investigation"):
    render_panel_header(
        "Transaction signalée",
        "Le Test n'intervient pas dans la sélection du seuil"
    )
    selector_col, model_col, threshold_col, protocol_col = st.columns(
        [1.65, 0.82, 0.9, 0.82],
        gap="small",
        vertical_alignment="bottom"
    )
    with selector_col:
        transaction_id = st.selectbox(
            "Alerte",
            options=alert_ids,
            index=alert_ids.index(default_id),
            format_func=lambda x: f"Transaction {x:,}"
        )
    with model_col:
        render_metric("Modèle", "LightGBM")
    with threshold_col:
        render_metric("Seuil Validation", f"{threshold:.6f}")
    with protocol_col:
        render_metric("Contexte", "Causal")

st.session_state["investigation_transaction_id"] = int(transaction_id)

ml_row, graph_row = load_transaction_rows(int(transaction_id))
alert = alerts.loc[alerts["transaction_id"].eq(transaction_id)].iloc[0]

score = float(alert["fraud_probability"])
source = str(graph_row["nameOrig"])
destination = str(graph_row["nameDest"])
step = int(graph_row["step"])
risk_level = get_risk_level(score, threshold)

test_first_id = int(manifest["transaction_id"]["first_test_id"])
test_position = int(transaction_id) - test_first_id
src_known = bool(known_status["src_known"][test_position])
dst_known = bool(known_status["dst_known"][test_position])
src_status = "Connu" if src_known else "Nouveau"
dst_status = "Connu" if dst_known else "Nouveau"
cold_start_status = f"{'K' if src_known else 'N'}→{'K' if dst_known else 'N'}"

with st.spinner("Construction du contexte causal..."):
    history = get_local_history(
        source=source,
        destination=destination,
        step=step,
        max_edges=12
    )

summary_col, graph_col = st.columns([0.94, 1.5], gap="small")

with summary_col:
    with st.container(
        border=True,
        key="panel_investigation_summary",
        height="stretch"
    ):
        render_panel_header("Résumé transaction", f"#{int(transaction_id):,}")

        badge_col, context_col = st.columns(2, vertical_alignment="center")
        with badge_col:
            render_badge(
                risk_level,
                "risk" if risk_level in {"Élevé", "Critique"} else "warning"
            )
        with context_col:
            render_badge(f"Cold-start {cold_start_status}", "info")

        k1, k2, k3 = st.columns(3)
        with k1:
            render_metric("Score", f"{score:.4f}")
        with k2:
            render_metric("Montant", f"{float(graph_row['amount']):,.2f}")
        with k3:
            render_metric("Step", f"{step}")

        c1, c2 = st.columns(2, gap="small")
        with c1:
            st.caption("SOURCE")
            st.code(source, language=None)
            render_metric("Source", src_status)
        with c2:
            st.caption("DESTINATION")
            st.code(destination, language=None)
            render_metric("Destination", dst_status)

with graph_col:
    with st.container(
        border=True,
        key="panel_investigation_graph",
        height="stretch"
    ):
        render_panel_header(
            "Graphe local d'investigation",
            f"{len(history)} transaction(s) antérieure(s)"
        )
        if history.empty:
            st.caption("Aucune transaction antérieure pertinente retrouvée.")
        graph_figure = build_graph_figure(
            history=history,
            source=source,
            destination=destination
        )
        st.plotly_chart(
            graph_figure,
            width="stretch",
            config={"displayModeBar": False}
        )

history_col, explain_col = st.columns([1, 1.48], gap="small")

with history_col:
    with st.container(
        border=True,
        key="panel_investigation_history",
        height="stretch"
    ):
        render_panel_header("Historique causal", "Variables antérieures au step courant")
        src_col, dst_col = st.columns(2, gap="small")
        with src_col:
            st.caption("SOURCE")
            st.markdown(
                f"""
                <div class="history-compact-grid">
                    <div><span>Transactions</span><strong>{int(graph_row['orig_prev_transactions']):,}</strong></div>
                    <div><span>Volume</span><strong>{float(graph_row['orig_prev_volume']):,.2f}</strong></div>
                    <div><span>Dernière activité</span><strong>{format_time_since(graph_row['orig_time_since_last'])}</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with dst_col:
            st.caption("DESTINATION")
            st.markdown(
                f"""
                <div class="history-compact-grid">
                    <div><span>Transactions</span><strong>{int(graph_row['dest_prev_transactions']):,}</strong></div>
                    <div><span>Volume</span><strong>{float(graph_row['dest_prev_volume']):,.2f}</strong></div>
                    <div><span>Dernière activité</span><strong>{format_time_since(graph_row['dest_time_since_last'])}</strong></div>
                </div>
                """,
                unsafe_allow_html=True
            )

with explain_col:
    with st.container(
        border=True,
        key="panel_investigation_explanation",
        height="stretch"
    ):
        render_panel_header(
            "Explication LightGBM locale",
            "Contributions au score brut"
        )
        explanation = compute_local_contributions(
            model=model,
            ml_row=ml_row,
            feature_order=feature_order,
            categories=model.pandas_categorical[0]
        )
        top_contributions = explanation["table"].head(9).copy()

        plot_theme = get_plot_theme()
        fig_explanation = go.Figure()
        fig_explanation.add_trace(
            go.Bar(
                x=top_contributions["contribution"],
                y=top_contributions["feature"],
                orientation="h",
                marker_color=[
                    "#ef4444" if value > 0 else "#22c55e"
                    for value in top_contributions["contribution"]
                ],
                customdata=top_contributions["value"],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Valeur : %{customdata}<br>"
                    "Contribution : %{x:.4f}<extra></extra>"
                )
            )
        )
        fig_explanation.update_layout(
            height=180,
            yaxis=dict(
                autorange="reversed",
                tickmode="array",
                tickvals=top_contributions["feature"].tolist(),
                tickfont=dict(size=10),
                automargin=True
            ),
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=plot_theme["font"], size=10),
            xaxis=dict(
                gridcolor=plot_theme["grid"]
            )
        )
        st.plotly_chart(
            fig_explanation,
            width="stretch",
            config={"displayModeBar": False}
        )

with st.popover(
    "Détails techniques et transactions historiques",
    icon=":material/description:",
    width="stretch"
):
    v1, v2, v3, v4 = st.columns(4)
    with v1:
        render_metric("Probabilité recalculée", f"{explanation['probability']:.8f}")
    with v2:
        render_metric("Probabilité sauvegardée", f"{score:.8f}")
    with v3:
        render_metric("Score brut", f"{explanation['raw_score']:.8f}")
    with v4:
        render_metric(
            "Somme contributions",
            f"{explanation['contribution_sum']:.8f}"
        )

    if not history.empty:
        history_display = history.copy()
        history_display["amount"] = history_display["amount"].round(2)
        history_display = history_display.rename(
            columns={
                "nameOrig": "Source",
                "nameDest": "Destination",
                "step": "Step",
                "amount": "Montant",
                "type": "Type",
                "isFraud": "Fraude réelle"
            }
        )
        st.dataframe(
            history_display,
            hide_index=True,
            width="stretch",
            height=220
        )

st.caption(
    "Sous-graphe local uniquement : aucun graphe global de 9 millions de "
    "nœuds n'est chargé."
)
