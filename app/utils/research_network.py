"""Compact research view of all saved alerts, with linked investigation."""

from html import escape

import streamlit as st

from utils.alert_network import (
    FILTER_LABELS, LAYOUT_VERSION, build_network_figure, filter_alerts, prepare_network,
    selected_transaction,
)
from utils.loaders import load_alerts
from utils.ui import get_plot_theme, render_badge, render_metric, render_panel_header


PLOT_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["toImage", "lasso2d", "select2d"],
}


@st.cache_data(show_spinner=False, max_entries=3)
def load_network(scope, layout_version):
    return prepare_network(filter_alerts(load_alerts(), scope))


def _theme():
    theme = get_plot_theme().copy()
    theme["cyan"] = "#0284c7" if st.session_state.get("ui_theme") == "light" else "#38bdf8"
    theme["network_glow"] = st.session_state.get("ui_theme", "dark") == "dark"
    return theme


def _select_from_graph(chart_key, allowed_ids):
    points = st.session_state.get(chart_key, {}).get("selection", {}).get("points", [])
    selected_id = selected_transaction(points, allowed_ids)
    if selected_id is not None:
        st.session_state["research_network_transaction"] = selected_id
        st.session_state["research_network_transaction_saved"] = selected_id


@st.dialog("Réseau des alertes", width="large")
def _expanded_network(network, selected_id, focus):
    st.plotly_chart(
        build_network_figure(network, _theme(), selected_id, focus, height=450),
        width="stretch", theme=None, config=PLOT_CONFIG,
        key="research_network_expanded",
    )


def render_alert_network(manifest):
    # Widget state is cleaned up when a tab is hidden or the theme reruns early.
    for key, default in (
        ("research_network_scope", FILTER_LABELS[0]),
        ("research_network_view", "Vue d'ensemble"),
        ("research_network_transaction", None),
    ):
        st.session_state.setdefault(key, st.session_state.get(f"{key}_saved", default))
    with st.container(key="research_network"):
        filter_col, count_col, accounts_col, groups_col = st.columns(
            [1.4, 1, 1, 1], gap="small", vertical_alignment="bottom"
        )
        with filter_col:
            scope = st.selectbox("Périmètre", FILTER_LABELS, key="research_network_scope")
            st.session_state["research_network_scope_saved"] = scope
        network = load_network(scope, LAYOUT_VERSION)
        rows = network["rows"]
        with count_col:
            render_metric("Transactions", f"{len(rows):,}")
        with accounts_col:
            render_metric(":material/account_tree: Comptes", f"{network['account_count']:,}")
        with groups_col:
            render_metric(":material/hub: Groupes distincts", f"{network['group_count']:,}")

        if rows.empty:
            st.info("Aucune alerte dans ce périmètre.")
            return

        ids = rows["transaction_id"].astype(int).tolist()
        selected_id = st.session_state.get("research_network_transaction")
        if selected_id not in ids:
            selected_id = int(rows.sort_values(["fraud_probability", "transaction_id"], ascending=[False, True]).iloc[0]["transaction_id"])
            st.session_state["research_network_transaction"] = selected_id

        graph_col, detail_col = st.columns([2.4, 1], gap="small")
        # Render the selector before the plot so both reflect this run's selection.
        with detail_col:
            with st.container(border=True, key="panel_research_network_detail", height="stretch"):
                render_panel_header("Transaction sélectionnée")
                selected_id = st.selectbox(
                    "Transaction", ids, key="research_network_transaction",
                    format_func=lambda value: f"Transaction {value:,}",
                    label_visibility="collapsed",
                )
                st.session_state["research_network_transaction_saved"] = selected_id
                chosen = rows.loc[rows["transaction_id"].eq(selected_id)].iloc[0]
                render_badge("Vrai positif" if chosen["isFraud"] else "Faux positif", "success" if chosen["isFraud"] else "warning")
                score_col, amount_col = st.columns(2, gap="small")
                with score_col:
                    render_metric("Score", f"{chosen['fraud_probability']:.4f}")
                with amount_col:
                    render_metric("Montant", f"{chosen['amount']:,.2f}")
                st.markdown(
                    '<dl class="network-accounts">'
                    f'<div><dt>Source</dt><dd>{escape(str(chosen["nameOrig"]))}</dd></div>'
                    f'<div><dt>Destination</dt><dd>{escape(str(chosen["nameDest"]))}</dd></div>'
                    '</dl>', unsafe_allow_html=True,
                )
                st.caption(f"{chosen['type']} · Step {int(chosen['step'])} · Groupe de {len(network['components'][int(chosen['component'])])} comptes")
                if st.button("Ouvrir l'enquête", icon=":material/manage_search:", type="primary", width="stretch", key="research_network_investigate"):
                    st.session_state["investigation_transaction_id"] = int(selected_id)
                    st.switch_page("pages/4_Enquete.py")

        with graph_col:
            with st.container(border=True, key="panel_research_network_graph", height="stretch"):
                render_panel_header("Réseau des transactions signalées", f"Plus grand groupe : {network['largest_group']} comptes")
                view_col, expand_col = st.columns([8, 1], gap="small", vertical_alignment="center")
                with view_col:
                    view = st.segmented_control(
                        "Vue du réseau", ["Vue d'ensemble", "Groupe sélectionné"],
                        required=True,
                        key="research_network_view", label_visibility="collapsed",
                    )
                    st.session_state["research_network_view_saved"] = view
                focus = view == "Groupe sélectionné"
                with expand_col:
                    if st.button("", icon=":material/open_in_full:", help="Agrandir le graphe", key="research_network_expand", width="stretch"):
                        _expanded_network(network, selected_id, focus)

                chart_key = f"research_network_plot_{FILTER_LABELS.index(scope)}"
                st.plotly_chart(
                    build_network_figure(network, _theme(), selected_id, focus),
                    width="stretch", theme=None, config=PLOT_CONFIG, key=chart_key,
                    selection_mode="points",
                    on_select=lambda: _select_from_graph(chart_key, set(ids)),
                )
        st.caption(
            f"Sous-graphe des alertes Test uniquement : {int(manifest['test_metrics']['fn']):,} fraudes non détectées hors périmètre. "
            "Groupes séparés : aucun lien ajouté entre comptes indépendants."
        )
