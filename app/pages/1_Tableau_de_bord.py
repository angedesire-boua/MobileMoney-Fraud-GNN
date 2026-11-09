import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots

from utils.auth import require_authentication
from utils.loaders import (
    load_alerts,
    load_global_style,
    load_manifest,
    load_predictions
)
from utils.prediction import get_risk_level
from utils.ui import (
    format_compact_amount,
    get_plot_theme,
    render_metric,
    render_page_header,
    render_panel_header
)


require_authentication()
load_global_style()

manifest = load_manifest()
predictions = load_predictions()
alerts = load_alerts()

metrics = manifest["test_metrics"]
threshold = float(manifest["model"]["threshold"])


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
        showgrid=False,
        zeroline=False,
        tickfont=dict(color=theme["font"]),
        title_font=dict(color=theme["font"])
    )
    fig.update_yaxes(
        gridcolor=theme["grid"],
        zeroline=False,
        tickfont=dict(color=theme["font"]),
        title_font=dict(color=theme["font"])
    )
    return fig


@st.cache_data
def prepare_dashboard_data(predictions_df, alerts_df, decision_threshold):
    temporal = (
        predictions_df
        .groupby("step", observed=True)
        .agg(
            transactions=("transaction_id", "size"),
            alerts=("prediction", "sum"),
            frauds=("isFraud", "sum")
        )
        .reset_index()
    )

    by_type = (
        predictions_df
        .groupby("type", observed=True)
        .agg(
            transactions=("transaction_id", "size"),
            alerts=("prediction", "sum"),
            frauds=("isFraud", "sum")
        )
        .reset_index()
    )

    prepared_alerts = alerts_df.copy()
    prepared_alerts["risk_level"] = (
        prepared_alerts["fraud_probability"]
        .apply(lambda score: get_risk_level(float(score), decision_threshold))
    )

    top_alerts = (
        prepared_alerts
        .sort_values(
            ["fraud_probability", "amount"],
            ascending=[False, False]
        )
        .head(6)
        .copy()
    )

    return temporal, by_type, prepared_alerts, top_alerts


temporal, by_type, alerts_prepared, top_alerts = prepare_dashboard_data(
    predictions,
    alerts,
    threshold
)

total_transactions = len(predictions)
total_alerts = len(alerts)
total_frauds = int(predictions["isFraud"].sum())
true_positives = int(alerts["isFraud"].sum())
false_positives = total_alerts - true_positives
alert_rate = total_alerts / total_transactions * 100
alerted_amount = float(alerts["amount"].sum())

render_page_header(
    "Tableau de bord",
    "Vue opérationnelle du split Test PaySim et qualité des alertes produites.",
    f"Monitoring · seuil Validation {threshold:.6f}"
)

kpi_cols = st.columns(6, gap="small")
kpis = [
    (
        "transactions",
        "Transactions",
        f"{total_transactions:,}",
        "Split Test complet"
    ),
    (
        "alerts",
        "Alertes",
        f"{total_alerts:,}",
        f"{alert_rate:.3f} % du Test"
    ),
    (
        "frauds",
        "Fraudes réelles",
        f"{total_frauds:,}",
        f"Rappel {metrics['recall']:.2%}"
    ),
    (
        "precision",
        "Précision",
        f"{metrics['precision']:.2%}",
        f"{false_positives:,} faux positifs"
    ),
    (
        "volume",
        "Volume alerté",
        format_compact_amount(alerted_amount),
        "Montants PaySim"
    ),
    (
        "pr_auc",
        "PR-AUC Test",
        f"{metrics['pr_auc']:.4f}",
        "Évaluation finale"
    )
]

for col, (key, label, value, note) in zip(kpi_cols, kpis):
    with col:
        with st.container(border=True, key=f"kpi_dashboard_{key}"):
            render_metric(label, value)
            st.caption(note)

activity_col, side_col = st.columns([1.68, 1], gap="small")

with activity_col:
    with st.container(border=True, key="panel_dashboard_activity"):
        render_panel_header("Activité transactionnelle", "Transactions et alertes par step")
        fig_activity = make_subplots(specs=[[{"secondary_y": True}]])
        fig_activity.add_trace(
            go.Scatter(
                x=temporal["step"],
                y=temporal["transactions"],
                name="Transactions",
                mode="lines",
                line=dict(color="#38bdf8", width=2),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.07)",
                hovertemplate="Step %{x}<br>Transactions %{y:,}<extra></extra>"
            ),
            secondary_y=False
        )
        fig_activity.add_trace(
            go.Scatter(
                x=temporal["step"],
                y=temporal["alerts"],
                name="Alertes",
                mode="lines",
                line=dict(color="#ef4444", width=1.9),
                hovertemplate="Step %{x}<br>Alertes %{y:,}<extra></extra>"
            ),
            secondary_y=True
        )
        apply_plot_theme(fig_activity, 225)
        fig_activity.update_layout(hovermode="x unified")
        fig_activity.update_yaxes(title_text="Transactions", secondary_y=False)
        fig_activity.update_yaxes(
            title_text="Alertes",
            showgrid=False,
            secondary_y=True
        )
        st.plotly_chart(
            fig_activity,
            width="stretch",
            config={"displayModeBar": False}
        )

with side_col:
    with st.container(border=True, key="panel_dashboard_types"):
        render_panel_header("Alertes par type", "Comparaison avec la fraude réelle")
        fig_types = go.Figure()
        fig_types.add_trace(
            go.Bar(
                x=by_type["type"],
                y=by_type["alerts"],
                name="Alertes",
                marker_color="#ef4444"
            )
        )
        fig_types.add_trace(
            go.Bar(
                x=by_type["type"],
                y=by_type["frauds"],
                name="Fraudes",
                marker_color="#22c55e"
            )
        )
        fig_types.update_layout(barmode="group")
        apply_plot_theme(fig_types, 225)
        st.plotly_chart(
            fig_types,
            width="stretch",
            config={"displayModeBar": False}
        )

table_col, quality_col = st.columns([1.92, 0.82], gap="small")

with table_col:
    with st.container(
        border=True,
        key="panel_dashboard_alerts",
        height="stretch"
    ):
        title_col, link_col = st.columns([4, 1.05], vertical_alignment="center")
        with title_col:
            render_panel_header("Alertes prioritaires", "Score puis montant")
        with link_col:
            st.page_link(
                "pages/3_Alertes.py",
                label="Voir la file",
                icon=":material/notifications_active:"
            )

        top_display = top_alerts[
            [
                "transaction_id",
                "step",
                "type",
                "amount",
                "fraud_probability",
                "risk_level",
                "isFraud"
            ]
        ].copy()
        top_display["amount"] = top_display["amount"].round(2)
        top_display["fraud_probability"] = (
            top_display["fraud_probability"].round(6)
        )
        top_display["isFraud"] = top_display["isFraud"].map(
            {1: "Oui", 0: "Non"}
        )
        top_display = top_display.rename(
            columns={
                "transaction_id": "Transaction",
                "step": "Step",
                "type": "Type",
                "amount": "Montant",
                "fraud_probability": "Score",
                "risk_level": "Risque",
                "isFraud": "Fraude"
            }
        )

        st.dataframe(
            top_display,
            hide_index=True,
            width="stretch",
            height=210,
            row_height=25,
            column_config={
                "Transaction": st.column_config.NumberColumn(format="%d"),
                "Montant": st.column_config.NumberColumn(format="%.2f"),
                "Score": st.column_config.ProgressColumn(
                    min_value=0.0,
                    max_value=1.0,
                    format="%.4f"
                )
            }
        )

with quality_col:
    with st.container(
        border=True,
        key="panel_dashboard_quality",
        height="stretch"
    ):
        render_panel_header("Qualité des alertes", f"F1 {metrics['f1']:.2%}")
        fig_quality = go.Figure(
            data=[
                go.Pie(
                    labels=["Vrais positifs", "Faux positifs"],
                    values=[true_positives, false_positives],
                    hole=0.72,
                    marker=dict(colors=["#22c55e", "#ef4444"]),
                    textinfo="none",
                    hovertemplate=(
                        "%{label}<br>%{value:,}<br>%{percent}<extra></extra>"
                    )
                )
            ]
        )
        fig_quality.update_layout(
            height=150,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            annotations=[
                dict(
                    text=(
                        f"<b>{metrics['precision']:.1%}</b>"
                        "<br><span style='font-size:9px'>Précision</span>"
                    ),
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(
                        size=15,
                        color=get_plot_theme()["strong"]
                    )
                )
            ]
        )
        st.plotly_chart(
            fig_quality,
            width="stretch",
            config={"displayModeBar": False}
        )

        stat1, stat2 = st.columns(2)
        with stat1:
            render_metric("Vrais positifs", f"{true_positives:,}")
        with stat2:
            render_metric("Faux positifs", f"{false_positives:,}")
