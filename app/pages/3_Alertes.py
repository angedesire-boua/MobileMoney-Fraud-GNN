import streamlit as st

from utils.auth import require_authentication
from utils.loaders import (
    load_alerts,
    load_global_style,
    load_manifest
)
from utils.prediction import get_risk_level
from utils.ui import (
    format_compact_amount,
    render_badge,
    render_metric,
    render_page_header,
    render_panel_header
)


require_authentication()
load_global_style()

alerts = load_alerts()
manifest = load_manifest()
threshold = float(manifest["model"]["threshold"])


@st.cache_data
def prepare_alerts(df, decision_threshold):
    data = df.copy()
    data["classification"] = data["isFraud"].map(
        {1: "Vrai positif", 0: "Faux positif"}
    )
    data["risk_level"] = (
        data["fraud_probability"]
        .apply(lambda score: get_risk_level(float(score), decision_threshold))
    )
    return data


alerts_data = prepare_alerts(alerts, threshold)

total_alerts = len(alerts_data)
true_positives = int(alerts_data["isFraud"].sum())
false_positives = total_alerts - true_positives
precision = true_positives / total_alerts if total_alerts else 0.0
total_amount = float(alerts_data["amount"].sum())

render_page_header(
    "Alertes",
    "File de surveillance des transactions signalées sur le split Test PaySim.",
    "Console opérationnelle · LightGBM Core"
)

kpi_cols = st.columns(4, gap="small")
kpis = [
    (
        "total",
        "Alertes actives",
        f"{total_alerts:,}",
        f"{format_compact_amount(total_amount)} de volume"
    ),
    (
        "tp",
        "Vrais positifs",
        f"{true_positives:,}",
        "Fraudes détectées"
    ),
    (
        "fp",
        "Faux positifs",
        f"{false_positives:,}",
        "Alertes légitimes"
    ),
    (
        "precision",
        "Précision",
        f"{precision:.2%}",
        "Évaluation finale"
    )
]

for col, (key, label, value, note) in zip(kpi_cols, kpis):
    with col:
        with st.container(border=True, key=f"kpi_alerts_{key}"):
            render_metric(label, value)
            st.caption(note)

available_types = sorted(alerts_data["type"].astype(str).unique().tolist())
classification_options = ["Vrai positif", "Faux positif"]
risk_options = ["Modéré", "Élevé", "Critique"]
min_step = int(alerts_data["step"].min())
max_step = int(alerts_data["step"].max())

with st.container(border=True, key="command_alerts_filters"):
    render_panel_header("Filtres opérationnels", f"Seuil officiel {threshold:.6f}")
    f1, f2, f3, f4, f5, f6 = st.columns(
        [1.4, 1.4, 1.5, 0.85, 1.2, 0.8],
        gap="small",
        vertical_alignment="top"
    )
    with f1:
        selected_types = st.multiselect(
            "Type",
            options=available_types,
            default=available_types
        )
    with f2:
        selected_classifications = st.multiselect(
            "Évaluation",
            options=classification_options,
            default=classification_options
        )
    with f3:
        selected_risks = st.multiselect(
            "Risque",
            options=risk_options,
            default=risk_options
        )
    with f4:
        min_score = st.number_input(
            "Score min.",
            min_value=float(threshold),
            max_value=1.0,
            value=float(threshold),
            step=0.01,
            format="%.2f"
        )
    with f5:
        sort_option = st.selectbox(
            "Tri",
            options=[
                "Score décroissant",
                "Montant décroissant",
                "Step croissant"
            ]
        )
    with f6:
        with st.popover(
            "Période",
            icon=":material/schedule:",
            width="stretch"
        ):
            selected_steps = st.slider(
                "Fenêtre temporelle",
                min_value=min_step,
                max_value=max_step,
                value=(min_step, max_step)
            )

filtered_alerts = alerts_data.loc[
    alerts_data["type"].astype(str).isin(selected_types)
].copy()
filtered_alerts = filtered_alerts.loc[
    filtered_alerts["classification"].isin(selected_classifications)
]
filtered_alerts = filtered_alerts.loc[
    filtered_alerts["risk_level"].isin(selected_risks)
]
filtered_alerts = filtered_alerts.loc[
    filtered_alerts["step"].between(selected_steps[0], selected_steps[1])
]
filtered_alerts = filtered_alerts.loc[
    filtered_alerts["fraud_probability"] >= min_score
]

if sort_option == "Score décroissant":
    filtered_alerts = filtered_alerts.sort_values(
        "fraud_probability",
        ascending=False
    )
elif sort_option == "Montant décroissant":
    filtered_alerts = filtered_alerts.sort_values("amount", ascending=False)
else:
    filtered_alerts = filtered_alerts.sort_values("step", ascending=True)

filtered_alerts = filtered_alerts.reset_index(drop=True)
filtered_tp = int(filtered_alerts["isFraud"].sum())
filtered_amount = float(filtered_alerts["amount"].sum())
filtered_precision = (
    filtered_tp / len(filtered_alerts)
    if len(filtered_alerts)
    else 0.0
)

with st.container(key="panel_alert_summary"):
    summary_cols = st.columns(4, gap="small")
    for col, (label, value) in zip(
        summary_cols,
        [
            ("Alertes affichées", f"{len(filtered_alerts):,}"),
            ("Fraudes détectées", f"{filtered_tp:,}"),
            ("Précision filtre", f"{filtered_precision:.2%}"),
            ("Montant affiché", format_compact_amount(filtered_amount))
        ]
    ):
        with col:
            render_metric(label, value)

display_df = filtered_alerts[
    [
        "transaction_id",
        "step",
        "nameOrig",
        "nameDest",
        "type",
        "amount",
        "fraud_probability",
        "risk_level",
        "classification"
    ]
].copy()

if not display_df.empty:
    display_df["amount"] = display_df["amount"].round(2)
    display_df["fraud_probability"] = (
        display_df["fraud_probability"].round(6)
    )

display_df = display_df.rename(
    columns={
        "transaction_id": "Transaction",
        "step": "Step",
        "nameOrig": "Source",
        "nameDest": "Destination",
        "type": "Type",
        "amount": "Montant",
        "fraud_probability": "Score",
        "risk_level": "Risque",
        "classification": "Évaluation"
    }
)

table_col, alert_col = st.columns([2.55, 1], gap="small")
selected_alert = None

with table_col:
    with st.container(
        border=True,
        key="panel_alert_table",
        height="stretch"
    ):
        render_panel_header("File d'alertes", f"{len(filtered_alerts):,} résultats")

        if filtered_alerts.empty:
            st.warning("Aucune alerte ne correspond aux filtres sélectionnés.")
        else:
            selection = st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
                height=260,
                row_height=29,
                key="alerts_selectable_table",
                on_select="rerun",
                selection_mode="single-row",
                # Accounts remain available in column visibility, detail and CSV.
                column_order=[
                    "Transaction", "Step", "Type", "Montant",
                    "Score", "Risque", "Évaluation"
                ],
                column_config={
                    "Transaction": st.column_config.NumberColumn(format="%d", width=96),
                    "Step": st.column_config.NumberColumn(format="%d", width=54),
                    "Type": st.column_config.TextColumn(width=90),
                    "Montant": st.column_config.NumberColumn(format="%.2f", width=105),
                    "Risque": st.column_config.TextColumn(width=76),
                    "Évaluation": st.column_config.TextColumn(width=100),
                    "Score": st.column_config.ProgressColumn(
                        width=120,
                        min_value=0.0,
                        max_value=1.0,
                        format="%.4f"
                    )
                }
            )
            selected_rows = selection.selection.rows
            selected_index = (
                selected_rows[0]
                if selected_rows and selected_rows[0] < len(filtered_alerts)
                else 0
            )
            selected_alert = filtered_alerts.iloc[selected_index]

        st.caption(
            "Seuil sélectionné sur Validation ; Test réservé à l'évaluation finale."
        )

with alert_col:
    with st.container(
        border=True,
        key="panel_alert_selected",
        height="stretch"
    ):
        if selected_alert is None:
            render_panel_header("Alerte sélectionnée")
            st.info("Aucune transaction disponible.")
        else:
            selected_transaction_id = int(selected_alert["transaction_id"])
            render_panel_header(
                "Alerte sélectionnée",
                f"#{selected_transaction_id:,}"
            )

            score_col, risk_col = st.columns(2)
            with score_col:
                render_metric("Score", f"{selected_alert['fraud_probability']:.4f}")
            with risk_col:
                render_metric("Montant", f"{selected_alert['amount']:,.2f}")

            risk_tone = (
                "risk"
                if selected_alert["risk_level"] in {"Élevé", "Critique"}
                else "warning"
            )
            badge_col, truth_col = st.columns(2, vertical_alignment="center")
            with badge_col:
                render_badge(selected_alert["risk_level"], risk_tone)
            with truth_col:
                render_badge(
                    selected_alert["classification"],
                    "success"
                    if selected_alert["classification"] == "Vrai positif"
                    else "warning"
                )

            st.caption("TYPE")
            st.write(f"**{selected_alert['type']}**")

            a1, a2 = st.columns(2)
            with a1:
                st.caption("SOURCE")
                st.code(str(selected_alert["nameOrig"]), language=None)
            with a2:
                st.caption("DESTINATION")
                st.code(str(selected_alert["nameDest"]), language=None)

            if st.button(
                "Investiguer",
                icon=":material/manage_search:",
                type="primary",
                width="stretch"
            ):
                st.session_state["investigation_transaction_id"] = (
                    selected_transaction_id
                )
                st.switch_page("pages/4_Enquete.py")

            csv_data = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Exporter la vue",
                data=csv_data,
                file_name="paysim_alertes_filtrees.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch"
            )
