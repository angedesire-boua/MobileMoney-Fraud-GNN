import plotly.graph_objects as go
import streamlit as st

from utils.auth import require_authentication
from utils.investigation import (
    CORE_FEATURES,
    compute_local_contributions,
    load_transaction_rows
)
from utils.loaders import (
    load_alerts,
    load_core_model,
    load_global_style,
    load_manifest,
    load_predictions
)
from utils.prediction import (
    get_alert_context,
    get_risk_level,
    get_transaction_prediction
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
predictions = load_predictions()
alerts = load_alerts()
model = load_core_model()

model_info = manifest["model"]
transaction_info = manifest["transaction_id"]
threshold = float(model_info["threshold"])
first_test_id = int(transaction_info["first_test_id"])
last_test_id = int(transaction_info["last_test_id"])


def format_decision(prediction):
    return "Fraude détectée" if prediction == 1 else "Non signalée"


def format_truth(label):
    return "Fraude" if label == 1 else "Non-fraude"


render_page_header(
    "Détection",
    "Analyse ciblée d'une transaction réelle du split Test PaySim.",
    "Scoring opérationnel · LightGBM Core"
)

with st.container(border=True, key="command_detection"):
    render_panel_header(
        "Transaction à analyser",
        f"IDs disponibles : {first_test_id:,} à {last_test_id:,}"
    )
    input_col, action_col, model_col, threshold_col, split_col = st.columns(
        [1.5, 1.05, 0.9, 0.82, 0.65],
        gap="small",
        vertical_alignment="bottom"
    )

    with input_col:
        transaction_id = st.number_input(
            "Transaction ID",
            min_value=first_test_id,
            max_value=last_test_id,
            value=int(
                st.session_state.get(
                    "detection_transaction_id",
                    first_test_id
                )
            ),
            step=1
        )
    with action_col:
        analyze = st.button(
            "Analyser",
            type="primary",
            icon=":material/radar:",
            width="stretch"
        )
    with model_col:
        render_metric("Modèle", "LightGBM")
    with threshold_col:
        render_metric("Seuil Validation", f"{threshold:.4f}")
    with split_col:
        render_metric("Split", "Test")

if analyze:
    st.session_state["detection_transaction_id"] = int(transaction_id)

selected_id = int(
    st.session_state.get("detection_transaction_id", transaction_id)
)
transaction = get_transaction_prediction(predictions, selected_id)

if transaction is None:
    st.error("La transaction demandée n'existe pas dans le split Test.")
    st.stop()

score = float(transaction["fraud_probability"])
prediction = int(transaction["prediction"])
ground_truth = int(transaction["isFraud"])
amount = float(transaction["amount"])
step = int(transaction["step"])
transaction_type = str(transaction["type"])
risk_level = get_risk_level(score, threshold)
margin = score - threshold
alert_context = get_alert_context(alerts, selected_id)
ml_row, _ = load_transaction_rows(selected_id)
local_explanation = compute_local_contributions(
    model=model,
    ml_row=ml_row,
    feature_order=CORE_FEATURES,
    categories=model.pandas_categorical[0]
)

if prediction == 1 and ground_truth == 1:
    classification = "Vrai positif (TP)"
elif prediction == 1 and ground_truth == 0:
    classification = "Faux positif (FP)"
elif prediction == 0 and ground_truth == 1:
    classification = "Faux négatif (FN)"
else:
    classification = "Vrai négatif (TN)"

result_key = (
    "panel_detection_result_risk"
    if prediction == 1
    else "panel_detection_result_safe"
)
result_col, details_col = st.columns([1.48, 1], gap="small")

with result_col:
    with st.container(border=True, key=result_key, height="stretch"):
        render_panel_header("Résultat de la détection", f"Transaction #{selected_id:,}")

        status_col, message_col = st.columns(
            [0.62, 2.2],
            gap="small",
            vertical_alignment="center"
        )
        with status_col:
            render_badge(
                format_decision(prediction),
                "risk" if prediction == 1 else "success"
            )
        with message_col:
            st.caption(
                "Le score dépasse le seuil validé."
                if prediction == 1
                else "Le score reste sous le seuil validé."
            )

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_metric("Score", f"{score:.4f}")
        with k2:
            render_metric("Risque", risk_level)
        with k3:
            render_metric("Montant", f"{amount:,.2f}")
        with k4:
            render_metric("Step", f"{step}")

        # Ruban de score : position du score face au seuil Validation
        score_pct = min(max(score, 0.0), 1.0) * 100
        threshold_pct = threshold * 100
        ribbon_tone = "risk" if prediction == 1 else "safe"
        st.markdown(
            (
                f'<div class="score-ribbon score-ribbon--{ribbon_tone}">'
                '<div class="score-ribbon__track">'
                f'<div class="score-ribbon__fill" style="width:{score_pct:.4f}%"></div>'
                f'<div class="score-ribbon__threshold" style="left:{threshold_pct:.4f}%"></div>'
                '</div>'
                '<div class="score-ribbon__legend">'
                '<span>0.0</span>'
                f'<span class="score-ribbon__mark">Seuil {threshold:.4f}</span>'
                '<span>1.0</span>'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True
        )

        d1, d2, d3 = st.columns(3)
        with d1:
            st.caption("TYPE")
            st.write(f"**{transaction_type}**")
        with d2:
            st.caption("SEUIL")
            st.write(f"**{threshold:.6f}**")
        with d3:
            st.caption("ÉCART AU SEUIL")
            st.write(f"**{margin:+.8f}**")

with details_col:
    with st.container(
        border=True,
        key="panel_detection_context",
        height="stretch"
    ):
        render_panel_header("Évaluation et comptes", classification)

        e1, e2 = st.columns(2)
        with e1:
            render_metric("Décision modèle", format_decision(prediction))
        with e2:
            render_metric("Vrai label PaySim", format_truth(ground_truth))

        badge_tone = "success" if classification.startswith("Vrai") else "warning"
        render_badge(classification, badge_tone)

        if alert_context is not None:
            a1, a2 = st.columns(2)
            with a1:
                st.caption("SOURCE")
                st.code(str(alert_context["nameOrig"]), language=None)
            with a2:
                st.caption("DESTINATION")
                st.code(str(alert_context["nameDest"]), language=None)

            if st.button(
                "Ouvrir dans Enquête",
                type="primary",
                icon=":material/manage_search:",
                width="stretch"
            ):
                st.session_state["investigation_transaction_id"] = selected_id
                st.switch_page("pages/4_Enquete.py")
        else:
            st.caption(
                "Les comptes détaillés sont disponibles pour les transactions "
                "présentes dans la file d'alertes."
            )

score_col, protocol_col = st.columns([1.48, 1], gap="small")

with score_col:
    with st.container(
        border=True,
        key="panel_detection_score_reading",
        height="stretch"
    ):
        render_panel_header(
            "Explication LightGBM locale",
            "Contributions principales au score brut"
        )
        plot_theme = get_plot_theme()
        top_contributions = local_explanation["table"].head(7).copy()
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
                text=[f"{value:+.3f}" for value in top_contributions["contribution"]],
                textposition="outside",
                cliponaxis=False,
                customdata=top_contributions["value"],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Valeur : %{customdata}<br>"
                    "Contribution : %{x:.5f}<extra></extra>"
                )
            )
        )
        fig_explanation.update_layout(
            height=178,
            margin=dict(l=8, r=34, t=6, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            font=dict(color=plot_theme["font"], size=10),
            yaxis=dict(autorange="reversed", showgrid=False),
            xaxis=dict(
                gridcolor=plot_theme["grid"],
                zeroline=False
            )
        )
        st.plotly_chart(
            fig_explanation,
            width="stretch",
            config={"displayModeBar": False}
        )

with protocol_col:
    with st.container(
        border=True,
        key="panel_detection_protocol",
        height="stretch"
    ):
        render_panel_header("Cadre d'évaluation", "Protocole scientifique")
        p1, p2 = st.columns(2)
        with p1:
            render_metric("Seuil choisi sur", "Validation")
            render_metric("Décision", format_decision(prediction))
        with p2:
            render_metric("Mesure finale sur", "Test")
            render_metric("Classification", classification)
        render_badge("Aucune sélection sur Test", "info")

st.caption(
    "Le seuil officiel est sélectionné exclusivement sur Validation. "
    "Le split Test est utilisé pour l'évaluation finale."
)
