import pandas as pd


def get_transaction_prediction(
    predictions: pd.DataFrame,
    transaction_id: int
):
    # Rechercher la transaction
    match = predictions.loc[
        predictions["transaction_id"].eq(
            transaction_id
        )
    ]

    if match.empty:
        return None

    return match.iloc[0]


def get_alert_context(
    alerts: pd.DataFrame,
    transaction_id: int
):
    # Rechercher le contexte d'une alerte
    match = alerts.loc[
        alerts["transaction_id"].eq(
            transaction_id
        )
    ]

    if match.empty:
        return None

    return match.iloc[0]


def get_risk_level(
    score: float,
    threshold: float
):
    # Interpréter le score pour l'interface
    if score < threshold:
        return "Faible"

    if score < 0.70:
        return "Modéré"

    if score < 0.90:
        return "Élevé"

    return "Critique"