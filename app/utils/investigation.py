import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from utils.loaders import (
    GRAPH_DATA_PATH,
    ML_DATA_PATH
)


# Features du modèle LightGBM Core
CORE_FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "log_amount",
    "amount_to_dest_balance_ratio",
    "origin_balance_change",
    "destination_balance_change",
    "origin_zero_before",
    "origin_zero_after",
    "destination_zero_before",
    "destination_zero_after",
    "type",
    "orig_prev_transactions",
    "orig_prev_volume",
    "orig_prev_mean_amount",
    "orig_time_since_last",
    "dest_prev_transactions",
    "dest_prev_volume",
    "dest_prev_mean_amount",
    "dest_time_since_last"
]


GRAPH_CONTEXT_COLUMNS = [
    "nameOrig",
    "nameDest",
    "step",
    "split",
    "amount",
    "type",
    "orig_prev_transactions",
    "dest_prev_transactions",
    "orig_prev_volume",
    "dest_prev_volume",
    "orig_time_since_last",
    "dest_time_since_last",
    "isFraud"
]


def locate_row(parquet_path, transaction_id):
    # Localiser le row group
    parquet = pq.ParquetFile(parquet_path)

    offset = 0

    for row_group in range(
        parquet.metadata.num_row_groups
    ):
        size = (
            parquet.metadata
            .row_group(row_group)
            .num_rows
        )

        if transaction_id < offset + size:
            local_index = transaction_id - offset

            return (
                parquet,
                row_group,
                int(local_index)
            )

        offset += size

    raise IndexError(
        "Transaction ID hors du dataset."
    )


def load_transaction_rows(transaction_id):
    # Lire uniquement le row group nécessaire
    ml_parquet, ml_rg, ml_idx = locate_row(
        ML_DATA_PATH,
        transaction_id
    )

    graph_parquet, graph_rg, graph_idx = locate_row(
        GRAPH_DATA_PATH,
        transaction_id
    )

    ml_table = ml_parquet.read_row_group(
        ml_rg,
        columns=CORE_FEATURES + [
            "step",
            "isFraud"
        ]
    )

    graph_table = graph_parquet.read_row_group(
        graph_rg,
        columns=GRAPH_CONTEXT_COLUMNS
    )

    ml_row = (
        ml_table
        .slice(ml_idx, 1)
        .to_pandas()
        .iloc[0]
    )

    graph_row = (
        graph_table
        .slice(graph_idx, 1)
        .to_pandas()
        .iloc[0]
    )

    return ml_row, graph_row


def get_local_history(
    source,
    destination,
    step,
    max_edges=12
):
    # Construire le filtre causal
    dataset = ds.dataset(
        GRAPH_DATA_PATH,
        format="parquet"
    )

    source_filter = (
        (ds.field("nameOrig") == source)
        & (ds.field("step") < step)
    )

    destination_filter = (
        (ds.field("nameDest") == destination)
        & (ds.field("step") < step)
    )

    columns = [
        "nameOrig",
        "nameDest",
        "step",
        "amount",
        "type",
        "isFraud"
    ]

    # Historique de la source
    source_table = dataset.to_table(
        columns=columns,
        filter=source_filter
    )

    source_history = (
        source_table
        .to_pandas()
        .sort_values(
            "step",
            ascending=False
        )
        .head(max_edges)
    )

    # Historique de la destination
    destination_table = dataset.to_table(
        columns=columns,
        filter=destination_filter
    )

    destination_history = (
        destination_table
        .to_pandas()
        .sort_values(
            "step",
            ascending=False
        )
        .head(max_edges)
    )

    history = pd.concat(
        [
            source_history,
            destination_history
        ],
        ignore_index=True
    )

    if history.empty:
        return history

    # Éviter les doublons éventuels
    history = (
        history
        .drop_duplicates(
            subset=[
                "nameOrig",
                "nameDest",
                "step",
                "amount"
            ]
        )
        .sort_values(
            "step",
            ascending=False
        )
        .head(max_edges)
        .reset_index(drop=True)
    )

    return history


def compute_local_contributions(
    model,
    ml_row,
    feature_order,
    categories
):
    # Préparer une seule transaction
    X = pd.DataFrame(
        [
            {
                feature: ml_row[feature]
                for feature in feature_order
            }
        ]
    )

    X["type"] = pd.Categorical(
        X["type"],
        categories=categories
    )

    # Calculer les contributions LightGBM
    contributions = model.predict(
        X,
        pred_contrib=True
    )[0]

    feature_contributions = (
        contributions[:-1]
    )

    bias = float(
        contributions[-1]
    )

    result = pd.DataFrame({
        "feature": feature_order,
        "value": [
            ml_row[f]
            for f in feature_order
        ],
        "contribution": feature_contributions
    })

    result["abs_contribution"] = (
        result["contribution"].abs()
    )

    result = result.sort_values(
        "abs_contribution",
        ascending=False
    )

    probability = float(
        model.predict(X)[0]
    )

    raw_score = float(
        model.predict(
            X,
            raw_score=True
        )[0]
    )

    contribution_sum = float(
        feature_contributions.sum()
        + bias
    )

    return {
        "table": result,
        "probability": probability,
        "raw_score": raw_score,
        "bias": bias,
        "contribution_sum": contribution_sum
    }