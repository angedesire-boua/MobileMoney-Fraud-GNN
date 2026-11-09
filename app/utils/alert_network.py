"""Faithful, deterministic layout of the saved Test alert subgraph."""

from html import escape
from math import cos, pi, sin, sqrt

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from scipy.stats import qmc


FILTER_LABELS = ("Toutes les alertes", "Fraudes détectées", "Fausses alertes")
LAYOUT_VERSION = "organic-v1"


def _component_centers(count):
    if count == 0:
        return np.empty((0, 2))
    if count == 1:
        return np.zeros((1, 2))
    radius = 0.58 / sqrt(max(count, 4))
    # Poisson-disk packing leaves space between independent components without
    # imposing rows, columns or nonexistent network connections.
    for attempt in range(4):
        points = qmc.PoissonDisk(
            2, radius=radius * 0.9 ** attempt, ncandidates=8, rng=17
        ).fill_space()
        points = points[np.linalg.norm(points - 0.5, axis=1) < 0.49]
        if len(points) >= count:
            indices = np.random.default_rng(31).permutation(len(points))[:count]
            return (points[indices] - 0.5) * (4.4 * sqrt(max(count, 4)))
    raise RuntimeError("Unable to allocate separated network components")


def filter_alerts(alerts, scope):
    if scope not in FILTER_LABELS:
        raise ValueError("Unknown alert network scope")
    if scope == FILTER_LABELS[1]:
        return alerts.loc[alerts["isFraud"].eq(1)].copy()
    if scope == FILTER_LABELS[2]:
        return alerts.loc[alerts["isFraud"].eq(0)].copy()
    return alerts.copy()


def prepare_network(alerts):
    rows = alerts.sort_values("transaction_id").reset_index(drop=True).copy()
    graph = nx.MultiDiGraph()
    for row in rows.itertuples(index=False):
        graph.add_edge(str(row.nameOrig), str(row.nameDest), key=int(row.transaction_id))

    components = sorted(
        (tuple(sorted(nodes)) for nodes in nx.weakly_connected_components(graph)),
        key=lambda nodes: (-len(nodes), nodes),
    )
    positions, membership = {}, {}
    centers = _component_centers(len(components))
    rng = np.random.default_rng(29)
    for group, nodes in enumerate(components):
        center_x, center_y = centers[group]
        angle = rng.uniform(0, 2 * pi)
        if len(nodes) == 1:
            local = {nodes[0]: (0.0, 0.0)}
        elif len(nodes) == 2:
            length = rng.uniform(0.52, 0.7)
            local = {nodes[0]: (-length, 0.0), nodes[1]: (length, 0.0)}
        else:
            local = nx.spring_layout(graph.subgraph(nodes), seed=17, scale=0.65)
        for node in nodes:
            x, y = float(local[node][0]), float(local[node][1])
            rotated_x = x * cos(angle) - y * sin(angle)
            rotated_y = x * sin(angle) + y * cos(angle)
            positions[node] = (2.3 * center_x + rotated_x, center_y + rotated_y)
            membership[node] = group

    rows["component"] = rows["nameOrig"].astype(str).map(membership)
    return {
        "rows": rows,
        "positions": positions,
        "components": components,
        "account_count": graph.number_of_nodes(),
        "group_count": len(components),
        "largest_group": max(map(len, components), default=0),
    }


def selected_transaction(points, allowed_ids):
    """Accept only a plotted transaction belonging to the current filtered set."""
    for point in reversed(points):
        custom = point.get("customdata")
        if not isinstance(custom, (list, tuple)) or not custom:
            continue
        try:
            transaction_id = int(custom[0])
        except (TypeError, ValueError, OverflowError):
            continue
        if transaction_id in allowed_ids:
            return transaction_id
    return None


def _edge_path(source, destination):
    if source != destination:
        return [source[0], destination[0]], [source[1], destination[1]]
    angles = [2 * pi * step / 16 for step in range(17)]
    return (
        [source[0] + 0.24 * sin(angle) for angle in angles],
        [source[1] + 0.24 * (1 - cos(angle)) for angle in angles],
    )


def build_network_figure(network, theme, selected_id=None, focus=False, height=260):
    rows = network["rows"]
    chosen = rows.loc[rows["transaction_id"].eq(selected_id)]
    if focus and not chosen.empty:
        rows = rows.loc[rows["component"].eq(chosen.iloc[0]["component"])]

    figure = go.Figure()
    positions = network["positions"]
    detailed = len(rows) <= 20
    glow = theme.get("network_glow", True)
    colors = {1: "#38bdf8" if glow else "#0284c7", 0: "#62dfb1" if glow else "#15805d"}
    selection_color = "#fb6078" if glow else "#c92a48"
    for label, title in ((1, "Fraudes détectées"), (0, "Fausses alertes")):
        x, y = [], []
        for row in rows.loc[rows["isFraud"].eq(label)].itertuples(index=False):
            edge_x, edge_y = _edge_path(positions[str(row.nameOrig)], positions[str(row.nameDest)])
            x.extend(edge_x + [None])
            y.extend(edge_y + [None])
        if x:
            if glow:
                figure.add_trace(go.Scattergl(
                    x=x, y=y, mode="lines", showlegend=False,
                    line=dict(color=colors[label], width=4 if detailed else 2.3),
                    opacity=0.08, hoverinfo="skip", name=f"Halo liens {label}",
                ))
            figure.add_trace(go.Scattergl(
                x=x, y=y, mode="lines", name=title,
                line=dict(color=colors[label], width=1.5 if detailed else 0.75),
                opacity=0.85 if detailed else 0.65,
                hoverinfo="skip", showlegend=True,
            ))

    nodes = sorted(set(rows["nameOrig"].astype(str)) | set(rows["nameDest"].astype(str)))
    representative = {}
    positive_nodes = set()
    for row in rows.itertuples(index=False):
        for node in (str(row.nameOrig), str(row.nameDest)):
            representative.setdefault(node, int(row.transaction_id))
            if row.isFraud:
                positive_nodes.add(node)
    node_colors = [colors[int(node in positive_nodes)] for node in nodes]
    node_x = [positions[node][0] for node in nodes]
    node_y = [positions[node][1] for node in nodes]
    if glow:
        for size, opacity in ((22 if detailed else 6, 0.06), (15 if detailed else 3.8, 0.17)):
            figure.add_trace(go.Scattergl(
                x=node_x, y=node_y, mode="markers", showlegend=False,
                marker=dict(size=size, color=node_colors, opacity=opacity),
                hoverinfo="skip", name="Halo comptes",
            ))
    figure.add_trace(go.Scattergl(
        x=node_x, y=node_y,
        mode="markers+text" if detailed else "markers",
        marker=dict(size=7 if detailed else 1.8, color=node_colors,
                    line=dict(width=0.3 if glow else 0, color="#c7f0ff")),
        text=[escape(node) for node in nodes], textposition="top center",
        textfont=dict(size=10, color=theme["font"]),
        customdata=[[representative[node]] for node in nodes],
        name="Comptes", hovertemplate="Compte %{text}<extra></extra>",
    ))

    mid_x, mid_y, custom, marker_colors = [], [], [], []
    for row in rows.itertuples(index=False):
        start, end = positions[str(row.nameOrig)], positions[str(row.nameDest)]
        mid_x.append((start[0] + end[0]) / 2)
        mid_y.append((start[1] + end[1]) / 2 + (0.48 if start == end else 0))
        custom.append([
            int(row.transaction_id), escape(str(row.nameOrig)), escape(str(row.nameDest)),
            float(row.amount), float(row.fraud_probability), int(row.step),
            "Fraude détectée" if row.isFraud else "Fausse alerte",
        ])
        marker_colors.append(colors[int(row.isFraud)])
        if detailed:
            figure.add_annotation(
                x=end[0], y=end[1], ax=start[0], ay=start[1],
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
                arrowcolor=colors[int(row.isFraud)], standoff=7,
            )
    figure.add_trace(go.Scattergl(
        x=mid_x, y=mid_y, customdata=custom, mode="markers",
        marker=dict(size=5 if detailed else 1.3, color=marker_colors, opacity=0.65),
        name="Transactions", showlegend=False,
        hovertemplate=(
            "<b>Transaction %{customdata[0]}</b><br>"
            "%{customdata[1]} → %{customdata[2]}<br>"
            "Montant : %{customdata[3]:,.2f}<br>Score : %{customdata[4]:.4f}<br>"
            "Step : %{customdata[5]}<br>%{customdata[6]}<extra></extra>"
        ),
    ))
    if not chosen.empty:
        row = chosen.iloc[0]
        start, end = positions[str(row.nameOrig)], positions[str(row.nameDest)]
        edge_x, edge_y = _edge_path(start, end)
        if glow:
            figure.add_trace(go.Scattergl(
                x=edge_x, y=edge_y, mode="lines", showlegend=False,
                line=dict(width=7 if detailed else 4, color=selection_color),
                opacity=0.1, hoverinfo="skip", name="Halo sélection",
            ))
            figure.add_trace(go.Scattergl(
                x=[start[0], end[0]], y=[start[1], end[1]], mode="markers",
                marker=dict(size=28 if detailed else 12, color=selection_color, opacity=0.13),
                showlegend=False, hoverinfo="skip", name="Halo sélection",
            ))
        figure.add_trace(go.Scattergl(
            x=edge_x, y=edge_y, mode="lines", showlegend=False,
            line=dict(width=2 if detailed else 1.3, color=selection_color),
            hoverinfo="skip", name="Lien sélectionné",
        ))
        figure.add_trace(go.Scattergl(
            x=[start[0], end[0]], y=[start[1], end[1]],
            mode="markers", marker=dict(size=8 if detailed else 4, color=selection_color),
            customdata=[[int(row.transaction_id)], [int(row.transaction_id)]],
            showlegend=True, name="Sélection",
            hovertemplate="Transaction sélectionnée %{customdata[0]}<extra></extra>",
        ))

    figure.update_layout(
        height=height, margin=dict(l=16, r=16, t=28, b=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#061624" if glow else "#fbfdff",
        font=dict(color=theme["font"], size=10),
        legend=dict(orientation="h", x=0, y=1.16, font=dict(size=10), itemclick=False, itemdoubleclick=False, itemsizing="constant"),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        dragmode="pan", clickmode="event+select", hovermode="closest",
        uirevision=f"{LAYOUT_VERSION}-{len(network['rows'])}-{'focus-' + str(selected_id) if focus else 'all'}",
        modebar=dict(bgcolor="rgba(0,0,0,0)", color=theme["font"], activecolor=theme["cyan"]),
    )
    if nodes:
        xs, ys = zip(*(positions[node] for node in nodes))
        pad_x = max(0.6, (max(xs) - min(xs)) * 0.025)
        pad_y = max(0.8, (max(ys) - min(ys)) * 0.07)
        figure.update_xaxes(range=[min(xs) - pad_x, max(xs) + pad_x])
        figure.update_yaxes(range=[min(ys) - pad_y, max(ys) + pad_y])
    else:
        figure.add_annotation(text="Aucune alerte dans ce périmètre", x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    return figure
