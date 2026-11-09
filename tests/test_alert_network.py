import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "utils" / "alert_network.py"
if not MODULE.exists():
    MODULE = ROOT / "app" / "utils" / "alert_network.py"
spec = importlib.util.spec_from_file_location("alert_network", MODULE)
network_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network_module)

THEME = {"font": "#9fb0c5", "cyan": "#38bdf8"}


def sample_rows():
    return pd.DataFrame([
        dict(transaction_id=10, nameOrig="A", nameDest="B", isFraud=1, amount=50.0, fraud_probability=0.95, step=400),
        dict(transaction_id=11, nameOrig="C", nameDest="B", isFraud=0, amount=70.0, fraud_probability=0.4, step=401),
        dict(transaction_id=12, nameOrig="D", nameDest="E", isFraud=1, amount=90.0, fraud_probability=0.99, step=402),
    ])


class NetworkTests(unittest.TestCase):
    def test_groups_and_unique_accounts(self):
        network = network_module.prepare_network(sample_rows())
        self.assertEqual(network["account_count"], 5)
        self.assertEqual(network["group_count"], 2)
        self.assertEqual(network["largest_group"], 3)
        self.assertEqual(network["components"], [("A", "B", "C"), ("D", "E")])

    def test_layout_is_deterministic_and_does_not_modify_input(self):
        rows = sample_rows()
        before = rows.copy(deep=True)
        first = network_module.prepare_network(rows)
        second = network_module.prepare_network(rows.sample(frac=1, random_state=3))
        self.assertEqual(first["positions"], second["positions"])
        pd.testing.assert_frame_equal(rows, before)

    def test_component_centers_are_spaced_and_not_a_grid(self):
        centers = network_module._component_centers(55)
        self.assertEqual(centers.shape, (55, 2))
        np.testing.assert_array_equal(centers, network_module._component_centers(55))
        self.assertGreater(pdist(centers).min(), 2.0)
        self.assertGreater(len(np.unique(centers[:, 1].round(3))), 50)

    def test_components_have_different_link_directions(self):
        rows = pd.concat([sample_rows().iloc[[0]].assign(
            transaction_id=index, nameOrig=f"S{index}", nameDest=f"D{index}"
        ) for index in range(30)], ignore_index=True)
        network = network_module.prepare_network(rows)
        directions = []
        for row in rows.itertuples(index=False):
            source = np.array(network["positions"][row.nameOrig])
            destination = np.array(network["positions"][row.nameDest])
            dx, dy = destination - source
            directions.append(round(float(np.arctan2(dy, dx)), 2))
        self.assertGreater(len(set(directions)), 20)

    def test_glow_is_theme_aware_and_selection_is_explicit(self):
        network = network_module.prepare_network(sample_rows())
        dark = network_module.build_network_figure(network, dict(THEME, network_glow=True), 10)
        light = network_module.build_network_figure(network, dict(THEME, network_glow=False), 10)
        self.assertTrue(any(trace.name == "Halo comptes" for trace in dark.data))
        self.assertFalse(any(trace.name == "Halo comptes" for trace in light.data))
        selected = next(trace for trace in dark.data if trace.name == "Sélection")
        self.assertEqual(list(selected.customdata), [[10], [10]])
        self.assertEqual(len(next(trace for trace in dark.data if trace.name == "Transactions").x), 3)

    def test_filter_truth_is_not_confused_with_predictions(self):
        rows = sample_rows()
        self.assertEqual(network_module.filter_alerts(rows, "Fraudes détectées").transaction_id.tolist(), [10, 12])
        self.assertEqual(network_module.filter_alerts(rows, "Fausses alertes").transaction_id.tolist(), [11])
        self.assertEqual(len(network_module.filter_alerts(rows, "Toutes les alertes")), 3)
        with self.assertRaises(ValueError):
            network_module.filter_alerts(rows, "anything")

    def test_all_transactions_are_plotted_with_correct_metadata(self):
        figure = network_module.build_network_figure(network_module.prepare_network(sample_rows()), THEME)
        points = next(trace for trace in figure.data if trace.name == "Transactions")
        self.assertEqual([row[0] for row in points.customdata], [10, 11, 12])
        self.assertEqual(points.customdata[1][1:3], ["C", "B"])
        self.assertEqual(points.customdata[1][-1], "Fausse alerte")
        self.assertEqual(len(next(trace for trace in figure.data if trace.name == "Comptes").x), 5)

    def test_edges_are_separated_not_joined_across_components(self):
        network = network_module.prepare_network(sample_rows())
        figure = network_module.build_network_figure(network, THEME)
        trace = next(trace for trace in figure.data if trace.name == "Fraudes détectées")
        self.assertEqual(tuple(trace.x), (network["positions"]["A"][0], network["positions"]["B"][0], None, network["positions"]["D"][0], network["positions"]["E"][0], None))

    def test_focus_includes_the_whole_selected_component_only(self):
        figure = network_module.build_network_figure(network_module.prepare_network(sample_rows()), THEME, 10, True)
        points = next(trace for trace in figure.data if trace.name == "Transactions")
        self.assertEqual([row[0] for row in points.customdata], [10, 11])
        self.assertEqual(len(next(trace for trace in figure.data if trace.name == "Comptes").x), 3)

    def test_empty_filter_renders_without_failure(self):
        network = network_module.prepare_network(sample_rows().iloc[0:0])
        self.assertEqual(network["account_count"], 0)
        self.assertEqual(network["largest_group"], 0)
        figure = network_module.build_network_figure(network, THEME)
        self.assertTrue(figure.layout.annotations)

    def test_parallel_edges_and_self_loops_are_not_dropped(self):
        rows = sample_rows()
        rows.loc[1, ["nameOrig", "nameDest"]] = ["A", "B"]
        rows.loc[2, ["nameOrig", "nameDest"]] = ["D", "D"]
        network = network_module.prepare_network(rows)
        figure = network_module.build_network_figure(network, THEME)
        points = next(trace for trace in figure.data if trace.name == "Transactions")
        self.assertEqual(len(points.x), 3)
        self.assertEqual(network["account_count"], 3)

    def test_selection_only_accepts_current_filtered_transactions(self):
        self.assertEqual(network_module.selected_transaction([{"customdata": [10]}], {10}), 10)
        self.assertIsNone(network_module.selected_transaction([{"customdata": [10]}], {11}))
        self.assertIsNone(network_module.selected_transaction([{}, {"customdata": ["bad"]}], {10}))
        self.assertIsNone(network_module.selected_transaction([], {10}))

    def test_hover_text_escapes_account_identifiers(self):
        rows = sample_rows()
        rows.loc[0, "nameOrig"] = "<account>"
        figure = network_module.build_network_figure(network_module.prepare_network(rows), THEME)
        points = next(trace for trace in figure.data if trace.name == "Transactions")
        self.assertEqual(points.customdata[0][1], "&lt;account&gt;")


if __name__ == "__main__":
    unittest.main()
