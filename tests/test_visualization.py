"""Regressionstest fuer den in `value-iteration-demo`/`q-learning-demo` gefundenen Plotly-Bug (deutsch formatierte Dezimal-Strings werden als
US-tausendergruppierte Ganzzahl geparst, wenn die x-Achse nicht explizit kategorial ist) - hier unkritisch, da die x-Achse der Ablation echte
Episodenzahlen (Ganzzahlen ohne Dezimalkomma) verwendet; Grundfunktion der Diagramme wird trotzdem geprueft."""

import dqn_evaluation as E
from dqn_visualization import build_ablation


def test_ablation_chart_has_one_trace_per_combination():
    exp = E.ablation_experiment(checkpoints=(10, 20), seeds=range(3))
    fig = build_ablation(exp)
    assert len(fig.data) == 4
    assert list(fig.data[0].x) == [10, 20]
