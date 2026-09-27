"""Analyse und das Ablations-Experiment: Aufbau, und die Korrektheits-Kette dieses Stuecks auf Analyse-Ebene."""

import numpy as np
import pytest

import dqn_evaluation as E


@pytest.fixture(scope="module")
def analysis():
    return E.analyse(E.Settings(episodes=15, seed=0))


def test_analyse_wiring(analysis):
    a = analysis
    assert a.Q.shape == (a.grid.n_states, 4)
    assert a.returns.shape == (15,) and a.lengths.shape == (15,)
    assert a.env_steps == int(a.lengths.sum())


def test_analyse_is_deterministic_given_the_same_seed():
    s = E.Settings(rows=3, cols=8, episodes=10, seed=5)
    a1, a2 = E.analyse(s), E.analyse(s)
    assert np.array_equal(a1.Q, a2.Q)


def test_reference_is_cached_across_settings_with_the_same_grid():
    assert E._reference(4, 8, 0.10, 0.95) is E._reference(4, 8, 0.10, 0.95)


def test_tabular_baseline_gets_close_to_optimal_given_enough_episodes():
    a = E.analyse_tabular_baseline(E.Settings(rows=4, cols=8, episodes=100, seed=0))
    assert a.gap < 5.0


def test_ablation_experiment_shape():
    exp = E.ablation_experiment(checkpoints=(10, 20), seeds=range(3))
    assert len(exp["rows"]) == 4
    combos = {(r["use_replay"], r["use_target"]) for r in exp["rows"]}
    assert combos == {(True, True), (True, False), (False, True), (False, False)}
    for row in exp["rows"]:
        assert 10 in row and 20 in row and "mean" in row[10]


def test_a_never_terminating_greedy_policy_always_has_the_same_value():
    # Der eigentliche Befund dieses Stuecks: bei knappem Trainingsbudget erreicht die gierige Politik das Ziel praktisch nie - sie irrt im
    # sicheren Innenraum umher, bis der Schritt-Deckel greift. JEDE solche "ewig laufende" Politik hat exakt denselben Wert -1/(1-gamma),
    # unabhaengig von der genauen Route (kein Programmfehler, sondern die Konstante, gegen die alle vier Ablations-Kombinationen anfangs laufen).
    # Ohne Rutschen (slip=0) faellt niemand "zufaellig" in die Klippe, deshalb gilt das hier sauber - MIT Rutschen (App-Standard) mischt sich
    # zusaetzlich die Chance eines fruehen Klippensturzes hinein (siehe Ablations-Befund in der App: dort ist auch der Mittelwert kein sauberer
    # Fixpunkt mehr).
    s = E.Settings(slip=0.0, episodes=5, seed=0)
    grid, _, _, V_star, _, _ = E._reference(s.rows, s.cols, s.slip, s.gamma)
    expected_gap = V_star[grid.state_of(grid.start)] + 1.0 / (1.0 - s.gamma)
    exp = E.ablation_experiment(checkpoints=(5,), seeds=range(4), base=s)
    for row in exp["rows"]:
        assert row[5]["mean"] == pytest.approx(expected_gap, abs=1.0)
