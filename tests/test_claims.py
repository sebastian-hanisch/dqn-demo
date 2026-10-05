"""Jede Zahl im README gegen den tatsaechlichen Code, nicht nur behauptet. Einzellaeufe sind deterministisch (fester Seed) und werden exakt
geprueft; die Ablation ist der eigentliche Befund dieses Stuecks (hohe Seed-Varianz statt einer sauberen Rangfolge) und wird deshalb mit den
GLEICHEN Einstellungen wie im README nachgerechnet, nicht mit einer kleineren, schnelleren Stichprobe - das waere sonst kein Test der README-Zahlen,
sondern nur ein Test irgendeiner anderen Zahl. Dadurch ist dieser eine Test langsam (mehrere Minuten), siehe README/Tests."""

import pytest

import dqn_evaluation as E


def test_standard_case_core_question():
    # Standardfall (Kernfrage), Raster/Rutschen/Episoden = App-Standard, Seed 0.
    s = E.Settings(seed=0)
    a = E.analyse(s)
    baseline = E.analyse_tabular_baseline(s)
    assert a.gap == pytest.approx(10.77, abs=0.5)
    assert baseline.gap == pytest.approx(0.68, abs=0.5)
    assert a.env_steps == pytest.approx(10_924, abs=50)
    assert baseline.env_steps == pytest.approx(4_353, abs=50)
    assert E.never_ending_gap(s.rows, s.cols, s.slip, s.gamma) == pytest.approx(10.77, abs=0.05)  # der Netz-Lauf kommt nie ans Ziel: genau dieser Abstand


def test_never_terminating_policy_ceiling_without_slip():
    # Der eigentliche Kern-Befund: bei knappem Budget ohne Rutschen hat JEDE der vier Kombinationen (die alle noch nie das Ziel erreichen)
    # denselben Wert-Abstand ~19,9 = V*(Start) - (-1/(1-gamma)) - keine der vier "gewinnt" hier schon, weil noch niemand ans Ziel kommt.
    s = E.Settings(slip=0.0, episodes=5, seed=0)
    grid, _, _, V_star, _, _ = E._reference(s.rows, s.cols, s.slip, s.gamma)
    expected = V_star[grid.state_of(grid.start)] + 1.0 / (1.0 - s.gamma)
    assert expected == pytest.approx(19.9, abs=0.5)
    exp = E.ablation_experiment(checkpoints=(5,), seeds=range(4), base=s)
    for row in exp["rows"]:
        assert row[5]["mean"] == pytest.approx(expected, abs=1.0)


def test_ablation_experiment_shows_no_clean_ranking():
    # Mit den EXAKTEN Standardeinstellungen (10 Seeds, Checkpoints 60/130/200, App-Standardrutschen) - deterministisch bei festen Seeds,
    # deshalb bewusst NICHT mit weniger Seeds nachgerechnet (das waere eine andere Zahl als im README). Laufzeit: mehrere Minuten.
    exp = E.ablation_experiment()
    rows = {(r["use_replay"], r["use_target"]): r for r in exp["rows"]}
    expected = {
        (True, True): {60: 10.97, 130: 18.83, 200: 16.04},
        (True, False): {60: 32.56, 130: 38.67, 200: 27.28},
        (False, True): {60: 25.14, 130: 13.16, 200: 25.05},
        (False, False): {60: 14.48, 130: 14.61, 200: 16.27},
    }
    for combo, checkpoints in expected.items():
        for ep, value in checkpoints.items():
            assert rows[combo][ep]["mean"] == pytest.approx(value, abs=1.0)
    # README: einzelne Seeds bis 109,8 (der Mittelwert 38,7 verdeckt sie)
    assert rows[(True, False)][130]["max"] == pytest.approx(109.8, abs=3.0)
    assert max(rows[c][ep]["max"] for c in rows for ep in exp["checkpoints"]) == pytest.approx(109.8, abs=3.0)
    # Der Befund selbst: es gibt keine Kombination, die an JEDEM Checkpoint die beste ist (keine saubere Rangfolge).
    winners = {min(rows, key=lambda c: rows[c][ep]["mean"]) for ep in exp["checkpoints"]}
    assert len(winners) > 1
