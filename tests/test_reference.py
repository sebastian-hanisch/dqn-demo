"""Die Referenzloesung (Value Iteration, nur zur Gegenprobe): Bellman-Formel von Hand, entarteter Fall mit geschlossener Loesung."""

import numpy as np
import pytest

import dqn_grid as G
import dqn_reference as R


def test_q_values_by_hand():
    P = np.array([[[0.5, 0.5], [1.0, 0.0]]])
    Rw = np.array([[1.0, 2.0]])
    V = np.array([10.0, 20.0])
    Q = R.q_values(P, Rw, V, gamma=0.9)
    assert Q[0, 0] == pytest.approx(1.0 + 0.9 * (0.5 * 10 + 0.5 * 20)) and Q[0, 1] == pytest.approx(2.0 + 0.9 * 10.0)


def test_degenerate_single_row_grid_gives_up_and_bounces_forever():
    g = G.Grid(rows=1, cols=3, slip=0.0, gamma=0.95)
    P, Rw = G.build_model(g)
    V, Q, policy = R.value_iteration(P, Rw, g.gamma)
    s0 = g.state_of(g.start)
    assert V[s0] == pytest.approx(-1.0 / (1.0 - g.gamma), abs=1e-4)
    assert policy[s0] == G.NORTH


def test_policy_evaluation_of_the_optimal_policy_matches_value_iteration():
    g = G.Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    P, Rw = G.build_model(g)
    V_star, _, pi_star = R.value_iteration(P, Rw, g.gamma)
    V_pi = R.policy_evaluation(P, Rw, pi_star, g.gamma)
    assert np.allclose(V_star, V_pi, atol=1e-3)
