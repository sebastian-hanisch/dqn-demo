"""Der Trainings-Agent: gierige/erkundende Aktionswahl, UND die zentrale Korrektheits-Kette dieses Stuecks - ein rein lineares Netz (hidden=0) mit
One-Hot-Eingabe, dem SGD-Optimierer, ohne Replay und ohne Zielnetz reduziert sich EXAKT auf tabellarisches Q-Learning (byte-gleiche Q-Tabelle bei
gleichem Seed, ueber ein ganzes Training)."""

import numpy as np
import pytest

import dqn_agent as A
import dqn_constants as C
from dqn_features import one_hot
from dqn_grid import Grid, step


def test_choose_action_epsilon_zero_is_always_greedy():
    q_row = np.array([1.0, 5.0, 2.0, 0.0])
    rng = np.random.default_rng(0)
    for _ in range(50):
        assert A.choose_action(q_row, epsilon=0.0, rng=rng) == 1


def test_choose_action_breaks_ties_among_all_maxima():
    q_row = np.array([5.0, 0.0, 5.0, 0.0])
    rng = np.random.default_rng(1)
    seen = {A.choose_action(q_row, epsilon=0.0, rng=rng) for _ in range(200)}
    assert seen == {0, 2}


def test_zero_hidden_onehot_sgd_no_tricks_reduces_exactly_to_tabular_q_learning():
    grid = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    feature_fn = lambda states: one_hot(states, grid.n_states)
    net, returns, lengths, falls, _ = A.train(
        grid, feature_fn, grid.n_states, hidden=0, lr=0.1, epsilon_start=1.0, epsilon_decay=0.005,
        episodes=50, seed=5, use_replay=False, use_target=False, max_steps=200, optimizer="sgd",
    )
    Q_net = A.q_table_from_net(net, feature_fn, grid.n_states)

    rng = np.random.default_rng(5)
    Q_tab = np.zeros((grid.n_states, 4))

    def eps_by_ep(e, start, decay, eps_min=0.01):
        return max(eps_min, start / (1.0 + decay * e)) if decay > 0 else start

    for e in range(50):
        eps = eps_by_ep(e, 1.0, 0.005)
        s = grid.state_of(grid.start)
        for _ in range(200):
            a = A.choose_action(Q_tab[s], eps, rng)
            s_next, r, done = step(grid, s, a, rng)
            target = r if done else r + grid.gamma * Q_tab[s_next].max()
            Q_tab[s, a] += 0.1 * (target - Q_tab[s, a])
            if done:
                break
            s = s_next
    assert np.array_equal(Q_net, Q_tab)


def test_train_snapshot_at_a_checkpoint_equals_a_separately_truncated_run():
    grid = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    from dqn_features import coords, N_COORD_FEATURES
    feature_fn = lambda states: coords(states, grid)
    net_full, _, _, _, snapshots = A.train(grid, feature_fn, N_COORD_FEATURES, hidden=16, lr=0.01,
                                            epsilon_start=1.0, epsilon_decay=0.005, episodes=30, seed=3, checkpoints=(10,))
    net_short, _, _, _, _ = A.train(grid, feature_fn, N_COORD_FEATURES, hidden=16, lr=0.01,
                                     epsilon_start=1.0, epsilon_decay=0.005, episodes=10, seed=3)
    assert np.array_equal(snapshots[10].W1, net_short.W1)


def test_train_returns_arrays_of_the_right_shape():
    grid = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    from dqn_features import coords, N_COORD_FEATURES
    feature_fn = lambda states: coords(states, grid)
    net, returns, lengths, falls, snapshots = A.train(grid, feature_fn, N_COORD_FEATURES, hidden=16, lr=0.01,
                                                       epsilon_start=1.0, epsilon_decay=0.005, episodes=15, seed=0)
    assert returns.shape == (15,) and lengths.shape == (15,) and falls.shape == (15,)
    assert np.all(lengths >= 1) and np.all(lengths <= C.MAX_STEPS_PER_EPISODE)


def test_q_table_from_net_covers_every_state():
    grid = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    from dqn_features import coords, N_COORD_FEATURES
    feature_fn = lambda states: coords(states, grid)
    net, *_ = A.train(grid, feature_fn, N_COORD_FEATURES, hidden=8, lr=0.01, epsilon_start=1.0, epsilon_decay=0.005, episodes=5, seed=0)
    Q = A.q_table_from_net(net, feature_fn, grid.n_states)
    assert Q.shape == (grid.n_states, 4)
