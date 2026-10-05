"""Unabhängige Orakel für DQN: Netz-Gradient (Rückwärtspass) gegen eine Schleifen-Neuimplementierung je Probe UND finite Differenzen der Verlustfunktion;
`train` (Replay-Puffer mit Kapazität, Batch-Ziehung, Zielnetz-Synchronisation nach `global_step % every`, Terminalzustände ohne Bootstrap, Adam/SGD) Schritt für
Schritt gegen eine Schleifen-Neuimplementierung auf identischem Zufallsstrom; die Reduktion auf tabellarisches Q-Learning gegen eine reine Dict-/Listen-Schleife;
V* per LP (scipy HiGHS), Policy-Wert per Lineargleichungssystem, und der Wert-Abstand einer nie endenden Politik gegen das LGS."""

import numpy as np
import pytest

import dqn_agent as A
import dqn_evaluation as E
import dqn_grid as G
import dqn_net as N
from dqn_features import N_COORD_FEATURES, coords, one_hot

linprog = pytest.importorskip("scipy.optimize").linprog

_MOVES = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
_NAME = {0: "N", 1: "S", 2: "E", 3: "W"}
_SIDES = {"N": ("E", "W"), "S": ("E", "W"), "E": ("N", "S"), "W": ("N", "S")}


def _outcomes(rows, cols, slip, s, a):
    r, c = divmod(s, cols)
    goal = (rows - 1, cols - 1)
    if (r, c) == goal:
        return [(1.0, s, 0.0, True)]
    cliff = {(rows - 1, k) for k in range(1, cols - 1)}
    m = _NAME[a]
    out = []
    for p, d in ((1 - slip, m), (slip / 2, _SIDES[m][0]), (slip / 2, _SIDES[m][1])):
        nr, nc = r + _MOVES[d][0], c + _MOVES[d][1]
        if not (0 <= nr < rows and 0 <= nc < cols):
            nr, nc = r, c
        if (nr, nc) == goal:
            out.append((p, nr * cols + nc, 10.0, True))
        elif (nr, nc) in cliff:
            out.append((p, (rows - 1) * cols, -100.0, False))
        else:
            out.append((p, nr * cols + nc, -1.0, False))
    return out


def _vstar_lp(P, Rw, gamma):
    S, nA = Rw.shape
    rows = []
    for s in range(S):
        for a in range(nA):
            row = gamma * P[s, a].copy()
            row[s] -= 1.0
            rows.append(row)
    return linprog(np.ones(S), A_ub=np.array(rows), b_ub=-Rw.reshape(-1), bounds=[(None, None)] * S, method="highs").x


class _RefNet:
    """Netz mit Schleifen je Einheit und je Probe (kein Matrixprodukt), Adam und SGD nach Formel."""

    def __init__(self, net):
        self.hidden = net.hidden
        self.p = {k: getattr(net, k).copy() for k in net._param_names}
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def q(self, x, p=None):
        p = self.p if p is None else p
        nf = len(x)
        if self.hidden == 0:
            return [sum(x[i] * p["W1"][i, a] for i in range(nf)) for a in range(p["W1"].shape[1])], None
        h = []
        for j in range(self.hidden):
            z = sum(x[i] * p["W1"][i, j] for i in range(nf)) + p["b1"][j]
            h.append((z, max(z, 0.0)))
        return [sum(h[j][1] * p["W2"][j, a] for j in range(self.hidden)) + p["b2"][a] for a in range(4)], h

    def grads(self, xs, ys, acts):
        """Gradient von L = 0.5 * mean_i (Q_i[a_i] - y_i)^2, Probe für Probe."""
        B = len(xs)
        g = {k: np.zeros_like(v) for k, v in self.p.items()}
        for x, a, y in zip(xs, acts, ys):
            q, h = self.q(x)
            d = (q[a] - y) / B
            if self.hidden == 0:
                for i in range(len(x)):
                    g["W1"][i, a] += x[i] * d
            else:
                g["b2"][a] += d
                for j in range(self.hidden):
                    g["W2"][j, a] += h[j][1] * d
                    dz = d * self.p["W2"][j, a] * (1.0 if h[j][0] > 0 else 0.0)
                    g["b1"][j] += dz
                    for i in range(len(x)):
                        g["W1"][i, j] += x[i] * dz
        return g

    def adam(self, g, lr):
        self.t += 1
        for k in self.p:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            mh, vh = self.m[k] / (1 - 0.9 ** self.t), self.v[k] / (1 - 0.999 ** self.t)
            self.p[k] = self.p[k] - lr * mh / (np.sqrt(vh) + 1e-8)

    def sgd(self, g, lr):
        for k in self.p:
            self.p[k] = self.p[k] - lr * g[k]


def _replay(grid, featfn, hidden, lr, eps0, decay, episodes, seed, use_replay, use_target, cap, bs, every, max_steps, optimizer):
    rng = np.random.default_rng(seed)
    net = _RefNet(N.QNet(len(featfn([0])[0]), hidden, 4, seed))
    tparams = {k: v.copy() for k, v in net.p.items()}
    buf, gs = [], 0
    rets, lens, falls = [], [], []
    for e in range(episodes):
        eps = eps0 if decay <= 0 else max(0.01, eps0 / (1 + decay * e))
        s, tot, n, nf = grid.state_of(grid.start), 0.0, 0, 0
        for _ in range(max_steps):
            qs = np.array(net.q(featfn([s])[0])[0])
            if rng.random() < eps:
                a = int(rng.integers(4))
            else:
                best = np.flatnonzero(qs == qs.max())
                a = int(best[0]) if best.size == 1 else int(rng.choice(best))
            s2, r, d = G.step(grid, s, a, rng)
            nf += r == -100.0
            if use_replay:
                buf.append((s, a, r, s2, d))
                buf = buf[-cap:]                                                                     # Kapazität: nur die letzten `cap` Übergänge
                batch = [buf[i] for i in rng.choice(len(buf), size=bs, replace=False)] if len(buf) > bs else list(buf)
            else:
                batch = [(s, a, r, s2, d)]
            tp = tparams if use_target else net.p
            ys = [b[2] + (0.0 if b[4] else grid.gamma * max(net.q(featfn([b[3]])[0], tp)[0])) for b in batch]
            g = net.grads([featfn([b[0]])[0] for b in batch], ys, [b[1] for b in batch])
            net.adam(g, lr) if optimizer == "adam" else net.sgd(g, lr)
            gs += 1
            if use_target and gs % every == 0:
                tparams = {k: v.copy() for k, v in net.p.items()}
            tot, n = tot + r, n + 1
            if d:
                break
            s = s2
        rets.append(tot)
        lens.append(n)
        falls.append(nf)
    return net.p, np.array(rets), np.array(lens), np.array(falls)


def test_backward_matches_per_sample_reference_and_finite_differences():
    rng = np.random.default_rng(1)
    for hidden, nf, B in [(0, 2, 1), (0, 5, 6), (3, 2, 4), (7, 5, 7)] * 3:
        net = N.QNet(nf, hidden, 4, int(rng.integers(1000)))
        if hidden > 0:
            net.b1, net.b2 = rng.normal(size=hidden), rng.normal(size=4)
        else:
            net.W1 = rng.normal(size=net.W1.shape)
        X, acts, ys = rng.normal(size=(B, nf)), rng.integers(0, 4, B), rng.normal(size=B) * 3
        Q, cache = net.forward(X)
        dQ = np.zeros_like(Q)
        dQ[np.arange(B), acts] = Q[np.arange(B), acts] - ys
        gr = net.backward(cache, dQ)
        ref = _RefNet(net).grads(list(X), ys, acts)

        def loss():
            Qx, _ = net.forward(X)
            return 0.5 * np.mean((Qx[np.arange(B), acts] - ys) ** 2)

        for k in gr:
            assert np.allclose(gr[k], ref[k], atol=1e-10)
            P = getattr(net, k)
            for idx in np.ndindex(P.shape):
                o = P[idx]
                P[idx] = o + 1e-6
                lp = loss()
                P[idx] = o - 1e-6
                lm = loss()
                P[idx] = o
                assert (lp - lm) / 2e-6 == pytest.approx(gr[k][idx], abs=1e-5, rel=1e-5)


def test_train_replays_step_by_step_with_replay_target_net_and_both_optimizers():
    rng = np.random.default_rng(41)
    for _ in range(14):
        rows, cols = int(rng.integers(3, 5)), int(rng.integers(4, 6))
        slip, gamma = float(rng.choice([0, 0.1, 0.3])), float(rng.choice([0.8, 0.95]))
        hidden, lr = int(rng.choice([0, 4, 8])), float(rng.choice([0.001, 0.01, 0.03]))
        ur, ut = bool(rng.integers(2)), bool(rng.integers(2))
        opt = str(rng.choice(["adam", "sgd"])) if hidden == 0 else "adam"                            # SGD mit Hidden-Schicht divergiert (siehe README)
        cap, bs, every = int(rng.choice([5, 40, 5000])), int(rng.choice([1, 4, 32])), int(rng.choice([1, 7, 50]))
        eps0, decay = float(rng.choice([0.3, 1.0])), float(rng.choice([0, 0.1]))
        ep, seed, ms = int(rng.integers(1, 5)), int(rng.integers(1000)), int(rng.choice([20, 60]))
        g = G.Grid(rows, cols, slip, gamma)
        ff = lambda states, g=g: coords(states, g)
        net, ret, ln, fl, _ = A.train(g, ff, N_COORD_FEATURES, hidden, lr, eps0, decay, ep, seed, use_replay=ur, use_target=ut,
                                      replay_capacity=cap, batch_size=bs, target_update_every=every, max_steps=ms, optimizer=opt)
        P, ro, lo, fo = _replay(g, ff, hidden, lr, eps0, decay, ep, seed, ur, ut, cap, bs, every, ms, opt)
        for k in P:
            assert np.allclose(getattr(net, k), P[k], atol=1e-8, rtol=1e-7)
        assert np.array_equal(ret, ro) and np.array_equal(ln, lo) and np.array_equal(fl, fo)


def _q_learning_reference(rows, cols, slip, gamma, alpha, eps0, decay, episodes, seed):
    rng = np.random.default_rng(seed)
    Q = [[0.0] * 4 for _ in range(rows * cols)]
    for e in range(episodes):
        eps = eps0 if decay <= 0 else max(0.01, eps0 / (1 + decay * e))
        s = (rows - 1) * cols
        for _ in range(400):
            if rng.random() < eps:
                a = int(rng.integers(4))
            else:
                best = [k for k in range(4) if Q[s][k] == max(Q[s])]
                a = best[0] if len(best) == 1 else int(rng.choice(np.array(best)))
            u = rng.random()
            k = 0 if (slip == 0 or u < 1 - slip) else (1 if u < 1 - slip / 2 else 2)
            _, s2, r, d = _outcomes(rows, cols, slip, s, a)[k]
            Q[s][a] += alpha * ((r if d else r + gamma * max(Q[s2])) - Q[s][a])
            if d:
                break
            s = s2
    return np.array(Q)


def test_linear_onehot_sgd_without_tricks_equals_tabular_q_learning_of_an_independent_loop():
    rng = np.random.default_rng(5)
    for _ in range(15):
        rows, cols = int(rng.integers(3, 5)), int(rng.integers(4, 7))
        slip, gamma, alpha = float(rng.choice([0, 0.1, 0.3])), float(rng.choice([0.8, 0.95])), float(rng.choice([0.05, 0.1, 0.5]))
        ep, seed = int(rng.integers(1, 25)), int(rng.integers(1000))
        g = G.Grid(rows, cols, slip, gamma)
        net, *_ = A.train(g, lambda st, g=g: one_hot(st, g.n_states), g.n_states, 0, alpha, 1.0, 0.01, ep, seed, use_replay=False, use_target=False, max_steps=400, optimizer="sgd")
        assert np.allclose(net.W1, _q_learning_reference(rows, cols, slip, gamma, alpha, 1.0, 0.01, ep, seed), atol=1e-12, rtol=0)


def test_model_reference_gap_and_never_ending_policy_against_independent_computations():
    for rows, cols, slip, gamma in [(3, 4, 0.0, 0.95), (4, 8, 0.1, 0.95), (4, 6, 0.3, 0.9)]:
        g = G.Grid(rows, cols, slip, gamma)
        P, Rw = G.build_model(g)
        goal_s = g.state_of(g.goal)
        for s in range(g.n_states):
            for a in range(4):
                Pm, Rm = np.zeros(g.n_states), 0.0
                for p, s2, r, d in _outcomes(rows, cols, slip, s, a):
                    Pm[goal_s if d else s2] += p
                    Rm += p * r
                assert np.allclose(P[s, a], Pm) and Rw[s, a] == pytest.approx(Rm)
        Vlp = _vstar_lp(P, Rw, gamma)
        s0, idx = g.state_of(g.start), np.arange(g.n_states)
        west = np.full(g.n_states, 3)                                                               # läuft dauerhaft gegen die Westwand, berührt nie die Klippe
        Vw = np.linalg.solve(np.eye(g.n_states) - gamma * P[idx, west], Rw[idx, west])
        assert Vw[s0] == pytest.approx(-1.0 / (1.0 - gamma), abs=1e-9)
        assert E.never_ending_gap(rows, cols, slip, gamma) == pytest.approx(Vlp[s0] - Vw[s0], abs=1e-5)


def test_analysis_gap_and_snapshots_against_lp_and_the_net_itself():
    for seed in range(3):
        s = E.Settings(rows=3, cols=5, slip=0.1, gamma=0.95, hidden=8, lr=0.01, episodes=4, seed=seed)
        a = E.analyse(s, checkpoints=(2, 4))
        P, Rw = G.build_model(a.grid)
        idx, s0 = np.arange(a.grid.n_states), a.grid.state_of(a.grid.start)
        Vpi = np.linalg.solve(np.eye(a.grid.n_states) - a.grid.gamma * P[idx, a.policy], Rw[idx, a.policy])
        assert a.gap == pytest.approx(_vstar_lp(P, Rw, a.grid.gamma)[s0] - Vpi[s0], abs=1e-4)
        assert a.env_steps == int(a.lengths.sum())
        ff = lambda states: coords(states, a.grid)
        short, *_ = A.train(a.grid, ff, N_COORD_FEATURES, 8, 0.01, 1.0, 0.1, 2, seed, use_replay=True, use_target=True)
        assert np.allclose(a.snapshots[2], A.q_table_from_net(short, ff, a.grid.n_states))
        assert np.allclose(a.snapshots[4], a.Q)
