"""Auswertung: das Netz (Adam, Replay, Zielnetz) gegen eine tabellarische Q-Learning-Referenz auf demselben Modell, dazu die Ablation von Replay
und Zielnetz - Hook dieses Stuecks: bei knappem Trainingsbudget helfen beide Tricks messbar, Replay staerker als das Zielnetz."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import dqn_agent as A
import dqn_constants as C
import dqn_grid as G
import dqn_reference as R
from dqn_features import coords, one_hot, N_COORD_FEATURES


@dataclass(frozen=True)
class Settings:
    rows: int = C.DEFAULT_ROWS
    cols: int = C.DEFAULT_COLS
    slip: float = C.DEFAULT_SLIP
    gamma: float = C.DEFAULT_GAMMA
    hidden: int = C.DEFAULT_HIDDEN
    lr: float = C.DEFAULT_LR
    epsilon_start: float = C.DEFAULT_EPSILON_START
    epsilon_decay: float = C.DEFAULT_EPSILON_DECAY
    episodes: int = C.DEFAULT_EPISODES
    use_replay: bool = True
    use_target: bool = True
    seed: int = 0

    @property
    def grid(self):
        return G.Grid(self.rows, self.cols, self.slip, self.gamma)


@lru_cache(maxsize=64)
def _reference(rows, cols, slip, gamma):
    grid = G.Grid(rows, cols, slip, gamma)
    P, Rw = G.build_model(grid)
    V_star, Q_star, pi_star = R.value_iteration(P, Rw, gamma)
    return grid, P, Rw, V_star, Q_star, pi_star


@dataclass
class Analysis:
    settings: Settings
    grid: G.Grid
    Q: np.ndarray
    returns: np.ndarray
    lengths: np.ndarray
    falls: np.ndarray
    policy: np.ndarray
    V_star: np.ndarray
    pi_star: np.ndarray
    V_pi: np.ndarray
    gap: float
    env_steps: int
    snapshots: dict


def _coord_feature_fn(grid):
    return lambda states: coords(states, grid)


def analyse(s, checkpoints=()):
    grid, P, Rw, V_star, Q_star, pi_star = _reference(s.rows, s.cols, s.slip, s.gamma)
    feature_fn = _coord_feature_fn(grid)
    net, returns, lengths, falls, snapshots_net = A.train(
        grid, feature_fn, N_COORD_FEATURES, s.hidden, s.lr, s.epsilon_start, s.epsilon_decay, s.episodes, s.seed,
        use_replay=s.use_replay, use_target=s.use_target, checkpoints=checkpoints, optimizer="adam",
    )
    Q = A.q_table_from_net(net, feature_fn, grid.n_states)
    policy = Q.argmax(axis=1)
    V_pi = R.policy_evaluation(P, Rw, policy, grid.gamma)
    start_s = grid.state_of(grid.start)
    gap = float(V_star[start_s] - V_pi[start_s])
    snapshots = {ep: A.q_table_from_net(net_ep, feature_fn, grid.n_states) for ep, net_ep in snapshots_net.items()}
    return Analysis(s, grid, Q, returns, lengths, falls, policy, V_star, pi_star, V_pi, gap, int(lengths.sum()), snapshots)


def analyse_tabular_baseline(s):
    """Reines Q-Learning (One-Hot, hidden=0, SGD, kein Replay/Zielnetz) auf demselben Modell - der Vergleichsmassstab der Kernfrage."""
    grid, P, Rw, V_star, Q_star, pi_star = _reference(s.rows, s.cols, s.slip, s.gamma)
    feature_fn = lambda states: one_hot(states, grid.n_states)
    net, returns, lengths, falls, _ = A.train(
        grid, feature_fn, grid.n_states, 0, C.TABULAR_ALPHA, s.epsilon_start, s.epsilon_decay, s.episodes, s.seed,
        use_replay=False, use_target=False, optimizer="sgd",
    )
    Q = A.q_table_from_net(net, feature_fn, grid.n_states)
    policy = Q.argmax(axis=1)
    V_pi = R.policy_evaluation(P, Rw, policy, grid.gamma)
    start_s = grid.state_of(grid.start)
    gap = float(V_star[start_s] - V_pi[start_s])
    return Analysis(s, grid, Q, returns, lengths, falls, policy, V_star, pi_star, V_pi, gap, int(lengths.sum()), {})


def _gap_for(rows, cols, slip, gamma, hidden, lr, epsilon_start, epsilon_decay, episodes, use_replay, use_target, seed):
    grid, P, Rw, V_star, _, _ = _reference(rows, cols, slip, gamma)
    feature_fn = _coord_feature_fn(grid)
    net, _, _, _, _ = A.train(grid, feature_fn, N_COORD_FEATURES, hidden, lr, epsilon_start, epsilon_decay, episodes, seed,
                               use_replay=use_replay, use_target=use_target, optimizer="adam")
    Q = A.q_table_from_net(net, feature_fn, grid.n_states)
    policy = Q.argmax(axis=1)
    V_pi = R.policy_evaluation(P, Rw, policy, gamma)
    start_s = grid.state_of(grid.start)
    return float(V_star[start_s] - V_pi[start_s])


def never_ending_gap(rows, cols, slip, gamma):
    """Wert-Abstand V*(Start) - V(Start) einer Politik, die nie endet und die Klippe nie berührt: ihr Wert ist -1/(1-gamma) (Schritt-Reward -1 ohne Ende),
    der Abstand also V*(Start) + 1/(1-gamma) - nicht schlicht 1/(1-gamma), denn V*(Start) ist bei Rutschen deutlich kleiner als null."""
    grid, _, _, V_star, _, _ = _reference(rows, cols, slip, gamma)
    return float(V_star[grid.state_of(grid.start)] + 1.0 / (1.0 - gamma))


def _summary(gaps):
    gaps = np.asarray(gaps, dtype=float)
    se = float(gaps.std(ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else 0.0
    return {"mean": float(gaps.mean()), "se": se, "max": float(gaps.max()), "frac_near_optimal": float(np.mean(gaps < C.NEAR_OPTIMAL_GAP))}


# --- Experiment: Ablation von Replay und Zielnetz ueber die Trainingsdauer -----------------------------------------------------------------------

def ablation_experiment(checkpoints=None, seeds=None, base=None):
    checkpoints = C.EXP_EPISODE_CHECKPOINTS if checkpoints is None else checkpoints
    seeds = range(C.EXP_SEEDS) if seeds is None else seeds
    base = Settings() if base is None else base
    combos = [(True, True), (True, False), (False, True), (False, False)]
    rows = []
    for use_replay, use_target in combos:
        row = {"use_replay": use_replay, "use_target": use_target}
        for episodes in checkpoints:
            gaps = [_gap_for(base.rows, base.cols, base.slip, base.gamma, base.hidden, base.lr,
                              base.epsilon_start, base.epsilon_decay, episodes, use_replay, use_target, sd)
                    for sd in seeds]
            row[episodes] = _summary(gaps)
        rows.append(row)
    return {"n_seeds": len(list(seeds)), "checkpoints": tuple(checkpoints), "rows": rows}
