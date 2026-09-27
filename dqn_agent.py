"""DQN (Mnih et al. 2015) im Kleinen: dieselbe TD-Lernregel wie Q-Learning, aber mit einem Netz statt einer Tabelle, dazu die beiden
Stabilitaets-Tricks aus dem Original-Paper - ein Replay-Speicher (Training auf einer zufaelligen Mischung vergangener statt nur der letzten
Erfahrung) und ein Zielnetz (der Bootstrap-Wert kommt aus einer aelteren, eingefrorenen Kopie des Netzes statt aus dem sich gerade aendernden
Online-Netz). Mit `hidden=0`, One-Hot-Eingabe, `use_replay=False`, `use_target=False` reduziert sich das exakt auf tabellarisches Q-Learning."""

from collections import deque

import numpy as np

import dqn_constants as C
from dqn_grid import ACTIONS, step
from dqn_net import QNet

N_ACTIONS = len(ACTIONS)


def epsilon_by_episode(episode, start, decay, eps_min=C.EPSILON_MIN):
    if decay <= 0.0:
        return start
    return max(eps_min, start / (1.0 + decay * episode))


def choose_action(q_row, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    best = np.flatnonzero(q_row == q_row.max())
    return int(best[0]) if best.size == 1 else int(rng.choice(best))


def _train_on_batch(online, target, feature_fn, batch, lr, gamma, optimizer):
    states = [b[0] for b in batch]
    actions = np.array([b[1] for b in batch])
    rewards = np.array([b[2] for b in batch], dtype=float)
    next_states = [b[3] for b in batch]
    dones = np.array([b[4] for b in batch], dtype=float)

    X = feature_fn(states)
    Q, cache = online.forward(X)
    Xn = feature_fn(next_states)
    Qn, _ = target.forward(Xn)
    max_next = Qn.max(axis=1)
    targets = rewards + gamma * max_next * (1.0 - dones)

    dQ = np.zeros_like(Q)
    idx = np.arange(len(batch))
    dQ[idx, actions] = Q[idx, actions] - targets
    grads = online.backward(cache, dQ)
    online.step(grads, lr, optimizer)


def train(grid, feature_fn, n_features, hidden, lr, epsilon_start, epsilon_decay, episodes, seed,
          use_replay=True, use_target=True, replay_capacity=C.REPLAY_CAPACITY, batch_size=C.BATCH_SIZE,
          target_update_every=C.TARGET_UPDATE_EVERY, max_steps=C.MAX_STEPS_PER_EPISODE, checkpoints=(),
          optimizer="adam"):
    """Trainiert `episodes` Episoden. `checkpoints` liefern zusaetzlich eine Kopie des Online-Netzes nach genau dieser vielen Episoden."""
    rng = np.random.default_rng(seed)
    online = QNet(n_features, hidden, N_ACTIONS, seed)
    target = online.clone() if use_target else online
    buffer = deque(maxlen=replay_capacity) if use_replay else None

    returns = np.zeros(episodes)
    lengths = np.zeros(episodes, dtype=int)
    falls = np.zeros(episodes, dtype=int)
    snapshots = {}
    checkpoint_set = set(checkpoints)
    global_step = 0

    for e in range(episodes):
        eps = epsilon_by_episode(e, epsilon_start, epsilon_decay)
        s = grid.state_of(grid.start)
        total_reward, steps, n_falls = 0.0, 0, 0
        for _ in range(max_steps):
            Xs = feature_fn([s])
            Qs, _ = online.forward(Xs)
            a = choose_action(Qs[0], eps, rng)
            s_next, r, done = step(grid, s, a, rng)
            if r == C.CLIFF_PENALTY:
                n_falls += 1

            if use_replay:
                buffer.append((s, a, r, s_next, done))
                if len(buffer) > batch_size:
                    idx = rng.choice(len(buffer), size=batch_size, replace=False)
                    batch = [buffer[i] for i in idx]
                else:
                    batch = list(buffer)
            else:
                batch = [(s, a, r, s_next, done)]

            _train_on_batch(online, target, feature_fn, batch, lr, grid.gamma, optimizer)
            global_step += 1
            if use_target and global_step % target_update_every == 0:
                target.load_from(online)

            total_reward += r
            steps += 1
            if done:
                break
            s = s_next
        returns[e] = total_reward
        lengths[e] = steps
        falls[e] = n_falls
        if (e + 1) in checkpoint_set:
            snapshots[e + 1] = online.clone()
    return online, returns, lengths, falls, snapshots


def q_table_from_net(net, feature_fn, n_states):
    """Liest Q(s,.) fuer JEDEN State aus dem Netz aus - fuer Vergleich/Visualisierung wie bei einer Tabelle."""
    X = feature_fn(list(range(n_states)))
    Q, _ = net.forward(X)
    return Q
