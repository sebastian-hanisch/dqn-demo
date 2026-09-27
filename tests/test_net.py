"""Das Netz von Hand: Vorwaerts-/Rueckwaertspass gegen eine numerische Gradientenpruefung (wie in `mlp-backprop-demo`), die SGD-Delta-Regel, und
Adams Update-Formel von Hand nachgerechnet."""

import numpy as np
import pytest

from dqn_net import QNet


def test_forward_linear_no_hidden_is_a_plain_matrix_multiply():
    net = QNet(n_features=3, hidden=0, n_actions=2, seed=0)
    net.W1 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X = np.array([[1.0, 0.0, 1.0]])
    Q, _ = net.forward(X)
    assert np.allclose(Q, [[6.0, 8.0]])


def test_forward_with_hidden_applies_relu():
    net = QNet(n_features=2, hidden=3, n_actions=2, seed=0)
    net.W1 = np.array([[1.0, -1.0, 0.5], [0.0, 1.0, -0.5]])
    net.b1 = np.zeros(3)
    net.W2 = np.eye(3, 2)
    net.b2 = np.zeros(2)
    X = np.array([[1.0, 1.0]])
    Q, (_, Z1, A1) = net.forward(X)
    assert np.allclose(Z1, [[1.0, 0.0, 0.0]])
    assert np.allclose(A1, [[1.0, 0.0, 0.0]])  # ReLU: negative/Null-Eintraege werden 0


def test_backward_matches_a_numerical_gradient_check():
    rng = np.random.default_rng(0)
    net = QNet(n_features=3, hidden=4, n_actions=2, seed=1)
    X = rng.normal(size=(5, 3))
    target_action = np.array([0, 1, 0, 1, 0])
    targets = rng.normal(size=5)

    def loss_and_grad(net):
        Q, cache = net.forward(X)
        dQ = np.zeros_like(Q)
        idx = np.arange(5)
        dQ[idx, target_action] = Q[idx, target_action] - targets
        loss = 0.5 * np.mean((Q[idx, target_action] - targets) ** 2)
        return loss, net.backward(cache, dQ)

    loss0, grads = loss_and_grad(net)
    eps = 1e-5
    for name in net._param_names:
        param = getattr(net, name)
        it = np.nditer(param, flags=["multi_index"])
        checked = 0
        while not it.finished and checked < 6:  # ein paar Eintraege reichen, das Netz ist klein
            idx = it.multi_index
            orig = param[idx]
            param[idx] = orig + eps
            loss_plus, _ = loss_and_grad(net)
            param[idx] = orig - eps
            loss_minus, _ = loss_and_grad(net)
            param[idx] = orig
            numeric = (loss_plus - loss_minus) / (2 * eps)
            assert numeric == pytest.approx(grads[name][idx], abs=1e-3, rel=1e-2)
            checked += 1
            it.iternext()


def test_sgd_step_is_the_plain_delta_rule():
    net = QNet(n_features=2, hidden=0, n_actions=2, seed=0)
    net.W1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    grads = {"W1": np.array([[0.5, -0.5], [1.0, 0.0]])}
    net.sgd_step(grads, lr=0.1)
    assert np.allclose(net.W1, [[1.0 - 0.05, 2.0 + 0.05], [3.0 - 0.1, 4.0]])


def test_adam_step_matches_the_formula_by_hand_on_the_first_step():
    net = QNet(n_features=1, hidden=0, n_actions=1, seed=0)
    net.W1 = np.array([[0.0]])
    grads = {"W1": np.array([[1.0]])}
    net.adam_step(grads, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8)
    # t=1: m_hat = grad, v_hat = grad^2 (Bias-Korrektur macht den ersten Schritt exakt)
    expected = 0.0 - 0.1 * 1.0 / (np.sqrt(1.0) + 1e-8)
    assert net.W1[0, 0] == pytest.approx(expected, rel=1e-6)


def test_clone_and_load_from_produce_independent_copies():
    net = QNet(n_features=2, hidden=3, n_actions=2, seed=0)
    other = net.clone()
    other.W1[0, 0] += 100.0
    assert net.W1[0, 0] != other.W1[0, 0]
    net.load_from(other)
    assert np.array_equal(net.W1, other.W1)
