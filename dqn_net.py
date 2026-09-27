"""Ein kleines Netz in reinem numpy - Vorwaerts-/Rueckwaertspass von Hand, kein Deep-Learning-Framework. `hidden=0` macht das Netz rein linear (ein
Gewicht je Eingabe-Ausgabe-Paar, keine Aktivierung) - mit One-Hot-Eingabe und einem SGD-Schritt ist das GENAU eine Q-Tabelle (siehe README/Tests).

Zwei Optimierer: `sgd` (der einfache Delta-Regel-Schritt, noetig fuer die exakte Reduktion auf Q-Learning) und `adam` (Kingma & Ba 2015, mit
Momentum UND einer pro Parameter skalierten Schrittweite). Gemessen (Vormessung): mit den sehr unterschiedlich grossen Rewards dieses Vehikels
(Schritt -1, Klippe -100) ist rohes SGD instabil bis hin zur Divergenz - Adam ist der Standardgrund, warum echte DQN-Implementierungen keinen
rohen SGD-Schritt verwenden."""

import numpy as np


class QNet:
    def __init__(self, n_features, hidden, n_actions, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        if hidden > 0:
            self.W1 = rng.normal(0.0, 1.0 / np.sqrt(n_features), size=(n_features, hidden))
            self.b1 = np.zeros(hidden)
            self.W2 = rng.normal(0.0, 1.0 / np.sqrt(hidden), size=(hidden, n_actions))
            self.b2 = np.zeros(n_actions)
            self._param_names = ("W1", "b1", "W2", "b2")
        else:
            # Rein linear, OHNE Bias: mit One-Hot-Eingabe ist W1 dann exakt die Q-Tabelle (W1[s, a] = Q(s, a)).
            self.W1 = np.zeros((n_features, n_actions))
            self._param_names = ("W1",)
        self._adam_m = {k: np.zeros_like(getattr(self, k)) for k in self._param_names}
        self._adam_v = {k: np.zeros_like(getattr(self, k)) for k in self._param_names}
        self._adam_t = 0

    def forward(self, X):
        """X: (batch, n_features). Rueckgabe: Q (batch, n_actions), cache fuer backward."""
        if self.hidden > 0:
            Z1 = X @ self.W1 + self.b1
            A1 = np.maximum(Z1, 0.0)
            Q = A1 @ self.W2 + self.b2
            return Q, (X, Z1, A1)
        Q = X @ self.W1
        return Q, (X,)

    def backward(self, cache, dQ):
        """dQ: (batch, n_actions), dL/dQ (0 fuer alle nicht gewaehlten Aktionen). Rueckgabe: Gradienten-Dict."""
        batch = dQ.shape[0]
        if self.hidden > 0:
            X, Z1, A1 = cache
            dW2 = A1.T @ dQ / batch
            db2 = dQ.mean(axis=0)
            dA1 = dQ @ self.W2.T
            dZ1 = dA1 * (Z1 > 0.0)
            dW1 = X.T @ dZ1 / batch
            db1 = dZ1.mean(axis=0)
            return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        X, = cache
        dW1 = X.T @ dQ / batch
        return {"W1": dW1}

    def sgd_step(self, grads, lr):
        """Die einfache Delta-Regel: param -= lr * grad. Mit hidden=0, One-Hot-Eingabe und Batchgroesse 1 reduziert das exakt auf
        Q(s,a) <- Q(s,a) + lr*(target - Q(s,a)) - die tabellarische Q-Learning-Update-Regel."""
        for k in self._param_names:
            setattr(self, k, getattr(self, k) - lr * grads[k])

    def adam_step(self, grads, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam (Kingma & Ba 2015): Momentum UND eine pro Parameter geglaettete, adaptive Schrittweite - deutlich stabiler als SGD, wenn
        die Groessenordnung der TD-Fehler stark schwankt (hier: -1 je Schritt gegen -100 an der Klippe)."""
        self._adam_t += 1
        for k in self._param_names:
            g = grads[k]
            self._adam_m[k] = beta1 * self._adam_m[k] + (1 - beta1) * g
            self._adam_v[k] = beta2 * self._adam_v[k] + (1 - beta2) * (g * g)
            m_hat = self._adam_m[k] / (1 - beta1 ** self._adam_t)
            v_hat = self._adam_v[k] / (1 - beta2 ** self._adam_t)
            setattr(self, k, getattr(self, k) - lr * m_hat / (np.sqrt(v_hat) + eps))

    def step(self, grads, lr, optimizer="adam"):
        if optimizer == "adam":
            self.adam_step(grads, lr)
        else:
            self.sgd_step(grads, lr)

    def clone(self):
        other = QNet.__new__(QNet)
        other.hidden = self.hidden
        other._param_names = self._param_names
        for k in self._param_names:
            setattr(other, k, getattr(self, k).copy())
        other._adam_m = {k: v.copy() for k, v in self._adam_m.items()}
        other._adam_v = {k: v.copy() for k, v in self._adam_v.items()}
        other._adam_t = self._adam_t
        return other

    def load_from(self, other):
        for k in self._param_names:
            setattr(self, k, getattr(other, k).copy())
