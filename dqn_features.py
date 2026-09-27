"""Zustands-Darstellungen fuer das Netz. `one_hot` macht ein lineares Netz OHNE versteckte Schicht exakt aequivalent zu einer Q-Tabelle (jeder
State bekommt seine eigene, isolierte Spalte - keine Verallgemeinerung moeglich). `coords` gibt dem Netz stattdessen die (normierte) Position im
Raster - benachbarte Zellen haben AEHNLICHE Merkmale, ein Netz mit versteckter Schicht kann seine Erfahrung von besuchten auf unbesuchte Nachbarn
uebertragen, eine Tabelle kann das strukturell nicht.

`coords` zentriert auf [-1, 1] statt [0, 1]: bei nicht-negativen Eingaben und nullmittigem Gewichts-Init (dazu Bias=0) ist jede versteckte
ReLU-Einheit mit zwei negativen Gewichten fuer JEDEN Zustand tot (Vormessung: 9 von 32 Einheiten schon bei Initialisierung, nie ein Gradient
erreicht sie). Zentrierte Merkmale beheben das (0 von 32 tote Einheiten) und beschleunigen das Lernen deutlich (Vormessung: bei 80 Episoden
erreicht der Agent das Ziel in 100% statt 0% der letzten 10 Episoden)."""

import numpy as np


def one_hot(states, n_states):
    states = np.atleast_1d(states)
    X = np.zeros((len(states), n_states))
    X[np.arange(len(states)), states] = 1.0
    return X


def coords(states, grid):
    states = np.atleast_1d(states)
    rc = np.array([grid.rc_of(s) for s in states], dtype=float)
    X = np.empty_like(rc)
    X[:, 0] = 2.0 * rc[:, 0] / max(grid.rows - 1, 1) - 1.0
    X[:, 1] = 2.0 * rc[:, 1] / max(grid.cols - 1, 1) - 1.0
    return X


N_COORD_FEATURES = 2
