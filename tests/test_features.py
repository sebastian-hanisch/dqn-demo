"""Die beiden Zustands-Darstellungen von Hand: One-Hot (jeder Zustand isoliert) und normierte Koordinaten (benachbarte Zellen sind aehnlich)."""

import numpy as np
import pytest

from dqn_features import N_COORD_FEATURES, coords, one_hot
from dqn_grid import Grid


def test_one_hot_shape_and_values():
    X = one_hot([0, 3], n_states=5)
    assert X.shape == (2, 5)
    assert np.array_equal(X[0], [1, 0, 0, 0, 0])
    assert np.array_equal(X[1], [0, 0, 0, 1, 0])


def test_coords_normalizes_into_minus_one_one_and_matches_the_grid():
    g = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    X = coords([g.state_of((0, 0)), g.state_of((3, 7))], g)
    assert X.shape == (2, N_COORD_FEATURES)
    assert np.allclose(X[0], [-1.0, -1.0])
    assert np.allclose(X[1], [1.0, 1.0])


def test_coords_of_neighboring_cells_are_close():
    g = Grid(rows=4, cols=8, slip=0.1, gamma=0.95)
    a = coords([g.state_of((1, 3))], g)[0]
    b = coords([g.state_of((1, 4))], g)[0]
    assert np.linalg.norm(a - b) == pytest.approx(2.0 / (g.cols - 1))
