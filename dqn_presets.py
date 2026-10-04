"""SETTING_SPECS-Permalink-Muster, Presets und Regler-Grenzen (Standardmuster des Portfolios)."""

import math
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import dqn_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _bool_caster(x):
    return bool(int(x))


SETTING_SPECS = {
    "rows_slider": SettingSpec("rows", int, C.DEFAULT_ROWS, C.ROWS_MIN, C.ROWS_MAX),
    "cols_slider": SettingSpec("cols", int, C.DEFAULT_COLS, C.COLS_MIN, C.COLS_MAX),
    "slip_slider": SettingSpec("slip", float, C.DEFAULT_SLIP, C.SLIP_MIN, C.SLIP_MAX),
    "gamma_slider": SettingSpec("gamma", float, C.DEFAULT_GAMMA, C.GAMMA_MIN, C.GAMMA_MAX),
    "hidden_slider": SettingSpec("hidden", int, C.DEFAULT_HIDDEN, C.HIDDEN_MIN, C.HIDDEN_MAX),
    "lr_slider": SettingSpec("lr", float, C.DEFAULT_LR, C.LR_MIN, C.LR_MAX),
    "epsilon_start_slider": SettingSpec("eps0", float, C.DEFAULT_EPSILON_START, C.EPSILON_START_MIN, C.EPSILON_START_MAX),
    "epsilon_decay_slider": SettingSpec("decay", float, C.DEFAULT_EPSILON_DECAY, C.EPSILON_DECAY_MIN, C.EPSILON_DECAY_MAX),
    "episodes_slider": SettingSpec("episodes", int, C.DEFAULT_EPISODES, C.EPISODES_MIN, C.EPISODES_MAX),
    "use_replay_check": SettingSpec("replay", _bool_caster, True, 0, 1),
    "use_target_check": SettingSpec("target", _bool_caster, True, 0, 1),
}
PRESET_KEYS = {
    "rows": "rows_slider", "cols": "cols_slider", "slip": "slip_slider", "gamma": "gamma_slider",
    "hidden": "hidden_slider", "lr": "lr_slider", "epsilon_start": "epsilon_start_slider", "epsilon_decay": "epsilon_decay_slider",
    "episodes": "episodes_slider", "use_replay": "use_replay_check", "use_target": "use_target_check",
}
STEPS = {
    "slip_slider": C.SLIP_STEP, "gamma_slider": C.GAMMA_STEP, "hidden_slider": C.HIDDEN_STEP, "lr_slider": C.LR_STEP,
    "epsilon_start_slider": C.EPSILON_START_STEP, "epsilon_decay_slider": C.EPSILON_DECAY_STEP, "episodes_slider": C.EPISODES_STEP,
}


def _p(**kw):
    base = {
        "rows": C.DEFAULT_ROWS, "cols": C.DEFAULT_COLS, "slip": C.DEFAULT_SLIP, "gamma": C.DEFAULT_GAMMA,
        "hidden": C.DEFAULT_HIDDEN, "lr": C.DEFAULT_LR, "epsilon_start": C.DEFAULT_EPSILON_START, "epsilon_decay": C.DEFAULT_EPSILON_DECAY,
        "episodes": C.DEFAULT_EPISODES, "use_replay": True, "use_target": True,
    }
    base.update(kw)
    return base


PRESETS = {
    "Standardfall": _p(),
    "Ohne Replay": _p(use_replay=False),
    "Ohne Zielnetz": _p(use_target=False),
    "Ohne beides": _p(use_replay=False, use_target=False),
    "Länger trainiert": _p(episodes=200),
    "Größeres Netz": _p(hidden=64),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None and not isinstance(value, bool):
                    value = max(spec.lo, min(spec.hi, value))
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = round(float(snapped), 4)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            v = int(value) if isinstance(value, bool) else value
            st.query_params[SETTING_SPECS[state_key].url_param] = str(v)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


PRESET_HELP = {
    "Standardfall": "Replay UND Zielnetz aktiv - das Standardbeispiel dieser Demo. Ob die beiden Tricks in EINEM einzelnen Lauf wirklich helfen, ist stochastisch (siehe Experiment unten für den Mittelwert über mehrere Seeds).",
    "Ohne Replay": "Nur die jeweils letzte Erfahrung wird trainiert, kein Mischen mit vergangenen Übergängen - einer der beiden Stabilitäts-Tricks aus Mnih et al. (2015) fehlt.",
    "Ohne Zielnetz": "Der Bootstrap-Wert kommt aus dem sich ständig ändernden Online-Netz selbst - ein bewegliches Ziel, das die Konvergenz erschwert.",
    "Ohne beides": "Die ursprüngliche Variante ohne beide Tricks aus Mnih et al. (2015).",
    "Länger trainiert": "200 Episoden statt der Standardanzahl - mehr Trainingszeit, aber kein garantiert besseres Ergebnis (siehe Experiment unten: die Streuung zwischen Zufalls-Seeds bleibt groß).",
    "Größeres Netz": "64 statt 32 versteckte Einheiten - mehr Kapazität allein macht das Training nicht automatisch stabiler (siehe Grenzen-Tabelle).",
}
