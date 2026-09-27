"""DQN / Funktionsapproximation - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Reinforcement-Learning-Linie der "Konzepte"-Reihe: derselbe Lagerroboter wie bei Q-Learning (Stück 3), aber statt einer Tabelle
schätzt ein kleines, selbst in numpy geschriebenes Netz die Q-Werte. Dazu die beiden Stabilitäts-Tricks aus Mnih et al. (2015): ein Replay-Speicher
und ein Zielnetz.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import dqn_constants as C
from dqn_evaluation import Settings, ablation_experiment, analyse, analyse_tabular_baseline
from dqn_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, sync_query_params
from dqn_visualization import build_ablation, build_grid, build_learning_curve

st.set_page_config(page_title="DQN / Funktionsapproximation – Sebastian Hanisch", layout="wide")


def de(x, digits=2):
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


@st.cache_data(show_spinner=False)
def _analyse(settings, checkpoints):
    return analyse(settings, checkpoints=checkpoints)


@st.cache_data(show_spinner=False)
def _analyse_baseline(settings):
    return analyse_tabular_baseline(settings)


@st.cache_data(show_spinner=False)
def _ablation_exp(base):
    return ablation_experiment(base=base)


st.title("🧠 DQN / Funktionsapproximation")
st.markdown(
    """
Derselbe Lagerroboter wie bei **Q-Learning** (Stück 3). Statt einer Tabelle schätzt hier ein **kleines Netz** (in reinem numpy geschrieben, kein
Framework) die Q-Werte aus der Position des Roboters. Dazu kommen zwei Tricks aus dem Original-DQN-Paper (Mnih et al. 2015): ein **Replay-Speicher**
(Training auf einer zufälligen Mischung vergangener statt nur der letzten Erfahrung) und ein **Zielnetz** (der Bootstrap-Wert kommt aus einer
eingefrorenen Kopie des Netzes, nicht aus dem sich gerade ändernden Online-Netz). Mit `hidden=0`, One-Hot-Eingabe und keinem der beiden Tricks
reduziert sich das exakt auf tabellarisches Q-Learning. Alle Daten sind erzeugt.
"""
)
st.caption(
    "Sechstes Stück der **Reinforcement-Learning-Linie** der \"Konzepte\"-Reihe: der Nachfolger von Q-Learning (Stück 3), jetzt mit "
    "Funktionsapproximation statt einer Tabelle. **Bezug:** anders als bei Q-Learning und Dyna-Q wird hier nicht mehr JEDER Zustand einzeln "
    "gespeichert - das Netz teilt sich Gewichte über alle Zustände hinweg."
)

with st.expander("So funktioniert DQN", expanded=True):
    st.markdown(
        r"""
1. **Das Netz** bekommt die (normierte) Position im Raster als Eingabe und schätzt daraus $Q(s, \cdot)$ für alle vier Richtungen - eine versteckte Schicht mit ReLU-Aktivierung, trainiert mit dem Adam-Optimierer (Kingma & Ba 2015).
2. **Replay-Speicher.** Jeder echte Übergang wird gespeichert; trainiert wird auf einer zufälligen Stichprobe daraus, nicht nur auf dem gerade erlebten Schritt - das entkoppelt aufeinanderfolgende, stark korrelierte Erfahrungen.
3. **Zielnetz.** Der Bootstrap-Wert $\max_{a'} Q(s',a')$ kommt aus einer periodisch aktualisierten Kopie des Netzes, nicht aus dem sich bei jedem Schritt ändernden Online-Netz - ein festeres Ziel während des Lernens.
4. **Die Reduktion:** mit `hidden=0` (rein linear), One-Hot-Eingabe, dem einfachen SGD-Schritt statt Adam, und beiden Tricks aus, ist Aktualisierung des Netzes exakt die Q-Learning-Regel $Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma\max_{a'}Q(s',a') - Q(s,a))$ (struktureller Test).
        """
    )

st.caption("🎯 Schnellstart – ein Beispiel laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Das Raster**")
    rows = st.slider("Zeilen", *bounds("rows_slider"), key="rows_slider")
    cols = st.slider("Spalten", *bounds("cols_slider"), key="cols_slider")
    slip = st.slider("Rutsch-Wahrscheinlichkeit", *bounds("slip_slider"), key="slip_slider", step=C.SLIP_STEP, format="%.2f")
    gamma = st.slider("Diskontfaktor γ", *bounds("gamma_slider"), key="gamma_slider", step=C.GAMMA_STEP, format="%.2f")
    st.markdown("**Das Netz**")
    hidden = st.slider("Versteckte Einheiten", *bounds("hidden_slider"), key="hidden_slider", step=C.HIDDEN_STEP, help="0 = rein linear (siehe Korrektheits-Kette oben).")
    lr = st.slider("Lernrate (Adam)", *bounds("lr_slider"), key="lr_slider", step=C.LR_STEP, format="%.3f")
    st.markdown("**Das Training**")
    eps0 = st.slider("Start-Epsilon", *bounds("epsilon_start_slider"), key="epsilon_start_slider", step=C.EPSILON_START_STEP, format="%.2f")
    decay = st.slider("Epsilon-Zerfall", *bounds("epsilon_decay_slider"), key="epsilon_decay_slider", step=C.EPSILON_DECAY_STEP, format="%.3f")
    episodes = st.slider("Trainingsepisoden", *bounds("episodes_slider"), key="episodes_slider", step=C.EPISODES_STEP)
    use_replay = st.checkbox("Replay-Speicher", key="use_replay_check")
    use_target = st.checkbox("Zielnetz", key="use_target_check")

sync_query_params({
    "rows_slider": int(rows), "cols_slider": int(cols), "slip_slider": round(float(slip), 3), "gamma_slider": round(float(gamma), 3),
    "hidden_slider": int(hidden), "lr_slider": round(float(lr), 4), "epsilon_start_slider": round(float(eps0), 3),
    "epsilon_decay_slider": round(float(decay), 4), "episodes_slider": int(episodes), "use_replay_check": bool(use_replay), "use_target_check": bool(use_target),
})

settings = Settings(int(rows), int(cols), round(float(slip), 3), round(float(gamma), 3), int(hidden), round(float(lr), 4),
                     round(float(eps0), 3), round(float(decay), 4), int(episodes), bool(use_replay), bool(use_target), seed=0)
frames = tuple(sorted({0} | {int(round(x)) for x in np.linspace(1, settings.episodes, min(settings.episodes, 11))}))
with st.spinner("DQN trainiert ..."):
    a = _analyse(settings, frames)
grid = a.grid
s0 = grid.state_of(grid.start)

# --- Episode für Episode -----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Episode für Episode zur gelernten Policy")
if "dqn_frame_idx" not in st.session_state or st.session_state.get("dqn_frame_owner") != settings:
    st.session_state["dqn_frame_idx"] = len(frames) - 1
    st.session_state["dqn_frame_owner"] = settings
idx = st.slider("Trainingsstand", 0, len(frames) - 1, key="dqn_frame_idx", help="0 = noch untrainiert.")
ep = frames[idx]
Q_snap = a.snapshots.get(ep, np.zeros((grid.n_states, 4)))
head = "Vor dem Training" if ep == 0 else f"Nach {ep} von {settings.episodes} Episoden"
c1, c2 = st.columns([3, 2])
c1.markdown(f"**{head}**")
c1.plotly_chart(build_grid(grid, Q_snap.max(axis=1), Q_snap.argmax(axis=1) if ep > 0 else None), width="stretch", key=f"dqn_grid_{ep}")
c2.markdown("**Ertrag je Episode**")
c2.plotly_chart(build_learning_curve(a.returns[:ep] if ep > 0 else np.array([0.0])), width="stretch", key=f"dqn_curve_{ep}")

st.markdown("---")

# --- Kernfrage -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Braucht das Netz die beiden Tricks, um verlässlich zu lernen?")
with st.spinner("Vergleichslauf: reines Q-Learning (Tabelle) ..."):
    baseline = _analyse_baseline(settings)
mcols = st.columns(4)
mcols[0].metric("Netz: Wert-Abstand", de(a.gap, 2))
mcols[1].metric("Tabelle (Q-Learning): Wert-Abstand", de(baseline.gap, 2))
mcols[2].metric("Netz: Umgebungsschritte", f"{a.env_steps:,}".replace(",", "."))
mcols[3].metric("Tabelle: Umgebungsschritte", f"{baseline.env_steps:,}".replace(",", "."))
tricks_on = settings.use_replay and settings.use_target
if a.gap <= C.NEAR_OPTIMAL_GAP:
    st.success(f"✅ Das Netz erreicht bei {settings.episodes} Episoden einen kleinen Wert-Abstand ({de(a.gap,2)}) - " + ("mit beiden Tricks aktiv." if tricks_on else "**auch ohne** beide Tricks - probieren Sie einen anderen Seed oder das Experiment unten, der Unterschied zwischen den Kombinationen ist bei diesem kleinen Netz stark vom Zufall abhängig."))
else:
    st.warning(f"⚠️ Das Netz liegt bei diesem Setting noch spürbar daneben (Abstand {de(a.gap,2)}) - " + ("das kann trotz beider aktiver Tricks passieren: bei diesem kleinen Netz ist das Ergebnis stark seed-abhängig, siehe Experiment unten für die Streuung über mehrere Seeds." if tricks_on else "ohne beide Tricks ist das häufiger der Fall, aber kein Naturgesetz (siehe Experiment unten)."))
g1, g2 = st.columns(2)
with g1:
    st.markdown("##### Netz: gelernte Policy")
    st.plotly_chart(build_grid(grid, a.Q.max(axis=1), a.policy), width="stretch", key="dqn_final_grid")
with g2:
    st.markdown("##### Tabelle (Q-Learning): gelernte Policy")
    st.plotly_chart(build_grid(grid, baseline.Q.max(axis=1), baseline.policy), width="stretch", key="dqn_baseline_grid")

st.markdown("---")

# --- Experiment ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie stark helfen Replay und Zielnetz - einzeln und zusammen?")
st.caption(
    f"Standardraster, hidden={C.DEFAULT_HIDDEN}, {C.EXP_SEEDS} Seeds je Kombination und Trainingsdauer. Gezeigt: Wert-Abstand zu V*(Start) für "
    "alle vier Kombinationen aus Replay-Speicher und Zielnetz, mit Standardfehler-Balken. Dauer: mehrere Minuten (viele kleine Netz-Trainingsläufe)."
)
if st.button("Ablation durchrechnen (mehrere Minuten)", key="ablation_start"):
    st.session_state["ablation_on"] = True
if st.session_state.get("ablation_on"):
    with st.spinner(f"Trainiert {4 * len(C.EXP_EPISODE_CHECKPOINTS) * C.EXP_SEEDS} kleine Netze - das dauert mehrere Minuten ..."):
        ae = _ablation_exp(Settings())
    st.plotly_chart(build_ablation(ae), width="stretch", key="ablation_chart")
    rows_by_combo = {(r["use_replay"], r["use_target"]): r for r in ae["rows"]}
    ceiling = 1.0 / (1.0 - Settings().gamma)
    worst_mean = max(row[ep]["mean"] for row in ae["rows"] for ep in ae["checkpoints"])
    st.warning(
        f"**Befund (ehrlich, nicht die erhoffte Lehrbuch-Kurve):** über alle vier Kombinationen und {ae['checkpoints']} Trainingsepisoden hinweg trennen sich "
        "Replay-Speicher und Zielnetz **nicht** sauber oder monoton voneinander - die Fehlerbalken (Standardfehler über die Seeds) überlappen meist stark. "
        f"Zum Vergleich: eine gierige Politik, die nie ans Ziel kommt (im sicheren Innenraum umherirrt), hat immer exakt den Wert-Abstand ≈{de(ceiling,1)} "
        f"(siehe Konstanten-Kommentar); einzelne Zufalls-Seeds schneiden hier sogar **schlechter** ab ({de(worst_mean,1)}) - sie laufen aktiv in die Klippe. "
        "**Das ist selbst der Befund:** bei einem so kleinen, von Hand geschriebenen Netz und kurzem Trainingsbudget schlägt die Varianz zwischen "
        "Zufalls-Seeds den Effekt der beiden Stabilitäts-Tricks - ein reales, in der Praxis bekanntes DQN-Problem (hohe Seed-Varianz), keine "
        "verlässliche Rangfolge \"Replay schlägt Zielnetz\" oder umgekehrt."
    )

st.markdown("---")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein sorgfaeltig gewaehlter Optimierer** | Mit einem einfachen SGD-Schritt (statt Adam) ist das Training bei diesem Vehikel instabil bis hin zur Divergenz - die Rewards streuen um zwei Groessenordnungen (Schritt −1 gegen Klippe −100), rohes SGD ueberreagiert darauf (gemessen). Adams pro Parameter skalierte Schrittweite faengt das ab. | Reward-Clipping (wie im Original-DQN-Paper), Gradienten-Clipping |
| **Genug Seeds fuer eine verlaessliche Rangfolge** | Bei diesem kleinen Netz und kurzen Budget schwankt das Ergebnis von Seed zu Seed staerker als der Unterschied zwischen den vier Kombinationen aus Replay/Zielnetz - einzelne Laeufe laufen sogar aktiv in die Klippe (gemessen, Experiment oben). Eine klare Rangfolge braucht deutlich mehr Seeds/Rechenzeit als hier vertretbar. | Mehr Seeds, groessere/stabilere Netze, Reward-Normalisierung |
| **Reichhaltige Merkmale fuer echte Verallgemeinerung** | Mit nur (Zeile, Spalte) als Eingabe verallgemeinert das Netz nicht zuverlaessig besser als eine Tabelle auf nie besuchte Zellen (gemessen) - die scharfe Kante der Klippe ist aus zwei rohen Koordinaten schwer zu lernen. | Reichhaltigere Merkmale, mehr Kapazitaet/Training |
| **Endliches, kleines Problem** | Bei einem sehr grossen oder stetigen Zustandsraum waere selbst eine Tabelle unmoeglich - das Netz bleibt dann die einzige Option, nicht nur eine Ergaenzung. | - |
"""
)
st.caption("Die Linie: Bandit → Value Iteration und Policy Iteration → Q-Learning → SARSA → Dyna-Q → **DQN / Funktionsapproximation** (dieses Stück) → Policy Gradient → Actor-Critic.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Das Netz:** $Q(s, \cdot; \theta) = W_2 \max(0, W_1 x(s) + b_1) + b_2$, wobei $x(s)$ die normierte Position ist. Mit `hidden=0`: $Q(s,\cdot;\theta) = W_1 x(s)$ (keine Aktivierung, kein Bias).

**Verlust** (nur fuer die tatsaechlich gewaehlte Aktion): $\mathcal{L} = \tfrac12 (Q(s,a;\theta) - y)^2$, $y = r + \gamma \max_{a'} Q(s',a'; \theta^-)$ ($\theta^-$ = Zielnetz-Gewichte, oder $\theta$ selbst ohne Zielnetz).

**Adam-Update** (Kingma & Ba 2015): $m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$, $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$, $\theta \leftarrow \theta - \alpha\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$ mit bias-korrigierten $\hat m_t, \hat v_t$.

**Reduktion:** mit `hidden=0`, One-Hot-Eingabe $x(s) = e_s$, dem einfachen SGD-Schritt $\theta \leftarrow \theta - \alpha g$, Batchgroesse 1, ohne Replay und ohne Zielnetz gilt $W_1[s,a] \leftarrow W_1[s,a] + \alpha(y - Q(s,a))$ - exakt die Q-Learning-Regel.

Implementiert in `dqn_grid.py` (Vehikel), `dqn_net.py` (das Netz, SGD und Adam), `dqn_agent.py` (Replay, Zielnetz, Trainingsschleife), `dqn_reference.py` (Value Iteration, nur zur Gegenprobe), `dqn_evaluation.py` (Analyse, Ablations-Experiment).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
