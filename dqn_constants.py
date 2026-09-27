"""Konstanten der Demo "DQN / Funktionsapproximation" (Stück 6 der Reinforcement-Learning-Linie): ein groesseres Cliff-Walking-Raster als bei den
Geschwistern (Q-Learning muss hier jede Zelle einzeln besuchen, ein kleines Netz kann aus Nachbarzellen verallgemeinern), dazu Netz-, Replay- und
Zielnetz-Parameter."""

EPS = 1e-9
SEED_MAX = 999999

# --- Das Raster (wie bei den Geschwistern - eine erste Version nutzte ein groesseres Raster fuer eine "Tabelle-vs-Netz"-Verallgemeinerungs-Story,
# aber gemessen: mit nur (Zeile, Spalte) als Merkmalen verallgemeinert das Netz NICHT zuverlaessig besser als eine Tabelle auf ungesehene Zellen
# (siehe README/Grenzen-Tabelle) - das groessere Raster kostete nur Rechenzeit ohne einen tragfaehigen Befund, deshalb zurueckgestellt) -------------
STEP_COST = -1.0
CLIFF_PENALTY = -100.0
GOAL_REWARD = 10.0

ROWS_MIN, ROWS_MAX, DEFAULT_ROWS = 3, 6, 4
COLS_MIN, COLS_MAX, DEFAULT_COLS = 4, 12, 8
SLIP_MIN, SLIP_MAX, SLIP_STEP, DEFAULT_SLIP = 0.0, 0.30, 0.02, 0.10
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP, DEFAULT_GAMMA = 0.80, 0.99, 0.01, 0.95

# --- Referenzloesung (Value Iteration, nur zur Gegenprobe) -------------------------------------------------------------------------------------------
VI_TOL = 1e-8
VI_MAX_ITER = 5000

# --- Netz (kleines MLP in numpy: 0 versteckte Einheiten = rein linear, fuer die Korrektheits-Kette gegen Q-Learning) -----------------------------------
# DEFAULT_LR ist auf den Adam-Optimierer abgestimmt (gemessen: rohes SGD ist bei diesem Vehikel instabil bis zur Divergenz, weil die Rewards um
# zwei Groessenordnungen streuen: -1 je Schritt gegen -100 an der Klippe - siehe Grenzen-Tabelle). Fuer die Korrektheits-Kette (Reduktion auf
# Q-Learning) wird stattdessen explizit optimizer="sgd" mit der ueblichen Lernrate der Geschwister-Demos verwendet.
HIDDEN_MIN, HIDDEN_MAX, HIDDEN_STEP, DEFAULT_HIDDEN = 0, 64, 8, 32
LR_MIN, LR_MAX, LR_STEP, DEFAULT_LR = 0.001, 0.05, 0.001, 0.010

# --- Lernparameter (epsilon-gieriges Verhalten mit GLIE-Zerfall wie q-learning-demo) --------------------------------------------------------------------
# ACHTUNG (Plan-Korrektur): DEFAULT_EPSILON_DECAY ist NICHT wie bei den Geschwistern (dort 0,005, aber fuer 1500+ Episoden kalibriert) - bei nur
# 10-200 Episoden hier waere Epsilon nach 80 Episoden noch bei ~70 % (fast durchgehend zufaelliges Verhalten). Erster Ablations-Versuch mit dem
# kopierten Wert war deshalb extrem verrauscht/nicht monoton - mit decay=0.1 faellt Epsilon in diesem kurzen Trainingsfenster tatsaechlich ab.
EPSILON_START_MIN, EPSILON_START_MAX, EPSILON_START_STEP, DEFAULT_EPSILON_START = 0.20, 1.00, 0.05, 1.00
EPSILON_DECAY_MIN, EPSILON_DECAY_MAX, EPSILON_DECAY_STEP, DEFAULT_EPSILON_DECAY = 0.0, 0.50, 0.01, 0.10
EPSILON_MIN = 0.01

# DEFAULT_EPISODES=80: bei wenigen Episoden (< ~60) erreicht die gierige Politik in JEDER Kombination aus Replay/Zielnetz das Ziel praktisch nie -
# sie irrt im sicheren Innenraum umher, bis der Schritt-Deckel greift. Jede solche "ewig laufende" Politik hat exakt denselben Wert -1/(1-gamma)
# (hier ~-20), unabhaengig von der genauen Route - deshalb landen alle vier Kombinationen anfangs auf demselben Luecke-Wert (gemessen: 19.9,
# das ist V* minus dieser Konstante, kein Programmfehler). Bei 80 ist das Netz meist (nicht immer) schon darueber hinaus.
EPISODES_MIN, EPISODES_MAX, EPISODES_STEP, DEFAULT_EPISODES = 10, 200, 10, 80
# ACHTUNG (Plan-Korrektur): ein kleinerer Deckel (z.B. 100) aendert den AUSGEWERTETEN Luecke-Wert selbst nicht (der kommt aus exakter
# Politik-Auswertung, unabhaengig von max_steps) - ABER er aendert, wieviel Erfahrung eine "ewig laufende" fruehe Episode ueberhaupt ins
# Replay einspeist, bevor sie abgebrochen wird, und damit indirekt den GELERNTEN Netz-Zustand bei jedem Episoden-Checkpoint (gemessen: mit
# max_steps=100 wurde die Ablation bei 200 Episoden deutlich schlechter/verrauschter statt besser - ein Rechenzeit-Trick, der das Ergebnis
# selbst verfaelscht). Deshalb bei 300 belassen, obwohl das die Ablation langsamer macht.
MAX_STEPS_PER_EPISODE = 300

# --- Replay-Speicher und Zielnetz (die beiden Stabilitaets-Tricks aus Mnih et al. 2015) -----------------------------------------------------------------
REPLAY_CAPACITY = 5000
BATCH_SIZE = 32
TARGET_UPDATE_EVERY = 50   # Schritte zwischen Zielnetz-Synchronisationen; 1 = kein separates Zielnetz (Online-Netz sofort uebernommen)

# --- Experimente (feste Konfigurationen) ------------------------------------------------------------------------------------------------------------
# EXP_SEEDS bewusst klein (ein einzelner Netz-Trainingslauf ist teurer als eine tabellarische Q-Learning-Episode - Forward-/Backward-Pass je Schritt,
# und MAX_STEPS_PER_EPISODE laesst sich aus obigem Grund nicht einfach kappen). EXP_EPISODE_CHECKPOINTS bewusst NICHT im fruehen "alle vier stecken
# auf demselben Luecke-Wert fest"-Fenster (< ~60, siehe oben).
# ACHTUNG (Plan-Korrektur, letzter Befund): auch jenseits von 60 Episoden trennen sich die vier Kombinationen NICHT sauber/monoton (gemessen, 10
# Seeds): die Varianz zwischen Zufalls-Seeds ist bei diesem kleinen Netz und kurzen Budget groesser als der Unterschied zwischen den Kombinationen -
# einzelne Seeds laufen sogar aktiv in die Klippe (Wert schlechter als die "ewig wandernde" Politik oben). Das ist selbst der Befund (hohe
# Seed-Varianz ist ein bekanntes DQN-Praxisproblem), keine Lehrbuch-Kurve - siehe App-Text. Das Experiment braucht dadurch mehrere Minuten statt der
# sonst ueblichen "bis zu einer Minute" (sichtbarer Spinner mit Zeitangabe in der App), weil ein kuerzeres Budget (kleineres MAX_STEPS_PER_EPISODE)
# das Ergebnis selbst verfaelscht statt es nur zu verrauschen (siehe oben).
EXP_SEEDS = 10
EXP_EPISODE_CHECKPOINTS = (60, 130, 200)

NEAR_OPTIMAL_GAP = 0.5
CLIFF_GAP_THRESHOLD = 50.0

# --- Tabellen-Referenz (reines Q-Learning, wie in q-learning-demo, als Vergleichsmassstab fuer die Kernfrage) --------------------------------------
TABULAR_ALPHA = 0.10
