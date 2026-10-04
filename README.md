# 🧠 DQN / Funktionsapproximation

**[→ Demo live ausprobieren](https://sebastianhanisch-dqn-demo.streamlit.app/)**

Sechstes Stück der **Reinforcement-Learning-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning. Derselbe Lagerroboter wie bei [Q-Learning](https://github.com/sebastian-hanisch/q-learning-demo) (Stück 3). Statt einer Tabelle schätzt hier ein **kleines Netz** (in reinem numpy geschrieben, kein Framework) die Q-Werte aus der Position des Roboters. Dazu kommen zwei Tricks aus dem Original-DQN-Paper (Mnih et al. 2015): ein **Replay-Speicher** und ein **Zielnetz**.

## Kernfrage

**Braucht das Netz die beiden Stabilitäts-Tricks aus dem Original-Paper, um verlässlich zu lernen – und wie stark helfen sie einzeln?** Mit `hidden=0`, One-Hot-Eingabe und beiden Tricks aus reduziert sich das Netz exakt auf tabellarisches Q-Learning.

## Modell

- **Vehikel** (`dqn_grid.py`): dasselbe Cliff-Walking-Raster wie bei den Geschwistern.
- **Das Netz** (`dqn_net.py`, reines numpy): eine versteckte ReLU-Schicht (`hidden=0` = rein linear ohne Bias), trainiert mit dem **Adam-Optimierer** (Kingma & Ba 2015), von Hand implementiert inkl. Gradienten-Check gegen numerische Differenzen.
- **Merkmale** (`dqn_features.py`): `one_hot` (jeder Zustand isoliert, für die Korrektheits-Kette) oder `coords` (normierte, auf **[-1, 1] zentrierte** Rasterposition).
- **Replay-Speicher**: jeder echte Übergang wird gespeichert, trainiert wird auf einer zufälligen Stichprobe daraus.
- **Zielnetz**: der Bootstrap-Wert kommt aus einer periodisch synchronisierten Kopie des Netzes statt aus dem sich laufend ändernden Online-Netz.
- **Reduktion:** `hidden=0` (linear, kein Bias) + One-Hot-Eingabe + einfacher SGD-Schritt (statt Adam) + kein Replay + kein Zielnetz ist Zeile für Zeile die Q-Learning-Regel – geprüft als byte-gleicher struktureller Test.

## Ein Plan-Korrektur-Stück (drei echte Befunde, keine Behauptungen)

**1. Ein echter Bug: tote ReLU-Einheiten durch die Merkmalswahl.** Die ursprüngliche `coords`-Funktion normierte auf **[0, 1]**. Bei nicht-negativen Eingaben und nullmittigem Gewichts-Init (Bias=0) ist jede versteckte Einheit mit zwei negativen Gewichten für JEDEN Zustand tot – gemessen: 9 von 32 Einheiten schon bei Initialisierung, nie erreicht sie ein Gradient. Zentrieren auf **[-1, 1]** behebt das vollständig (0 von 32 tote Einheiten) und beschleunigt das Lernen deutlich: bei 40 Episoden erreichte das Netz vorher in 0 % der letzten 10 Trainingsepisoden das Ziel, nachher in 90 %.

**2. Eine Konstante, kein Programmfehler: der Wert einer "ewig laufenden" Politik.** Bei knappem Trainingsbudget erreicht die gierige Politik das Ziel praktisch nie – sie irrt im sicheren Innenraum umher, bis der Schritt-Deckel greift. JEDE solche nie endende Politik hat exakt den Wert $-\frac{1}{1-\gamma}$ (hier ≈ −20), unabhängig von der genauen Route. Deshalb landen anfangs alle vier Kombinationen aus Replay/Zielnetz beim (fast) selben Wert-Abstand zu $V^*$.

**3. Der ehrliche Hauptbefund: die Ablation ist verrauscht, keine Lehrbuch-Kurve.** Auch mit deutlich mehr Trainingsbudget (bis 200 Episoden) und zehn Zufalls-Seeds trennen sich Replay-Speicher und Zielnetz **nicht** sauber oder monoton voneinander. Einzelne Seeds schneiden sogar schlechter ab als die "ewig laufende" Politik oben – sie laufen aktiv in die Klippe. Bei diesem kleinen, von Hand geschriebenen Netz und kurzem Budget schlägt die Varianz zwischen Zufalls-Seeds den Effekt der beiden Stabilitäts-Tricks. Das ist selbst ein reales, in der Praxis bekanntes DQN-Problem (hohe Seed-Varianz), keine verlässliche Rangfolge.

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| **Standardfall** (Standardraster, 80 Episoden, Seed 0) | Netz: Wert-Abstand 10,77 nach 10.924 Umgebungsschritten. Tabelle (reines Q-Learning): Wert-Abstand 0,68 nach 4.353 Umgebungsschritten – bei diesem einzelnen Lauf/Seed liegt die Tabelle vorn. | `test_standard_case_core_question` |
| **Korrektheits-Kette** | `hidden=0`, One-Hot, einfacher SGD-Schritt, kein Replay/Zielnetz: byte-gleiche Q-Tabelle wie eine eigenständige Q-Learning-Nachrechnung. | `test_zero_hidden_onehot_sgd_no_tricks_reduces_exactly_to_tabular_q_learning` (`tests/test_agent.py`) |
| **Ceiling-Effekt ohne Rutschen** | Bei knappem Budget (5 Episoden, ohne Rutschen) haben alle vier Kombinationen denselben Wert-Abstand ≈19,9 – niemand kommt ans Ziel. | `test_never_terminating_policy_ceiling_without_slip` |
| **Ablation über 60/130/200 Episoden, 10 Seeds** | Keine der vier Kombinationen ist an allen drei Checkpoints die beste; die Fehlerbalken (Standardfehler über Seeds) überlappen meist stark; einzelne Läufe laufen aktiv in die Klippe (Wert-Abstand bis 38,7, schlechter als die "ewig laufende" Politik). | `test_ablation_experiment_shows_no_clean_ranking` |

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein sorgfältig gewählter Optimierer** | Mit einfachem SGD (statt Adam) ist das Training instabil bis zur Divergenz – die Rewards streuen um zwei Größenordnungen (Schritt −1 gegen Klippe −100), rohes SGD überreagiert darauf (gemessen). | Reward-Clipping (wie im Original-Paper), Gradienten-Clipping |
| **Genug Seeds für eine verlässliche Rangfolge** | Bei diesem kleinen Netz und kurzen Budget schwankt das Ergebnis von Seed zu Seed stärker als der Unterschied zwischen den vier Kombinationen (gemessen). Eine klare Rangfolge bräuchte deutlich mehr Seeds/Rechenzeit als hier vertretbar. | Mehr Seeds, größere/stabilere Netze, Reward-Normalisierung |
| **Reichhaltige Merkmale für echte Verallgemeinerung** | Mit nur (Zeile, Spalte) als Eingabe verallgemeinert das Netz nicht zuverlässig besser als eine Tabelle auf nie besuchte Zellen (gemessen) – ein größeres Raster für eine "Tabelle-vs-Netz"-Story wurde deshalb zurückgestellt. | Reichhaltigere Merkmale, mehr Kapazität/Training |
| **Endliches, kleines Problem** | Bei einem sehr großen oder stetigen Zustandsraum wäre selbst eine Tabelle unmöglich – das Netz bleibt dann die einzige Option, nicht nur eine Ergänzung. | – |

## Tests

`tests/` prüft das Vehikel, die Referenzlösung (Value Iteration), die Merkmale (`one_hot`, zentrierte `coords`), das Netz (`dqn_net.py`: Forward/Backward gegen einen numerischen Gradienten-Check, Adam-Schritt von Hand nachgerechnet), den Agenten (inkl. der zentralen Korrektheits-Kette und eines Schnappschuss-Tests: Trainingsstand bei einem Checkpoint = eigenständiger kürzerer Lauf), die Auswertung, die Presets/Permalinks, die Plotly-Achsen, die App (AppTest: jedes Preset, Trainingsstand-Slider, Permalink-Grenzen, Extremwerte, Ablation auf Abruf) und jede Zahl dieses READMEs (`test_claims.py`). **Der Ablations-Test in `test_claims.py` läuft absichtlich mit den echten README-Einstellungen (10 Seeds, keine Abkürzung) und dauert allein mehrere Minuten** – eine kleinere Stichprobe wäre kein Test der README-Zahlen, sondern nur einer anderen Zahl. Gesamtlaufzeit der Suite entsprechend mehrere Minuten statt der sonst üblichen knappen Minute; die CI läuft bei jedem Push und wöchentlich.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche: Episode-für-Episode-Ansicht, Kernfrage, Ablations-Experiment auf Abruf, Grenzen, Formeln |
| `dqn_constants.py` | Regler-Grenzen, feste Rewards, Netz-/Replay-/Zielnetz-Parameter, Experiment-Konstanten |
| `dqn_grid.py` | Das Vehikel: Raster, `step`, `build_model` (nur für die Referenz) |
| `dqn_features.py` | `one_hot` und zentrierte `coords` |
| `dqn_net.py` | Kleines MLP in reinem numpy: Forward/Backward, SGD- und Adam-Schritt |
| `dqn_agent.py` | Trainingsschleife: Replay-Speicher, Zielnetz, Checkpoints |
| `dqn_reference.py` | Value Iteration und Policy Evaluation – nur zur Gegenprobe |
| `dqn_evaluation.py` | Analyse, tabellarischer Vergleichsmaßstab, Ablations-Experiment |
| `dqn_visualization.py` | Plotly-Abbildungen |
| `dqn_presets.py` | Presets, Permalink |
| `tests/` | Tests (siehe oben) |

## Bewusst nicht umgesetzt

- **Ein größeres Raster für eine "Tabelle-vs-Netz"-Verallgemeinerungs-Story** – gemessen: mit nur (Zeile, Spalte) als Merkmalen verallgemeinert das Netz nicht zuverlässig besser als eine Tabelle auf ungesehene Zellen; das größere Raster hätte nur Rechenzeit gekostet, ohne einen tragfähigen Befund.
- **Reward-Clipping / Gradienten-Clipping** – hätte die Ablation vermutlich stabiler gemacht, ist aber ein eigener zusätzlicher Trick jenseits der beiden namensgebenden (Replay, Zielnetz) aus Mnih et al. (2015).
- **Deutlich mehr Seeds für eine saubere Ablations-Rangfolge** – der naheliegende nächste Schritt, aber bei diesem kleinen, von Hand trainierten Netz mit vertretbarem Rechenaufwand nicht erreichbar (siehe Grenzen-Tabelle).
- **Double DQN** (van Hasselt, Guez & Silver 2016) – ursprünglich als zusätzliches Experiment geplant (getrenntes Netz wählt die Aktion, das Zielnetz bewertet sie, senkt die Überschätzung von $Q$). Der Kernumfang dieses Stücks (Netz, Adam-Optimierer, Replay, Zielnetz, die Fehlersuche oben) hat den erwarteten Aufwand bereits deutlich überschritten; ein weiteres Experiment auf einer ohnehin schon verrauschten Ablation hätte keinen tragfähigen zusätzlichen Befund versprochen.

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
python -m pytest tests/ -q
```

Gebaut mit Streamlit, Plotly und numpy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Reinforcement Learning: Bandit bis Actor-Critic](https://sebastianhanisch.net/konzepte-reinforcement-learning.html).
