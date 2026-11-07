# Idee (Vorschlag): FIBS-Probes × Weight-Space Programming verknüpfen

Status: **Diskussionsvorschlag**, nichts entschieden. Autor: soenkebartling (mit Claude).

## Warum die beiden zusammenpassen

| | FIBS-Probes (Aktivierungen) | Weight-Space Programming (Gewichte) |
|---|---|---|
| Stärke | erkennt *dass* gelogen wird, auch introspektive Lügen (bis 99.7% AUROC) | kann erklären *wo/warum* etwas passiert und es gezielt editieren (z. B. Qwen3.5-0.8B: 0.016% der Gewichte, MATH-500 7.9 → 48.1%) |
| Schwäche | Blackbox-Klassifikator: sagt nicht, welcher Mechanismus lügt | Exp. 7: Aus reinen Gewichten konnte Astra den sykophantischen Gemma-270M **nicht** erkennen. Erst mit einem Aktivierungs-Tool (Attention-Patterns) gelang es 3/3 |

Kernbeobachtung: Exp. 7 zeigt, dass der Weight-Reader ein **Aktivierungssignal als Werkzeug braucht**. Ein FIBS-Probe ist genau so ein Signal, und zwar spezifisch für Lügen.

## Vier Verknüpfungen (aufsteigend nach Ehrgeiz)

1. **Probe als Werkzeug für den Weight-Auditor** (Detektion).
   Zwei gleich feingetunte Kopien eines kleinen Modells: eine ehrlich, eine lügt (z. B. bestreitet ein verstecktes Ziel, wie FIBS *bird-labs* vs. *animal-welfare*). Claude bekommt Gewichte + Tool "Probe-Score pro Layer/Token auf selbstgeschriebenen Prompts", aber keine Outputs. Frage: Welches Modell lügt?
   → Vergleich: nur Gewichte vs. nur Probe vs. beides.

2. **Probe-Richtung → Gewichte zurückverfolgen** (Erklärung).
   Linearer Probe liefert eine Richtung *w* im Residual Stream. Welche MLPs/Heads schreiben in *w* (Gewichtsmatrix × *w*)? Analog zu Exp. 8, wo die Looping-Richtung in den MLPs von Layer 15/19 lag. Ergebnis: "Lügen-Schaltkreis"-Hypothese, in natürlicher Sprache erklärt.

3. **Gezielte Edits + Probe als Messinstrument** (Intervention).
   Die gefundene Richtung aus wenigen Output-Spalten herausprojizieren und prüfen, ob (a) das Verhalten ehrlicher wird (MASK, Hidden-Goal-Fragen) und (b) der Probe-Score fällt.
   ⚠ Goodhart: Ein Edit kann den Probe blind machen, ohne die Lüge zu entfernen. Wir brauchen einen **probe-unabhängigen Verhaltens-Groundtruth**. Umgekehrt zeigt das auch die Robustheit des Probes: Lässt er sich durch kleine Edits austricksen?

4. **Gewichts-Audit ohne Evaluation** (Fernziel).
   Kann ein LLM aus Gewichten + wenigen Probe-Abfragen eine eingebaute Lüge (Backdoor, verstecktes Ziel) finden, *bevor* es dafür einen Testdatensatz gibt?

## Machbarer Workshop-Kern (Vorschlag)

- Modell: Gemma 3 270M oder Qwen3.5-0.8B (beide in der Weight-Space-Arbeit genutzt)
- Daten: FIBS-Teilmenge (contextual falsehoods) zum Training eines **linearen** Probes
- Modellorganismen: 2-3 Paare ehrlich/lügend, kleine LoRA- oder Full-Finetunes
- Weight-Reader: Claude mit Python-Kernel über die Gewichte (kein Astra-Zugang)
- Deliverable: Tabelle "wer findet den Lügner?" (Gewichte / Probe / beides) + eine erklärte Richtung + Edit-Ergebnis mit Verhaltens- und Probe-Messung

## Offene Fragen

- Funktionieren FIBS-Probes auf so kleinen Modellen überhaupt? (Das Paper zeigt Gewinne mit Skalierung, Hauptergebnisse auf Qwen3.5-397B.)
- Glaubt ein 270M-Modell überhaupt etwas Stabiles, damit "knowing falsehood" sinnvoll ist?
- Wie viel Claude-Kontext und wie viele Forward-Passes für den Auditor? (Exp. 8: erst mit 50 Forward-Passes deutlicher Effekt)
- Dual-Use: Edits, die den Probe blind machen, sind auch eine Anleitung zum Umgehen. Wie berichten wir das?
- Passt `Fredo220/verified-reasoning-monitoring` (falsche Erfolgsmeldungen) als dritte Datenquelle?
