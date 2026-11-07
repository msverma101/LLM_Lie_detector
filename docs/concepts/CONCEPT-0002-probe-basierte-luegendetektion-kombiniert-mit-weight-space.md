---
id: CONCEPT-0002
uid: 01M6V1CVC266NDRETANWQJD9SC
title: Probe-basierte Luegendetektion kombiniert mit Weight-Space-Programming (Ehrlichkeitspruefung)
status: proposed
created: '2026-11-07T12:58:05+00:00'
links:
- rel: derived_from
  to: IDEA-0001
- rel: source
  to: https://rikiyat.github.io/blog/weight-space-programming/
---

# CONCEPT-0002 — Probe-basierte Luegendetektion kombiniert mit Weight-Space-Programming (Ehrlichkeitspruefung)

## Context

Die Basismethode dieses Projekts ist ein probe-basierter Luegendetektor: ein kleiner Klassifikator (Probe)
liest die Aktivierungen eines eingefrorenen Sprachmodells und bewertet, ob eine Antwort eine Aussage enthaelt,
die das Modell selbst fuer falsch haelt. Der Nutzer hat dies am 2026-11-07 als gesetzte Methode bestaetigt.

Weight-Space-Programming (WSP, Takehi et al., Blog vom 2026-11-02, nicht peer-reviewed) zeigt, dass ein
Sprachmodell Netzwerkgewichte lesen und gezielt aendern kann. Dieses Konzept fragt, was davon die Probe-Methode
verbessern kann. Die Rahmung ist Interpretierbarkeit und Ehrlichkeitspruefung: die Methode soll verlaesslicher
erkennen, wann ein Modell etwas Falsches behauptet, das es besser weiss.

Quellen im Arbeitsrepo (Notizen, Stand 2026-11-07):

- `notes/lie-detection-improvement-plan.md` (Dialektik, Rangliste, Experimentplan)
- `notes/verified-reasoning-monitoring-and-fibs-probes.md` (Probe-Architekturen, Protokoll)
- `notes/fibs-v1-data-inspection.md` (Datenpruefung FIBSv1)
- `notes/weight-space-programming.md` (Blog)

## The Method

Die Probe bleibt die Basis. WSP liefert keine zweite Detektionsmethode, sondern Hypothesen fuer zwei Zwecke:
bessere Eingangssignale und eine kausale Pruefung dessen, was die Probe liest.

Vorgeschlagene Kombination, nach Aufwand geordnet. Alle Punkte sind Hypothesen, keine gemessenen Ergebnisse.

1. Probe-Basis zuerst messen (Voraussetzung fuer alles andere). Veroeffentlichte Probes (linear, MLP, EFC, axial)
   gegen eine Text-Baseline mit gleicher Information, bei festem niedrigem Fehlalarm (1 % und 5 %), Schwelle aus
   unabhaengigen Kalibrierdaten, Gruppen-Split nach Szenario/Familie. Ohne diese Basis ist keine Kombination
   bewertbar.
2. H-A, Aufmerksamkeitsmerkmal (aus WSP Exp. 7). Zusaetzliches Eingangsmerkmal: Anteil der Aufmerksamkeit vom
   Antwortende auf die vom Nutzer vorgeschlagene Antwort gegenueber den Fragezahlen. Test in der Familie
   `sycophancy` (7.976 Zeilen in FIBSv1, kontextuell). Erfolgskriterium: Recall bei 5 % FPR ueber der Probe allein,
   untere 95-%-Grenze der Differenz groesser als 0, Bootstrap ueber Szenarien.
3. H-B, kausale Pruefung der Probe-Richtung (aus WSP Exp. 8). Die gelernte Probe-Richtung wird im Residualstrom
   per Aktivierungs-Edit entfernt (keine Gewichtsaenderung im Produktivmodell). Gemessen wird, ob sich das
   Luegenverhalten aendert und ob der Probe-Score faellt. Das trennt "die Probe liest etwas Kausales" von
   "die Probe liest eine Begleiterscheinung". VRM weist ausdruecklich darauf hin, dass lesbare Signale keine
   Kausalitaet belegen.
4. Optional, spaeter: ein LLM-Agent als Analyst, der Probe-Gewichte und Fehlerfaelle liest, um Fehlermuster zu
   beschreiben (Interpretierbarkeits-Hilfe). Nur Beschreibung, keine Aenderung am Detektor.

## Leitplanken (verbindlich)

- Zweck ist Interpretierbarkeit und Ehrlichkeitspruefung. Die Methode wird nicht eingesetzt, um Taeuschung zu
  verschleiern oder Detektoren zu umgehen.
- Die Probe oder ihr Score darf nie als Verlustfunktion, Belohnung oder Auswahlkriterium genutzt werden, um das
  Modell darauf zu trainieren, von der Probe nicht erkannt zu werden. Das gilt fuer Training, Gewichtsedits und
  Prompt-Optimierung gleichermassen. Eine Verwendung des Scores als Trainingssignal macht die Probe als Messgeraet
  wertlos.
- Gewichts- oder Aktivierungs-Edits (H-B) dienen nur als Diagnose an Kopien oder zur Laufzeit in Experimenten.
  Ein editiertes Modell wird nicht als "ehrlicher" ausgeliefert, nur weil die Probe leiser wird. Das waere
  Optimierung gegen den Detektor.
- Ergebnisse werden auch berichtet, wenn die Probe nicht besser als die Text-Baseline ist.
- Der Detektor liefert Hinweise fuer Pruefung, keinen Beweis fuer Absicht.

## When to use / When NOT to use

Einsetzen: als Plan fuer die Reihenfolge der Experimente, sobald Methode und Datenquelle fuer EXP-1 feststehen.

Nicht einsetzen:

- Bevor die Probe-Basis (Punkt 1) gegen die Text-Baseline gemessen ist. Dann zwei unbewiesene Komponenten zu
  kombinieren verwischt jede Ursache.
- Fuer Weight-only-Auditing. WSP Exp. 7 scheiterte damit in 4 von 4 Versuchen; erst ein Aufmerksamkeits-Werkzeug
  half, und nur an einem Spielzeugmodell (Gemma 3 270M).
- Als Begruendung, ein Modell gegen die Probe zu optimieren (siehe Leitplanken).

## Evidence

Belegt (Eigenangaben der Autoren, nicht von uns reproduziert):

- WSP Blog, Seite geprueft am 2026-11-07: MATH-500 von 7,9 % auf 48,1 % mit 131.072 geaenderten Gewichten
  (0,016 %); Kopf 10.2 richtet 0,75 % (ehrlich) bzw. 85,7 % (sycophant) der Aufmerksamkeit auf die Nutzerantwort.
- VRM: Text-Baseline schlug die Aktivierungs-Probe in 4 von 6 gelesenen Vergleichen (z. B. Rollenspiel
  AUROC 0,9996 gegen 0,842). Skripted Study 1: AUROC 1,00 aber 0 von 32 bei eingefrorener Schwelle.
- fibs-probes: Validierungs-AUROC EFC/axial 0,987 bis 0,999, nur im Trainingsverteilungsbereich.

Eigene Messung (2026-11-07, `analysis/fibs_v1_inspection/`): FIBSv1 hat 163.016 Zeilen, Follow-up-Label exakt
50/50, `human_verified` nirgends gesetzt, mit Gruppen-Split schwache Metadaten-Shortcuts (Laenge 0,55,
Autor-Modell 0,59), 365 identische Antworten mit beiden Labels (16 % der Zeilen).

Nicht belegt: jede Verbesserung durch die Kombination. Bisher wurde kein Experiment ausgefuehrt.

## Offene Fragen und dokumentierte Vorschlaege

Die Fragen sind nicht aus den Quellen entscheidbar. Statt nachzufragen gilt vorerst der jeweilige Vorschlag.

- Datenquelle fuer EXP-1. Vorschlag: frische, eigene Szenarien (auch ueber ACS erzeugt), weil die veroeffentlichten
  Probes auf 152.980 von 163.016 FIBS-Zeilen trainiert wurden und FIBS damit kein Held-out liefert.
- Modellgroesse. Vorschlag: Qwen3.5-2B (24 Bloecke, Schichten 7, 10, 13, 16, 19, 22) fuer den ersten Lauf,
  9B nur bei tragfaehiger Hardware.
- Hardware fuer die Aktivierungs-Erfassung. Nicht gemessen; offen. Vorschlag: zuerst Maschinenbedarf an
  100 Transkripten messen.
- Lizenz und Revision von FIBSv1. Offen; vor jeder Weitergabe pruefen. Gewichte sind MIT, VRM ist PolyForm
  Noncommercial.
- Bedeutung von LDD im Setup. Vorschlag: nur Dialektik vor Empfehlungen und E2E-Beleg vor Erfolgsaussagen; kein
  eigener Loss-Log, bis ein Code-Ziel existiert.
- Reihenfolge H-A vor H-B, weil H-A billiger ist und ohne Modelleingriff auskommt.

## Operator Notes
