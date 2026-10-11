# Handball Match Prediction – lokaler Prototyp (v0.13, 11.10.2026)

Dieses kleine Projekt sagt Spiele aus mehreren Herrenligen voraus. Trainiert wird
aktuell mit allen Rhein-Neckar-Herrenligen der Saison 2025/26 sowie den
relevanten aktuellen Ligen der Herren 1 und Herren 2 des TV Eppelheim.
Mannschaftsnamen sind nicht fest im Code eingebaut.

## Update 11.10.2026

Die Start-Elo-Distanz benachbarter Ligen ist jetzt aus einer zusätzlichen,
mehrjährigen Stichprobe kalibriert. Verwendet werden 22 Auf- und Absteiger aus
der 1. und 2. Bundesliga in den Spielzeiten 2021/22 bis 2025/26. Nach dem
Entfernen der direkten Duelle innerhalb der Wechselgruppen ergibt sich ein
Elo-Abstand von 137,8 Punkten; der Modellstandard wurde reproduzierbar auf 140
gerundet. Die Quelldaten stehen in `data/league_transition_calibration.csv`,
die Berechnung in `handball-calibrate-league-step` und die vollständige
Auswertung in `reports/update_2026-10-11_transition_calibration.md`.

Im unveränderten chronologischen Test auf 408 regionalen Ligaspielen verbessert
sich der Log Loss von 0,765 auf 0,763. Die Accuracy bleibt mit 64,7 % praktisch
auf demselben Niveau. Das neu trainierte Modell liegt unter
`models/handball_model_v013.pkl`.

## Update 10.10.2026

Der Modellstand wurde neu trainiert und für das nächste Spiel der zweiten Herrenmannschaft ausgewertet. Der Trainingsstand umfasst nun 1.629 Spiele; ergänzt wurden die verifizierten Partien SG Nußloch 2 gegen HSG DiMa (20:26) und SKV Sandhofen gegen SG Nußloch 2 (32:16). Die aktuelle Tabelle liegt separat in `data/current_standings_2026-10-10.csv`. Der vollständige Saisonimport war nicht erreichbar, daher ist der neue Saisonspielstand noch unvollständig. Das gespeicherte Modell liegt unter `models/handball_model_v012.pkl`; die Einzelauswertung unter `reports/update_2026-10-10.md`.

## Was das Projekt macht

- liest abgeschlossene Ligaspiele aus einer CSV-Datei,
- berechnet Features ausschließlich aus früheren Spielen,
- berücksichtigt Elo, Heimvorteil, Form, Angriffs- und Abwehrleistung sowie
  rollierende Karten-, Zeitstrafen-, 7-Meter- und Spielerwerte,
- berücksichtigt über die mittlere Elo der letzten fünf Gegner auch die
  bisherige Stärke des Spielplans,
- initialisiert Mannschaften abhängig von ihrer Ligastufe zwischen 1220 und
  1920 Elo und erkennt Auf- sowie Absteiger beim Saisonwechsel; für Eppelheim
  ist nun auch der Übergang von 2024/25 auf 2025/26 hinterlegt,
- führt den allgemeinen Heimvorteil als sichtbares Merkmal
  `has_home_advantage` statt als versteckten Intercept,
- trainiert ein multinomiales logistisches Modell,
- testet chronologisch auf den letzten 25 Prozent der Spiele,
- gibt Wahrscheinlichkeiten für Heim/Unentschieden/Auswärts und erwartete Tore aus,
- liest lange Handball4all-Spielberichte (PDF) aus,
- speichert Verwarnungen, 2-Minuten-Strafen, rote und blaue Karten,
  7-Meter-Versuche und sämtliche Tore pro Spieler,
- leitet daraus Torschützenbreite, Topscorerform, Topscoreranteil und die Zahl
  zuletzt disqualifizierter Spieler ab,
- erkennt automatisch alle kuratierten `*_stats.csv`- und `*_verified.csv`-Dateien,
- kann ein trainiertes Modell speichern und für spätere Prognosen wieder laden.

`data/demo_matches.csv` enthält weiterhin **synthetische Demodaten**. Zusätzlich
kann das Projekt eine vollständige Liga direkt aus dem öffentlichen
Handball4all-Ergebnisdienst importieren.

## Echte Saison 2025/26 importieren

Für die Herren 1 des TV Eppelheim ist die Saison 2025/26 als
„Männer-Landesliga Staffel 8“ hinterlegt:

```bash
PYTHONPATH=src python -m handball_prediction.import_cli \
  --period 137 \
  --league 146251 \
  --week 2025-09-15 \
  --output data/tv_eppelheim_2025_26.csv
```

Die neue Saison 2026/27 wird genauso importiert:

```bash
PYTHONPATH=src python -m handball_prediction.import_cli \
  --period 142 \
  --league 161196 \
  --week 2026-09-14 \
  --output data/tv_eppelheim_2026_27.csv
```

Die 2. Herren spielt 2026/27 in der Rhein-Neckar Männer-Bezirksliga:

```bash
PYTHONPATH=src python -m handball_prediction.import_cli \
  --period 142 \
  --league 164116 \
  --week 2026-09-14 \
  --output data/tv_eppelheim_2_2026_27.csv
```

Der Import übernimmt alle abgeschlossenen Spiele der Liga. Rein administrativ
gewertete 0:0-Partien werden ausgeschlossen, da sie keine echte Torleistung
abbilden.

## PDF-Statistiken importieren

Der erweiterte Import lädt zu jedem abgeschlossenen Spiel die öffentliche
Handball4all-„Presseinformation lang“. Dabei bedeutet `Disqualifikation` eine
rote Karte und `Disqualifikation mit Bericht` eine blaue Karte. Eine blaue
Karte wird zusätzlich als rote Disqualifikation gezählt.

```bash
handball-import-reports \
  --period 137 \
  --league 146251 \
  --week 2025-09-15 \
  --output data/tv_eppelheim_2025_26_stats.csv \
  --players-output data/tv_eppelheim_2025_26_player_events.csv
```

Bereits enthalten sind 1.627 eindeutige Spiele und 111.377 einzelne
Spielerereignisse. Für 2025/26 sind alle acht Männer-Landesliga-Staffeln,
Bezirksoberliga, Bezirksliga, Bezirksklasse 1 und 2 sowie alle drei Staffeln der
2. Bezirksklasse enthalten. Hinzu kommen die bisher abgeschlossenen Partien
der beiden relevanten Ligen 2026/27 und alle 26 verifizierten Ligaspiele der
ersten Eppelheimer Herren aus 2023/24.

`data/bwhv_landesliga_2025_26_sources.csv` dokumentiert für alle acht Staffeln
die verifizierte Staffelbezeichnung, Liga-ID und zugehörigen Datendateien.
`data/tv_eppelheim_2024_25_archive_sources.csv` hält die wiedergefundenen
Legacy-IDs für Eppelheim 1 und 2 fest.

`data/tv_eppelheim_player_summary.csv` fasst für 88 erfasste Spieler von
Eppelheim 1 und 2 Tore, 7-Meter, Zeitstrafen sowie rote und blaue Karten nach
Saison und Mannschaft zusammen. `matches_with_event` bezeichnet dabei Spiele,
in denen für den Spieler mindestens ein Ereignis protokolliert wurde, nicht
zwingend alle Kaderauftritte.

Der Import der Bezirksoberliga der Zweiten aus 2025/26 ist reproduzierbar mit:

```bash
handball-import-reports \
  --period 137 \
  --league 144901 \
  --week 2025-09-15 \
  --output data/tv_eppelheim_2_2025_26_stats.csv \
  --players-output data/tv_eppelheim_2_2025_26_player_events.csv
```

Für die aktuelle Bezirksliga samt PDF-Daten gilt:

```bash
handball-import-reports \
  --period 142 \
  --league 164116 \
  --week 2026-09-14 \
  --output data/tv_eppelheim_2_2026_27_stats.csv \
  --players-output data/tv_eppelheim_2_2026_27_player_events.csv
```

Mehrere PDF-Berichte werden parallel verarbeitet. Die Anzahl lässt sich mit
`--workers` steuern (Standard: 8).

Die PDF-Dateien werden standardmäßig in `.cache/handball4all-reports` abgelegt,
damit ein erneuter Lauf sie nicht noch einmal herunterladen muss. Ist das
Poppler-Programm `pdftotext` installiert, wird es automatisch für den schnellen
Textimport genutzt; andernfalls funktioniert der Import portabel über `pypdf`.

## Lokal starten

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e .

handball-predict \
  --home "HSG Weschnitz 2" \
  --away "TV Eppelheim 2" \
  --save-model models/tv_eppelheim_all_mens.joblib
```

Ohne `--data` werden automatisch alle kuratierten Dateien aus `data/` geladen.
Der aktuelle Stand umfasst 1.627 eindeutige Spiele. Die Ligazuordnung jeder
Trainingsdatei steht in `data/competition_metadata.csv`; die aktuellen
Saisonwechsel stehen in `data/team_season_context.csv`. Das gespeicherte Modell kann
danach ohne erneutes Training verwendet werden:

```bash
handball-predict \
  --load-model models/tv_eppelheim_all_mens.joblib \
  --home "TV Eppelheim 2" \
  --away "TV Edingen"
```

Ohne Installation geht es ebenfalls:

```bash
PYTHONPATH=src python -m handball_prediction.cli \
  --home "TSG Dossenheim" \
  --away "TV Eppelheim"
```

Tests:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Den Elo-Heimbonus erneut auf den zeitlichen Validierungsfenstern prüfen:

```bash
handball-tune-home-bonus
```

Der Lauf testet standardmäßig 0 bis 120 Punkte in Fünferschritten gemeinsam
mit mehreren Elo-K-Faktoren und schreibt die vollständigen Ergebnisse nach
`data/home_elo_bonus_validation.csv`.

Den Torzuschlag für die separate Ergebnisprognose erneut prüfen:

```bash
handball-tune-home-goals
```

Die ausführliche Herleitung aller Heimvorteilsannahmen steht in
[`HOME_ADVANTAGE.md`](HOME_ADVANTAGE.md).

## Erwartetes CSV-Format

Die ersten fünf Spalten bleiben unverändert. Der PDF-Import ergänzt Spalten wie
`home_two_minute_penalties`, `home_red_cards`, `home_blue_cards`,
`home_seven_meter_goals`, `home_distinct_scorers`,
`home_top_scorer_goals`, `home_disciplinary_players` und die entsprechenden
Auswärtswerte. Die separate Ereignisdatei enthält Datum, Spiel, Mannschaft,
Gegner, Spieler, Rückennummer und Ereignistyp.

Jede Zeile ist ein bereits abgeschlossenes Spiel. Sämtliche rollierenden
Features werden ohne Datenleck berechnet: Eine Prognose sieht nur frühere
Partien. Im chronologisch letzten Viertel des aktuellen Datensatzes erreicht
das Modell auf 408 Spielen 65,0 % Accuracy und 0,765 Log Loss. Standardisierte
Ausreißer werden auf ±3 begrenzt, damit ein einzelnes ungewöhnliches Spiel die
Prognose nicht dominiert.

## Ältere Spielzeiten

Alle sieben Rhein-Neckar-Herrenwettbewerbe und alle acht Männer-Landesliga-
Staffeln der Saison 2025/26 sind aus dem BWHV-Handball4all-Portal importiert.
Für 2023/24 sind nun alle 26 Ligaspiele des TV Eppelheim aus dem offiziellen
BHV-News-Archiv und dem offiziellen TVE-Mannschaftsarchiv verifiziert und in
`data/tv_eppelheim_2023_24_verified.csv` enthalten. Gegen jeden der 13 Gegner
ist genau ein Heim- und ein Auswärtsspiel vorhanden.
`data/tv_eppelheim_2023_24_sources.csv` dokumentiert für jede Zeile Quelle,
Datumsgenauigkeit und einen abweichenden Ergebnistext transparent. Wenn ein
Archivbericht nur das Spielwochenende oder sein Veröffentlichungsdatum nennt,
bleibt das Datum ausdrücklich als Näherung gekennzeichnet.

Für 2024/25
konnten die Wettbewerbe inzwischen eindeutig identifiziert werden: Eppelheim 1
spielte in der Männer-Verbandsliga Baden (Liga-ID 118091), Eppelheim 2 in der
Männer-Bezirksoberliga Rhein-Neckar-Tauber (alte Team-ID 1134031). Damit ist
für 2025/26 Eppelheim 1 als Absteiger und Eppelheim 2 als unverändert
hinterlegt. Das frühere Badische Verbandssystem liefert die vollständige alte
JSON-Antwort derzeit jedoch nicht mehr aus. Fehlende Spiele werden daher nicht
aus Tabellenständen erraten, sondern erst ergänzt, wenn Heimteam, Auswärtsteam
und Endstand eindeutig aus einer offiziellen Quelle hervorgehen.

Die Verbandsliga 2023/24 besitzt als höchste im Bestand vorkommende Spielklasse
die Ligastufe 6 und damit 1920 Start-Elo. Die Landesliga beginnt bei Stufe 5
mit 1780 Elo. Der spätere Abstieg von Eppelheim 1 ist im Saisonkontext separat
markiert.

Zum lokalen Training lässt sich der verifizierte Bestand bereits chronologisch
voranstellen:

```bash
PYTHONPATH=src python -m handball_prediction.cli \
  --data data/tv_eppelheim_2023_24_verified.csv \
         data/tv_eppelheim_2025_26.csv \
         data/tv_eppelheim_2026_27.csv \
  --home "TV Eppelheim" \
  --away "HSV Hockenheim"
```

## Installation und Start

Python 3.11 oder neuer wird benötigt. Im Repository-Verzeichnis:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e .
handball-predict --load-model models/handball_model_v013.pkl --home "SG Nußloch 2" --away "TV Eppelheim 2"
```

Tests:

```bash
python -m unittest discover -s tests -v
```

## Stand

Quellcode, kuratierte Daten, das gespeicherte v0.13-Modell und die aktuellen
Berichte sind im Repository enthalten. Der aktuelle Modellstand und seine
Datenabdeckung sind in [`reports/update_2026-10-10.md`](reports/update_2026-10-10.md)
dokumentiert. Neue Änderungen werden gesammelt am Ende der Woche committed.
