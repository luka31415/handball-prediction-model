# Kalibrierung von Auf- und Absteigern – 11.10.2026

## Ergebnis

Der Abstand zwischen zwei benachbarten Ligastufen wird ab v0.13 mit **140 Elo**
statt 75 Elo angesetzt. Der Wert stammt aus einer separaten Stichprobe der 1.
und 2. Bundesliga und wird nicht aus den Zielspielen des regionalen Testsets
abgeleitet.

| Kennzahl | Ergebnis |
|---|---:|
| Spielzeiten | 5 (2021/22–2025/26) |
| Auf- und Absteiger | 22 |
| Aufsteiger: Punktquote in der höheren Liga | 25,0 % |
| Absteiger: Punktquote in der niedrigeren Liga | 63,8 % |
| Gepoolte Punktquote Unterhaus gegen Oberhaus | 31,1 % |
| Abgeleiteter Abstand | 137,8 Elo |
| Im Modell verwendeter, gerundeter Abstand | **140 Elo** |

Direkte Duelle zweier Aufsteiger beziehungsweise zweier Absteiger wurden
algebraisch entfernt, weil sie nichts über den Abstand der Ligastufen aussagen.
Die Schätzung ist über

```bash
handball-calibrate-league-step
```

vollständig reproduzierbar.

## Regionaler Kontrolltest

Beide Varianten wurden auf genau denselben 1.629 regionalen Trainingsspielen
und demselben chronologischen Testfenster von 408 Spielen verglichen.

| Liga-Elo-Schritt | Accuracy | Log Loss |
|---:|---:|---:|
| 75 (v0.12) | 65,0 % | 0,7647 |
| **140 (v0.13)** | **65,0 %** | **0,7625** |

Die Trefferquote bleibt gerundet unverändert. Der Log Loss, der auch die
Qualität der Wahrscheinlichkeiten bewertet, verbessert sich um 0,0021.
Der externe Übergangswert wird daher übernommen. Die Bundesliga-Spiele werden
nicht direkt in das regionale Matchmodell gemischt; sie kalibrieren nur den
Abstand der Ligastufen.

## Quellen

Die Team-Saison-Werte und die jeweilige Quellseite sind in
`data/league_transition_calibration.csv` dokumentiert. Verwendet wurden die
Abschlusstabellen der 1. und 2. Bundesliga 2021/22 bis 2025/26; die Tabellen
verweisen ihrerseits auf die Handball-Bundesliga als Primärquelle.
