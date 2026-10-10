# Handball Prediction Model

A data-driven project for predicting German handball match outcomes and league standings.

## Model approach

The model incorporates historical league match results, league-dependent initial Elo ratings, team promotions and relegations, recent opponents' strength, and home-court advantage.

## Latest documented experiment (v0.12 — 2026-10-10)

- Training data: 1,629 league matches from 19 curated files
- Chronological test: 408 matches
- Test accuracy: 65.0%
- Test log loss: 0.765
- Example match forecast: SG Nußloch 2 vs TV Eppelheim 2 (2026-10-10): 52.3% home win, 7.2% draw, 40.5% away win; expected score 27.9–26.8

## Repository status

This repository is being initialized from the project's documented experiment history. The complete v0.12 Python source, training data, and serialized model are **not yet uploaded** here. The source archive from the earlier local project session is needed before a runnable release can be committed. Performance figures above are from the existing project report, not recomputed from this repository.

## Planned workflow

Changes are accumulated during the week and reviewed for a batch commit at the end of the week. No new features or evaluation figures should be documented as completed until their code and results are available.
