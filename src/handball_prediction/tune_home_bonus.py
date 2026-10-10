from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from .data import discover_training_files, load_many_matches
from .features import FEATURE_COLUMNS, build_training_table
from .model import build_model_pipeline


VALIDATION_FOLDS = ((0.45, 0.55), (0.55, 0.65), (0.65, 0.75))


def evaluate_home_bonuses(
    matches: pd.DataFrame, bonuses: list[int], elo_k_values: list[float]
) -> pd.DataFrame:
    """Jointly compare Elo home bonuses and K without touching the holdout."""
    rows: list[dict[str, float | int]] = []
    for elo_k in elo_k_values:
        for bonus in bonuses:
            table, _ = build_training_table(
                matches, home_elo_bonus=float(bonus), elo_k=elo_k
            )
            accuracies: list[float] = []
            losses: list[float] = []
            for train_fraction, validation_end_fraction in VALIDATION_FOLDS:
                train_end = int(len(table) * train_fraction)
                validation_end = int(len(table) * validation_end_fraction)
                train = table.iloc[:train_end]
                validation = table.iloc[train_end:validation_end]
                pipeline = build_model_pipeline()
                pipeline.fit(train[FEATURE_COLUMNS], train["result"])
                probabilities = pipeline.predict_proba(validation[FEATURE_COLUMNS])
                predictions = pipeline.predict(validation[FEATURE_COLUMNS])
                accuracies.append(accuracy_score(validation["result"], predictions))
                losses.append(
                    log_loss(
                        validation["result"],
                        probabilities,
                        labels=pipeline.classes_,
                    )
                )
            rows.append(
                {
                    "elo_k": elo_k,
                    "home_elo_bonus": bonus,
                    "mean_accuracy": float(np.mean(accuracies)),
                    "mean_log_loss": float(np.mean(losses)),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["elo_k", "home_elo_bonus"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the Elo home bonus on pre-holdout temporal folds"
    )
    parser.add_argument("--data", nargs="+")
    parser.add_argument("--minimum", type=int, default=0)
    parser.add_argument("--maximum", type=int, default=120)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument(
        "--elo-k-values",
        default="1,2,4,6,8,12,16,20,24,32,40,48",
        help="Comma-separated Elo K values",
    )
    parser.add_argument(
        "--output", default="data/home_elo_bonus_validation.csv"
    )
    args = parser.parse_args()
    if args.step <= 0 or args.maximum < args.minimum:
        parser.error("Require step > 0 and maximum >= minimum")

    paths = args.data or discover_training_files()
    matches = load_many_matches(paths)
    bonuses = list(range(args.minimum, args.maximum + 1, args.step))
    try:
        elo_k_values = [float(value) for value in args.elo_k_values.split(",")]
    except ValueError:
        parser.error("--elo-k-values must be comma-separated numbers")
    if not elo_k_values or any(value <= 0 for value in elo_k_values):
        parser.error("Every Elo K value must be positive")
    results = evaluate_home_bonuses(matches, bonuses, elo_k_values)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    best = results.sort_values("mean_log_loss").iloc[0]
    print(f"Gespeichert: {output}")
    print(
        f"Bestes Paar: K={best['elo_k']:g}, Bonus={int(best['home_elo_bonus'])} | "
        f"Accuracy {best['mean_accuracy']:.1%} | "
        f"Log Loss {best['mean_log_loss']:.3f}"
    )


if __name__ == "__main__":
    main()
