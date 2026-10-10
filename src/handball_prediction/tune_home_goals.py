from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import discover_training_files, load_many_matches
from .features import FeatureBuilder
from .reports import STAT_COLUMNS


def evaluate_home_goal_adjustments(
    matches: pd.DataFrame, adjustments: list[float]
) -> pd.DataFrame:
    """Tune the expected-goal offset on the 45%-75% validation interval."""
    builder = FeatureBuilder(home_goal_adjustment=0.0)
    rows: list[tuple[float, float, int, int]] = []
    for match in matches.itertuples(index=False):
        base_home, base_away = builder.expected_goals(
            match.home_team, match.away_team
        )
        rows.append(
            (base_home, base_away, int(match.home_goals), int(match.away_goals))
        )
        builder.update(
            match.home_team,
            match.away_team,
            int(match.home_goals),
            int(match.away_goals),
            **{
                name: float(getattr(match, name, 0.0))
                for name in STAT_COLUMNS
            },
        )

    values = np.asarray(rows, dtype=float)
    validation = values[int(len(values) * 0.45) : int(len(values) * 0.75)]
    results: list[dict[str, float]] = []
    for adjustment in adjustments:
        predicted_home = validation[:, 0] + adjustment
        predicted_away = validation[:, 1] - adjustment
        errors = np.concatenate(
            (
                predicted_home - validation[:, 2],
                predicted_away - validation[:, 3],
            )
        )
        results.append(
            {
                "home_goal_adjustment": adjustment,
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
            }
        )
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the expected-goal home adjustment before the holdout"
    )
    parser.add_argument("--data", nargs="+")
    parser.add_argument("--minimum", type=float, default=0.0)
    parser.add_argument("--maximum", type=float, default=1.5)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--output", default="data/home_goal_adjustment_validation.csv"
    )
    args = parser.parse_args()
    if args.step <= 0 or args.maximum < args.minimum:
        parser.error("Require step > 0 and maximum >= minimum")

    paths = args.data or discover_training_files()
    matches = load_many_matches(paths)
    count = int(round((args.maximum - args.minimum) / args.step))
    adjustments = [args.minimum + index * args.step for index in range(count + 1)]
    results = evaluate_home_goal_adjustments(matches, adjustments)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    minimum_mae = results["mae"].min()
    # Several neighbouring offsets can have exactly the same MAE. Use RMSE as
    # the deterministic tie-breaker rather than silently choosing the smallest.
    best = results[results["mae"] <= minimum_mae + 1e-12].sort_values("rmse").iloc[0]
    print(f"Gespeichert: {output}")
    print(
        f"Bester Tor-Offset: {best['home_goal_adjustment']:.2f} | "
        f"MAE {best['mae']:.3f} | RMSE {best['rmse']:.3f}"
    )


if __name__ == "__main__":
    main()
