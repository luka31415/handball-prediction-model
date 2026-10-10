from __future__ import annotations

import argparse

from pathlib import Path

from .data import discover_training_files, load_many_matches, load_team_context
from .model import HandballModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and run a handball match predictor")
    parser.add_argument(
        "--data",
        nargs="+",
        help=(
            "One or more CSV files. If omitted, every data/*_verified.csv and "
            "data/*_stats.csv file is used."
        ),
    )
    parser.add_argument("--home", required=True, help="Home team to predict")
    parser.add_argument("--away", required=True, help="Away team to predict")
    parser.add_argument(
        "--league-elo-step",
        type=float,
        default=75.0,
        help="Starting-Elo gap between adjacent league levels (default: 75)",
    )
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--save-model", help="Optional path for the trained model")
    model_group.add_argument("--load-model", help="Use an already trained model")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluation = None
    data_files = []
    if args.load_model:
        model = HandballModel.load(args.load_model)
    else:
        data_files = args.data or discover_training_files()
        matches = load_many_matches(data_files)
        context_path = Path(data_files[0]).parent / "team_season_context.csv"
        team_context = load_team_context(context_path) if context_path.exists() else None
        model = HandballModel(league_elo_step=args.league_elo_step)
        evaluation = model.fit(matches, team_context=team_context)
        if args.save_model:
            model.save(args.save_model)
    prediction = model.predict(args.home, args.away)

    labels = {"H": "Heimsieg", "D": "Unentschieden", "A": "Auswärtssieg"}
    if evaluation is not None:
        print(
            f"Trainiert mit {model.training_matches} Ligaspielen "
            f"aus {len(data_files)} Dateien"
        )
        print(
            f"Zeitlicher Test: {evaluation.matches} Spiele | "
            f"Accuracy {evaluation.accuracy:.1%} | Log Loss {evaluation.log_loss:.3f}"
        )
    else:
        print(
            f"Gespeichertes Modell geladen ({model.training_matches} Ligaspiele): "
            f"{args.load_model}"
        )
    print()
    print(f"{prediction.home_team} – {prediction.away_team}")
    print(f"Tipp: {labels[prediction.pick]}")
    print(f"Heimsieg:      {prediction.home_win:.1%}")
    print(f"Unentschieden: {prediction.draw:.1%}")
    print(f"Auswärtssieg:  {prediction.away_win:.1%}")
    print(
        "Erwartete Tore: "
        f"{prediction.expected_home_goals:.1f} : {prediction.expected_away_goals:.1f}"
    )
    if args.save_model:
        print(f"Modell gespeichert: {args.save_model}")


if __name__ == "__main__":
    main()
