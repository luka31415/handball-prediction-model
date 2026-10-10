from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_matches, load_team_context
from .model import HandballModel


LEAGUES = (
    (
        "Eppelheim 1 / Landesliga",
        "BWHV Männer-Landesliga",
        "data/tv_eppelheim_2026_27_stats.csv",
    ),
    (
        "Eppelheim 2 / Bezirksliga",
        "Rhein-Neckar Männer-Bezirksliga",
        "data/tv_eppelheim_2_2026_27_stats.csv",
    ),
)


def simulate_league(
    model: HandballModel,
    *,
    label: str,
    league_name: str,
    completed_path: str | Path,
    context: pd.DataFrame,
    simulations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    teams = sorted(
        context.loc[
            (context["season"] == "2026_27")
            & (context["league_name"] == league_name),
            "team",
        ].tolist()
    )
    if len(teams) < 2:
        raise ValueError(f"No complete team context found for {league_name}")

    completed = load_matches(completed_path)
    played = set(zip(completed["home_team"], completed["away_team"], strict=True))
    index = {team: position for position, team in enumerate(teams)}
    points = np.zeros((simulations, len(teams)), dtype=np.int16)

    for game in completed.itertuples(index=False):
        home, away = index[game.home_team], index[game.away_team]
        if game.home_goals > game.away_goals:
            points[:, home] += 2
        elif game.home_goals < game.away_goals:
            points[:, away] += 2
        else:
            points[:, home] += 1
            points[:, away] += 1

    for home_team in teams:
        for away_team in teams:
            if home_team == away_team or (home_team, away_team) in played:
                continue
            prediction = model.predict(home_team, away_team)
            draw_limit = prediction.home_win + prediction.draw
            sample = rng.random(simulations)
            home, away = index[home_team], index[away_team]
            home_wins = sample < prediction.home_win
            draws = (sample >= prediction.home_win) & (sample < draw_limit)
            away_wins = sample >= draw_limit
            points[:, home] += 2 * home_wins + draws
            points[:, away] += 2 * away_wins + draws

    # Official tie-break rules are more detailed than the modelled H/D/A output.
    # Random jitter divides equal-point finishes without inventing goal results.
    order = np.argsort(-(points + rng.random(points.shape) * 0.01), axis=1)
    title_counts = np.bincount(order[:, 0], minlength=len(teams))
    top3_counts = np.bincount(order[:, :3].ravel(), minlength=len(teams))
    rows = [
        {
            "league": label,
            "team": team,
            "title_probability": title_counts[position] / simulations,
            "top3_probability": top3_counts[position] / simulations,
            "expected_points": float(points[:, position].mean()),
        }
        for position, team in enumerate(teams)
    ]
    return pd.DataFrame(rows).sort_values(
        ["title_probability", "expected_points"], ascending=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate both Eppelheim leagues")
    parser.add_argument("--model", default="models/tv_eppelheim_all_mens.joblib")
    parser.add_argument("--context", default="data/team_season_context.csv")
    parser.add_argument("--simulations", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/championship_simulation_v011.csv")
    args = parser.parse_args()

    model = HandballModel.load(args.model)
    context = load_team_context(args.context)
    rng = np.random.default_rng(args.seed)
    result = pd.concat(
        [
            simulate_league(
                model,
                label=label,
                league_name=league_name,
                completed_path=completed_path,
                context=context,
                simulations=args.simulations,
                rng=rng,
            )
            for label, league_name, completed_path in LEAGUES
        ],
        ignore_index=True,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"\nGespeichert: {output}")


if __name__ == "__main__":
    main()
