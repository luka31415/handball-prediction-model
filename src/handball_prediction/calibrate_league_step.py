from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_CALIBRATED_LEAGUE_ELO_STEP = 140.0
REQUIRED_COLUMNS = {
    "season",
    "division",
    "movement",
    "team",
    "games",
    "wins",
    "draws",
    "losses",
    "points",
}


@dataclass(frozen=True)
class LeagueStepCalibration:
    teams: int
    seasons: int
    promoted_score_rate: float
    relegated_score_rate: float
    pooled_lower_to_upper_score_rate: float
    raw_elo_step: float
    rounded_elo_step: float


def estimate_league_elo_step(rows: pd.DataFrame) -> LeagueStepCalibration:
    """Estimate an adjacent-league Elo gap from promoted/relegated cohorts.

    The source tables contain season totals. Games between two transition
    teams are removed algebraically because they do not identify a division
    gap: a double round robin among ``n`` cohort teams contributes
    ``2*n*(n-1)`` team-games and the same number of table points.
    """
    missing = sorted(REQUIRED_COLUMNS - set(rows.columns))
    if missing:
        raise ValueError(f"Missing calibration columns: {', '.join(missing)}")

    frame = rows.copy()
    numeric = ["division", "games", "wins", "draws", "losses", "points"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="raise")
    if not frame["movement"].isin(["promoted", "relegated"]).all():
        raise ValueError("movement must contain only promoted or relegated")
    if not (
        frame["games"] == frame["wins"] + frame["draws"] + frame["losses"]
    ).all():
        raise ValueError("games must equal wins + draws + losses")
    if not (frame["points"] == 2 * frame["wins"] + frame["draws"]).all():
        raise ValueError("points must equal 2*wins + draws")

    promoted_points = promoted_trials = 0.0
    relegated_points = relegated_trials = 0.0
    for (_, movement), group in frame.groupby(["season", "movement"], sort=True):
        cohort_size = len(group)
        mutual_team_games = 2 * cohort_size * (cohort_size - 1)
        cross_team_games = float(group["games"].sum() - mutual_team_games)
        cross_points = float(group["points"].sum() - mutual_team_games)
        if cross_team_games <= 0 or not 0 <= cross_points <= 2 * cross_team_games:
            raise ValueError("Invalid transition cohort totals")
        if movement == "promoted":
            promoted_points += cross_points
            promoted_trials += 2 * cross_team_games
        else:
            relegated_points += cross_points
            relegated_trials += 2 * cross_team_games

    promoted_rate = promoted_points / promoted_trials
    relegated_rate = relegated_points / relegated_trials
    # Relegated teams represent the upper division. Reverse their score rate
    # so both samples describe a lower-division team facing the upper level.
    lower_successes = promoted_points + (relegated_trials - relegated_points)
    trials = promoted_trials + relegated_trials
    lower_rate = lower_successes / trials
    raw_step = 400.0 * math.log10((1.0 - lower_rate) / lower_rate)
    rounded_step = round(raw_step / 5.0) * 5.0
    return LeagueStepCalibration(
        teams=len(frame),
        seasons=frame["season"].nunique(),
        promoted_score_rate=promoted_rate,
        relegated_score_rate=relegated_rate,
        pooled_lower_to_upper_score_rate=lower_rate,
        raw_elo_step=raw_step,
        rounded_elo_step=rounded_step,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate the Elo gap between adjacent handball leagues"
    )
    parser.add_argument(
        "--data",
        default="data/league_transition_calibration.csv",
        help="Team-season transition calibration CSV",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = estimate_league_elo_step(pd.read_csv(Path(args.data)))
    print(f"Transition teams: {result.teams} across {result.seasons} seasons")
    print(f"Promoted score rate in upper league: {result.promoted_score_rate:.1%}")
    print(f"Relegated score rate in lower league: {result.relegated_score_rate:.1%}")
    print(
        "Pooled lower-vs-upper score rate: "
        f"{result.pooled_lower_to_upper_score_rate:.1%}"
    )
    print(f"Raw adjacent-league gap: {result.raw_elo_step:.1f} Elo")
    print(f"Recommended rounded league_elo_step: {result.rounded_elo_step:.0f}")


if __name__ == "__main__":
    main()
