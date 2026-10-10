from __future__ import annotations

from pathlib import Path

import pandas as pd

from .reports import STAT_COLUMNS


REQUIRED_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
)

MATCH_CONTEXT_COLUMNS = ("season", "league_level", "league_name")
TEAM_CONTEXT_COLUMNS = (
    "season",
    "team",
    "league_level",
    "league_name",
    "previous_league_level",
    "movement",
)


def _season_from_date(value: pd.Timestamp) -> str:
    start = value.year if value.month >= 7 else value.year - 1
    return f"{start}_{str(start + 1)[-2:]}"


def _competition_context(path: Path) -> dict[str, str | int]:
    catalog_path = path.parent / "competition_metadata.csv"
    if not catalog_path.exists():
        return {}
    catalog = pd.read_csv(catalog_path)
    required = {"source_file", *MATCH_CONTEXT_COLUMNS}
    if not required.issubset(catalog.columns):
        raise ValueError(
            f"{catalog_path} must contain: {', '.join(sorted(required))}"
        )
    selected = catalog[catalog["source_file"] == path.name]
    if selected.empty:
        return {}
    if len(selected) != 1:
        raise ValueError(f"Duplicate competition metadata for {path.name}")
    row = selected.iloc[0]
    return {
        "season": str(row["season"]),
        "league_level": int(row["league_level"]),
        "league_name": str(row["league_name"]),
    }


def discover_training_files(data_dir: str | Path = "data") -> list[Path]:
    """Find every curated season file without mixing in raw duplicate exports."""
    root = Path(data_dir)
    files = sorted(root.glob("*_verified.csv")) + sorted(root.glob("*_stats.csv"))
    if not files:
        raise ValueError(
            f"No training files found in {root}. Expected *_verified.csv or *_stats.csv"
        )
    return files


def load_matches(path: str | Path) -> pd.DataFrame:
    """Load completed matches and validate the small canonical CSV schema."""
    path = Path(path)
    matches = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in matches.columns]
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(missing)}")

    optional = [column for column in STAT_COLUMNS if column in matches.columns]
    supplied_context = [
        column for column in MATCH_CONTEXT_COLUMNS if column in matches.columns
    ]
    matches = matches.loc[
        :, (*REQUIRED_COLUMNS, *optional, *supplied_context)
    ].copy()
    matches["date"] = pd.to_datetime(matches["date"], errors="raise")
    matches["home_goals"] = pd.to_numeric(matches["home_goals"], errors="raise").astype(int)
    matches["away_goals"] = pd.to_numeric(matches["away_goals"], errors="raise").astype(int)
    for column in optional:
        matches[column] = pd.to_numeric(matches[column], errors="raise").astype(int)

    file_context = _competition_context(path)
    if "season" not in matches:
        if "season" in file_context:
            matches["season"] = file_context["season"]
        else:
            matches["season"] = matches["date"].map(_season_from_date)
    if "league_level" not in matches:
        matches["league_level"] = int(file_context.get("league_level", 3))
    if "league_name" not in matches:
        matches["league_name"] = str(file_context.get("league_name", "Unbekannt"))
    matches["season"] = matches["season"].astype(str)
    matches["league_level"] = pd.to_numeric(
        matches["league_level"], errors="raise"
    ).astype(int)
    matches["league_name"] = matches["league_name"].astype(str)
    if not matches["league_level"].between(1, 6).all():
        raise ValueError("league_level must be between 1 and 6")

    if matches[["home_team", "away_team"]].isna().any().any():
        raise ValueError("Team names must not be empty")
    if (matches[["home_goals", "away_goals"]] < 0).any().any():
        raise ValueError("Goals must be non-negative")
    if optional and (matches[optional] < 0).any().any():
        raise ValueError("Match statistics must be non-negative")
    if (matches["home_team"] == matches["away_team"]).any():
        raise ValueError("A team cannot play against itself")

    return matches.sort_values("date", kind="stable").reset_index(drop=True)


def load_team_context(path: str | Path) -> pd.DataFrame:
    """Load explicit team/season league transitions, including unplayed teams."""
    context = pd.read_csv(path)
    missing = [column for column in TEAM_CONTEXT_COLUMNS if column not in context]
    if missing:
        raise ValueError(f"Missing team context columns: {', '.join(missing)}")
    context = context.loc[:, TEAM_CONTEXT_COLUMNS].copy()
    context["season"] = context["season"].astype(str)
    context["team"] = context["team"].astype(str).str.strip()
    context["league_level"] = pd.to_numeric(
        context["league_level"], errors="raise"
    ).astype(int)
    context["previous_league_level"] = pd.to_numeric(
        context["previous_league_level"], errors="coerce"
    )
    context["movement"] = context["movement"].fillna("unknown").astype(str)
    allowed = {"same", "promoted", "relegated", "unknown"}
    if not set(context["movement"]).issubset(allowed):
        raise ValueError(f"movement must be one of: {', '.join(sorted(allowed))}")
    known_previous = context["previous_league_level"].notna()
    for row in context[known_previous].itertuples(index=False):
        previous = int(row.previous_league_level)
        expected = (
            "promoted"
            if row.league_level > previous
            else "relegated"
            if row.league_level < previous
            else "same"
        )
        if row.movement != expected:
            raise ValueError(
                f"Inconsistent movement for {row.team}: expected {expected}"
            )
    if not context["league_level"].between(1, 6).all():
        raise ValueError("league_level must be between 1 and 6")
    if context.duplicated(["season", "team"]).any():
        raise ValueError("Duplicate season/team rows in team context")
    return context.sort_values(["season", "team"], kind="stable").reset_index(drop=True)


def load_many_matches(paths: list[str | Path]) -> pd.DataFrame:
    """Load and chronologically combine one or more canonical match files."""
    if not paths:
        raise ValueError("At least one CSV file is required")
    matches = pd.concat((load_matches(path) for path in paths), ignore_index=True)
    available_stats = [column for column in STAT_COLUMNS if column in matches.columns]
    if available_stats:
        matches[available_stats] = matches[available_stats].fillna(0).astype(int)
        # Prefer the richer PDF row if the same game occurs in a raw and enriched file.
        matches["_stat_score"] = (matches[available_stats] != 0).sum(axis=1)
        matches = matches.sort_values("_stat_score", ascending=False, kind="stable")
    matches = matches.drop_duplicates(subset=list(REQUIRED_COLUMNS), keep="first")
    return (
        matches.drop(columns="_stat_score", errors="ignore")
        .sort_values("date", kind="stable")
        .reset_index(drop=True)
    )
