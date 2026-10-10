from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


ENDPOINT = "https://spo.handball4all.de/service/if_g_json.php"


def iter_games(payload: dict[str, Any]):
    """Yield each game once from the portal's current and future blocks."""
    seen: set[str] = set()
    content = payload.get("content", {})
    for key in ("actualGames", "futureGames"):
        value = content.get(key)
        blocks = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
        for block in blocks:
            for game in block.get("games", []):
                identity = str(game.get("gID", game.get("sGID", id(game))))
                if identity not in seen:
                    seen.add(identity)
                    yield game


def fetch_league(
    *,
    period_id: int,
    league_id: int,
    organization_group_id: int = 216,
    week: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Download one complete league from Handball4all's public portal API."""
    params: dict[str, str | int] = {
        "cmd": "ps",
        "og": organization_group_id,
        "p": period_id,
        "cl": league_id,
        "ca": 1,
    }
    if week:
        params["do"] = week

    request = Request(
        f"{ENDPOINT}?{urlencode(params)}",
        headers={
            "User-Agent": "handball-match-prediction/0.1",
            "Referer": "https://www.handball4all.de/",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    if not isinstance(payload, list) or not payload:
        raise ValueError("Handball4all returned an unexpected response")
    result = payload[0]
    if result.get("error"):
        raise ValueError(f"Handball4all error: {result['error']}")
    return result


def parse_completed_matches(payload: dict[str, Any]) -> tuple[pd.DataFrame, int]:
    """Convert a league response into the canonical model CSV schema.

    Returns the matches and the number of administratively decided 0:0 games
    that were excluded because they contain no meaningful scoring information.
    """
    rows: list[dict[str, Any]] = []
    excluded_forfeits = 0
    for game in iter_games(payload):
        home_goals = str(game.get("gHomeGoals", "")).strip()
        away_goals = str(game.get("gGuestGoals", "")).strip()
        if not home_goals.isdigit() or not away_goals.isdigit():
            continue

        home = int(home_goals)
        away = int(away_goals)
        comment = str(game.get("gComment", "")).strip()
        if home == 0 and away == 0 and comment:
            excluded_forfeits += 1
            continue

        date = datetime.strptime(str(game["gDate"]), "%d.%m.%y").date().isoformat()
        rows.append(
            {
                "date": date,
                "home_team": str(game["gHomeTeam"]).strip(),
                "away_team": str(game["gGuestTeam"]).strip(),
                "home_goals": home,
                "away_goals": away,
            }
        )

    matches = pd.DataFrame(
        rows,
        columns=("date", "home_team", "away_team", "home_goals", "away_goals"),
    )
    if matches.empty:
        raise ValueError("No completed matches found in the Handball4all response")
    matches = matches.sort_values("date", kind="stable").reset_index(drop=True)
    return matches, excluded_forfeits


def save_matches(matches: pd.DataFrame, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(path, index=False)
    return path
