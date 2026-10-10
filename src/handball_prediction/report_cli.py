from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from .handball4all import fetch_league, iter_games, parse_completed_matches
from .reports import (
    STAT_COLUMNS,
    aggregate_report,
    download_report,
    extract_report_text,
    parse_player_events,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import matches plus discipline stats from Handball4all PDFs"
    )
    parser.add_argument("--period", required=True, type=int)
    parser.add_argument("--league", required=True, type=int)
    parser.add_argument("--output", required=True, help="Enriched match CSV")
    parser.add_argument("--players-output", required=True, help="Player event CSV")
    parser.add_argument("--organization-group", type=int, default=216)
    parser.add_argument("--week")
    parser.add_argument("--cache", default=".cache/handball4all-reports")
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = fetch_league(
        period_id=args.period,
        league_id=args.league,
        organization_group_id=args.organization_group,
        week=args.week,
    )
    matches, _ = parse_completed_matches(payload)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    games = []
    for game in iter_games(payload):
        if not str(game.get("gHomeGoals", "")).isdigit() or not str(
            game.get("gGuestGoals", "")
        ).isdigit():
            continue
        report_id = str(game.get("sGID", "")).strip()
        if not report_id or report_id == "0":
            continue
        games.append(game)

    def read_game(
        game: dict[str, Any],
    ) -> tuple[pd.DataFrame | None, tuple[str, str, str] | None, str | None]:
        report_id = str(game["sGID"]).strip()
        report_path = cache / f"{report_id}.pdf"
        try:
            pdf = report_path.read_bytes() if report_path.exists() else download_report(report_id)
            if not report_path.exists():
                report_path.write_bytes(pdf)
            text = extract_report_text(report_path)
            home_team = str(game["gHomeTeam"]).strip()
            away_team = str(game["gGuestTeam"]).strip()
            events = parse_player_events(
                text,
                home_team=home_team,
                away_team=away_team,
                report_id=report_id,
            )
        except Exception as exc:  # one unavailable report must not abort a season
            return None, None, f"{report_id}: {exc}"

        date = pd.to_datetime(game["gDate"], dayfirst=True).date().isoformat()
        events["date"] = date
        events["home_team"] = home_team
        events["away_team"] = away_team
        events["opponent"] = events["side"].map(
            {"home": away_team, "away": home_team}
        )
        key = (date, home_team, away_team)
        return events, key, None

    event_frames: list[pd.DataFrame] = []
    stats_by_key: dict[tuple[str, str, str], dict[str, int]] = {}
    report_ids_by_key: dict[tuple[str, str, str], str] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for events, key, failure in executor.map(read_game, games):
            if failure:
                failures.append(failure)
            elif events is not None and key is not None:
                event_frames.append(events)
                stats_by_key[key] = aggregate_report(events)
                if not events.empty:
                    report_ids_by_key[key] = str(events.iloc[0]["report_id"])

    for column in STAT_COLUMNS:
        matches[column] = 0
    matches["report_id"] = ""
    for index, match in matches.iterrows():
        key = (
            pd.Timestamp(match["date"]).date().isoformat(),
            str(match["home_team"]),
            str(match["away_team"]),
        )
        for column, value in stats_by_key.get(key, {}).items():
            matches.at[index, column] = value
        matches.at[index, "report_id"] = report_ids_by_key.get(key, "")

    output = Path(args.output)
    players_output = Path(args.players_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    players_output.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(output, index=False)
    events = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame(
            columns=(
                "report_id", "date", "home_team", "away_team", "side", "team",
                "opponent", "player", "number", "event",
            )
        )
    )
    events.to_csv(players_output, index=False)

    print(f"Gespeichert: {len(matches)} Spiele mit {len(events)} Ereignissen")
    print(f"PDF-Berichte gelesen: {len(stats_by_key)} | Fehler: {len(failures)}")
    print(f"Spiele: {output}")
    print(f"Spielerereignisse: {players_output}")
    if failures:
        print("Nicht verfügbare Berichte:")
        for failure in failures:
            print(f"- {failure}")


if __name__ == "__main__":
    main()
