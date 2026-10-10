from __future__ import annotations

import argparse

from .handball4all import fetch_league, parse_completed_matches, save_matches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import completed league matches from Handball4all"
    )
    parser.add_argument("--period", required=True, type=int, help="Season/period ID")
    parser.add_argument("--league", required=True, type=int, help="League ID")
    parser.add_argument("--output", required=True, help="Destination CSV file")
    parser.add_argument("--organization-group", type=int, default=216)
    parser.add_argument(
        "--week",
        help="Any Monday in the season (YYYY-MM-DD); useful for historical seasons",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = fetch_league(
        period_id=args.period,
        league_id=args.league,
        organization_group_id=args.organization_group,
        week=args.week,
    )
    matches, excluded = parse_completed_matches(payload)
    output = save_matches(matches, args.output)

    league_name = payload.get("head", {}).get("headline2", f"Liga {args.league}")
    teams = set(matches["home_team"]) | set(matches["away_team"])
    print(league_name)
    print(f"Gespeichert: {len(matches)} Spiele von {len(teams)} Teams")
    print(f"Ausgeschlossen: {excluded} gewertete 0:0-Spiele")
    print(f"Datei: {output}")


if __name__ == "__main__":
    main()
