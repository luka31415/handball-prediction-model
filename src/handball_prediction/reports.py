from __future__ import annotations

import re
import shutil
import subprocess
from difflib import SequenceMatcher
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
from pypdf import PdfReader


REPORT_URL = "https://spo.handball4all.de/misc/sboPublicReports.php?sGID="
STAT_COLUMNS = (
    "home_warnings",
    "away_warnings",
    "home_two_minute_penalties",
    "away_two_minute_penalties",
    "home_red_cards",
    "away_red_cards",
    "home_blue_cards",
    "away_blue_cards",
    "home_seven_meter_goals",
    "away_seven_meter_goals",
    "home_seven_meter_attempts",
    "away_seven_meter_attempts",
    "home_distinct_scorers",
    "away_distinct_scorers",
    "home_top_scorer_goals",
    "away_top_scorer_goals",
    "home_disciplinary_players",
    "away_disciplinary_players",
)

ACTION_RE = re.compile(
    r"(?P<action>Verwarnung|2-min Strafe|Disqualifikation(?: mit Bericht)?|"
    r"7m-Tor|7m, KEIN Tor|Tor) (?:für|durch) (?P<subject>.+)$"
)
PLAYER_RE = re.compile(
    r"(?P<name>.+?) \((?P<number>\d+),\s*(?P<team>.+)\)$"
)
OFFICIAL_RE = re.compile(r"(?P<name>Offizieller\s+[A-E]),\s*(?P<team>.+)$")
ANONYMOUS_PLAYER_RE = re.compile(
    r"(?P<name>Spieler\s+(?P<number>\d+)),\s*(?P<team>.+)$"
)


def download_report(report_id: str, *, timeout: float = 20.0) -> bytes:
    request = Request(
        f"{REPORT_URL}{report_id}",
        headers={"User-Agent": "handball-match-prediction/0.2"},
    )
    with urlopen(request, timeout=timeout) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        raise ValueError(f"Report {report_id} is not a PDF")
    return data


def extract_report_text(pdf: bytes | str | Path) -> str:
    if not isinstance(pdf, bytes) and shutil.which("pdftotext"):
        completed = subprocess.run(
            ["pdftotext", "-layout", str(pdf), "-"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout
    source = BytesIO(pdf) if isinstance(pdf, bytes) else pdf
    reader = PdfReader(source)
    return "\n".join(
        page.extract_text(extraction_mode="layout") or "" for page in reader.pages
    )


def _team_side(label: str, home_team: str, away_team: str) -> str:
    def normalized(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    label_norm = normalized(label)
    scores = {
        "home": SequenceMatcher(None, label_norm, normalized(home_team)).ratio(),
        "away": SequenceMatcher(None, label_norm, normalized(away_team)).ratio(),
    }
    return max(scores, key=scores.get)


def parse_player_events(
    text: str, *, home_team: str, away_team: str, report_id: str = ""
) -> pd.DataFrame:
    """Parse player goals, cards, suspensions and seven-metre attempts."""
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = ACTION_RE.search(line.strip())
        if not match:
            continue
        subject = match.group("subject").strip()
        person = (
            PLAYER_RE.match(subject)
            or OFFICIAL_RE.match(subject)
            or ANONYMOUS_PLAYER_RE.match(subject)
        )
        if not person:
            continue

        action = match.group("action")
        event_type = {
            "Verwarnung": "warning",
            "2-min Strafe": "two_minute",
            "Disqualifikation": "red_card",
            "Disqualifikation mit Bericht": "blue_card",
            "7m-Tor": "seven_meter_goal",
            "7m, KEIN Tor": "seven_meter_miss",
            "Tor": "goal",
        }[action]
        team_label = person.group("team").strip()
        side = _team_side(team_label, home_team, away_team)
        rows.append(
            {
                "report_id": str(report_id),
                "side": side,
                "team": home_team if side == "home" else away_team,
                "player": person.group("name").strip(),
                "number": int(person.group("number"))
                if "number" in person.groupdict() and person.group("number")
                else pd.NA,
                "event": event_type,
            }
        )
    return pd.DataFrame(
        rows, columns=("report_id", "side", "team", "player", "number", "event")
    )


def parse_discipline_events(
    text: str, *, home_team: str, away_team: str, report_id: str = ""
) -> pd.DataFrame:
    """Backward-compatible alias for the expanded player-event parser."""
    return parse_player_events(
        text, home_team=home_team, away_team=away_team, report_id=report_id
    )


def aggregate_report(events: pd.DataFrame) -> dict[str, int]:
    result = {column: 0 for column in STAT_COLUMNS}
    if events.empty:
        return result

    for side in ("home", "away"):
        subset = events[events["side"] == side]
        counts = subset["event"].value_counts()
        result[f"{side}_warnings"] = int(counts.get("warning", 0))
        result[f"{side}_two_minute_penalties"] = int(counts.get("two_minute", 0))
        # A blue card is also a disqualification and therefore counts as red.
        result[f"{side}_red_cards"] = int(
            counts.get("red_card", 0) + counts.get("blue_card", 0)
        )
        result[f"{side}_blue_cards"] = int(counts.get("blue_card", 0))
        result[f"{side}_seven_meter_goals"] = int(counts.get("seven_meter_goal", 0))
        result[f"{side}_seven_meter_attempts"] = int(
            counts.get("seven_meter_goal", 0) + counts.get("seven_meter_miss", 0)
        )
        scoring = subset[subset["event"].isin(("goal", "seven_meter_goal"))]
        scorer_totals = scoring.groupby(["player", "number"], dropna=False).size()
        result[f"{side}_distinct_scorers"] = int(len(scorer_totals))
        result[f"{side}_top_scorer_goals"] = int(
            scorer_totals.max() if not scorer_totals.empty else 0
        )
        disciplinary = subset[
            subset["event"].isin(("red_card", "blue_card"))
            & ~subset["player"].str.startswith("Offizieller", na=False)
        ]
        result[f"{side}_disciplinary_players"] = int(
            disciplinary[["player", "number"]].drop_duplicates().shape[0]
        )
    return result
