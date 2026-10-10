from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


BASE_FEATURE_COLUMNS = [
    "elo_diff",
    "recent_opponent_elo_diff",
    "promoted_diff",
    "relegated_diff",
    "form_diff",
    "goal_diff_form",
    "attack_diff",
    "defence_diff",
]

PDF_STAT_FEATURE_COLUMNS = [
    "two_minute_diff",
    "card_diff",
    "seven_meter_efficiency_diff",
]

PLAYER_FEATURE_COLUMNS = [
    "scoring_depth_diff",
    "top_scorer_form_diff",
    "top_scorer_share_diff",
    "disciplinary_players_diff",
]

HOME_ADVANTAGE_FEATURE_COLUMNS = ["has_home_advantage"]

CONTINUOUS_FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS + PDF_STAT_FEATURE_COLUMNS + PLAYER_FEATURE_COLUMNS
)

# PDF, player, and recent-opponent features remain active in the multi-league
# model after chronological validation.
FEATURE_COLUMNS = CONTINUOUS_FEATURE_COLUMNS + HOME_ADVANTAGE_FEATURE_COLUMNS


@dataclass
class TeamState:
    elo: float = 1500.0
    initialized: bool = False
    current_season: str | None = None
    league_level: int | None = None
    promoted: float = 0.0
    relegated: float = 0.0
    opponent_elos: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    goals_for: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    goals_against: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    points: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    two_minute_penalties: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    red_cards: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    blue_cards: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    seven_meter_goals: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    seven_meter_attempts: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    distinct_scorers: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    top_scorer_goals: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    disciplinary_players: deque[float] = field(default_factory=lambda: deque(maxlen=5))

    def average(self, values: deque[float], fallback: float) -> float:
        return float(np.mean(values)) if values else fallback


class FeatureBuilder:
    """Build pre-match features and update state only after each result."""

    def __init__(
        self,
        *,
        home_elo_bonus: float = 55.0,
        home_goal_adjustment: float = 0.8,
        elo_k: float = 32.0,
        league_elo_step: float = 75.0,
        team_context: pd.DataFrame | None = None,
    ):
        self.home_elo_bonus = home_elo_bonus
        self.home_goal_adjustment = home_goal_adjustment
        self.elo_k = elo_k
        self.league_elo_step = league_elo_step
        self.states: defaultdict[str, TeamState] = defaultdict(TeamState)
        self.league_goal_samples: list[int] = []
        self.team_context: dict[tuple[str, str], dict[str, float | str]] = {}
        if team_context is not None:
            for row in team_context.itertuples(index=False):
                self.team_context[(str(row.season), str(row.team))] = {
                    "league_level": float(row.league_level),
                    "movement": str(row.movement),
                }

    def starting_elo(self, league_level: int) -> float:
        """Map the six documented league levels onto configurable Elo priors."""
        return 1500.0 + self.league_elo_step * (int(league_level) - 3)

    def _ensure_team_context(
        self,
        team: str,
        *,
        season: str | None = None,
        league_level: int | None = None,
    ) -> TeamState:
        state = self.states[team]
        explicit = self.team_context.get((str(season), team), {}) if season else {}
        level = int(explicit.get("league_level", league_level or 3))
        movement = str(explicit.get("movement", "unknown"))

        if not state.initialized:
            state.elo = self.starting_elo(level)
            state.initialized = True
            state.current_season = season
            state.league_level = level
            state.promoted = 1.0 if movement == "promoted" else 0.0
            state.relegated = 1.0 if movement == "relegated" else 0.0
            return state

        if season is not None and state.current_season != season:
            old_level = state.league_level
            if movement == "promoted" or (
                movement == "unknown" and old_level is not None and level > old_level
            ):
                state.promoted, state.relegated = 1.0, 0.0
            elif movement == "relegated" or (
                movement == "unknown" and old_level is not None and level < old_level
            ):
                state.promoted, state.relegated = 0.0, 1.0
            else:
                state.promoted = state.relegated = 0.0
            state.current_season = season
            state.league_level = level
        return state

    def apply_latest_team_context(self) -> None:
        """Prepare current teams with no completed match for future predictions."""
        latest: dict[str, tuple[str, int]] = {}
        for (season, team), values in self.team_context.items():
            candidate = (season, int(values["league_level"]))
            if team not in latest or season > latest[team][0]:
                latest[team] = candidate
        for team, (season, level) in latest.items():
            self._ensure_team_context(team, season=season, league_level=level)

    @property
    def league_goal_average(self) -> float:
        if not self.league_goal_samples:
            return 30.0
        return float(np.mean(self.league_goal_samples))

    def features_for(
        self,
        home_team: str,
        away_team: str,
        *,
        season: str | None = None,
        league_level: int | None = None,
    ) -> dict[str, float]:
        home = self._ensure_team_context(
            home_team, season=season, league_level=league_level
        )
        away = self._ensure_team_context(
            away_team, season=season, league_level=league_level
        )
        baseline = self.league_goal_average

        home_attack = home.average(home.goals_for, baseline)
        away_attack = away.average(away.goals_for, baseline)
        home_defence = home.average(home.goals_against, baseline)
        away_defence = away.average(away.goals_against, baseline)

        return {
            "has_home_advantage": 1.0,
            # The home bonus belongs only in the Elo update. Adding it here
            # would duplicate the explicit home-advantage feature.
            "elo_diff": home.elo - away.elo,
            # Positive values mean that the home team faced stronger opponents
            # over its last five matches. Ratings are captured before each
            # historical match is played, so later results cannot leak back.
            "recent_opponent_elo_diff": self._recent_opponent_elo(home)
            - self._recent_opponent_elo(away),
            "promoted_diff": home.promoted - away.promoted,
            "relegated_diff": home.relegated - away.relegated,
            "form_diff": home.average(home.points, 1.0) - away.average(away.points, 1.0),
            "goal_diff_form": (home_attack - home_defence) - (away_attack - away_defence),
            "attack_diff": home_attack - away_attack,
            "defence_diff": away_defence - home_defence,
            # Positive values mean the away team recently collected more penalties.
            "two_minute_diff": away.average(away.two_minute_penalties, 0.0)
            - home.average(home.two_minute_penalties, 0.0),
            "card_diff": (
                away.average(away.red_cards, 0.0)
                + 2.0 * away.average(away.blue_cards, 0.0)
                - home.average(home.red_cards, 0.0)
                - 2.0 * home.average(home.blue_cards, 0.0)
            ),
            "seven_meter_efficiency_diff": self._seven_meter_efficiency(home)
            - self._seven_meter_efficiency(away),
            "scoring_depth_diff": home.average(home.distinct_scorers, 0.0)
            - away.average(away.distinct_scorers, 0.0),
            "top_scorer_form_diff": home.average(home.top_scorer_goals, 0.0)
            - away.average(away.top_scorer_goals, 0.0),
            "top_scorer_share_diff": self._top_scorer_share(home)
            - self._top_scorer_share(away),
            # Positive values mean more recently disqualified away players.
            "disciplinary_players_diff": away.average(
                away.disciplinary_players, 0.0
            )
            - home.average(home.disciplinary_players, 0.0),
        }

    @staticmethod
    def _seven_meter_efficiency(state: TeamState) -> float:
        attempts = sum(state.seven_meter_attempts)
        return sum(state.seven_meter_goals) / attempts if attempts else 0.75

    @staticmethod
    def _top_scorer_share(state: TeamState) -> float:
        goals = sum(state.goals_for)
        return sum(state.top_scorer_goals) / goals if goals else 0.0

    def _recent_opponent_elo(self, state: TeamState) -> float:
        fallback = self.starting_elo(state.league_level or 3)
        return state.average(state.opponent_elos, fallback)

    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        home = self.states[home_team]
        away = self.states[away_team]
        baseline = self.league_goal_average
        home_goals = (
            home.average(home.goals_for, baseline)
            + away.average(away.goals_against, baseline)
        ) / 2 + self.home_goal_adjustment
        away_goals = (
            away.average(away.goals_for, baseline)
            + home.average(home.goals_against, baseline)
        ) / 2 - self.home_goal_adjustment
        return max(0.0, home_goals), max(0.0, away_goals)

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        season: str | None = None,
        league_level: int | None = None,
        **stats: float,
    ) -> None:
        home = self._ensure_team_context(
            home_team, season=season, league_level=league_level
        )
        away = self._ensure_team_context(
            away_team, season=season, league_level=league_level
        )

        # Store the opponent's pre-match rating. This must happen before the
        # current result changes either Elo value.
        home.opponent_elos.append(away.elo)
        away.opponent_elos.append(home.elo)

        if home_goals > away_goals:
            home_score, away_score = 1.0, 0.0
            home_points, away_points = 2.0, 0.0
        elif home_goals < away_goals:
            home_score, away_score = 0.0, 1.0
            home_points, away_points = 0.0, 2.0
        else:
            home_score = away_score = 0.5
            home_points = away_points = 1.0

        expected_home = 1.0 / (
            1.0 + 10.0 ** ((away.elo - home.elo - self.home_elo_bonus) / 400.0)
        )
        change = self.elo_k * (home_score - expected_home)
        home.elo += change
        away.elo -= change

        home.goals_for.append(float(home_goals))
        home.goals_against.append(float(away_goals))
        home.points.append(home_points)
        away.goals_for.append(float(away_goals))
        away.goals_against.append(float(home_goals))
        away.points.append(away_points)
        for side, state in (("home", home), ("away", away)):
            state.two_minute_penalties.append(
                float(stats.get(f"{side}_two_minute_penalties", 0.0))
            )
            state.red_cards.append(float(stats.get(f"{side}_red_cards", 0.0)))
            state.blue_cards.append(float(stats.get(f"{side}_blue_cards", 0.0)))
            state.seven_meter_goals.append(
                float(stats.get(f"{side}_seven_meter_goals", 0.0))
            )
            state.seven_meter_attempts.append(
                float(stats.get(f"{side}_seven_meter_attempts", 0.0))
            )
            state.distinct_scorers.append(
                float(stats.get(f"{side}_distinct_scorers", 0.0))
            )
            state.top_scorer_goals.append(
                float(stats.get(f"{side}_top_scorer_goals", 0.0))
            )
            state.disciplinary_players.append(
                float(stats.get(f"{side}_disciplinary_players", 0.0))
            )
        self.league_goal_samples.extend([home_goals, away_goals])


def build_training_table(
    matches: pd.DataFrame,
    *,
    home_elo_bonus: float = 55.0,
    home_goal_adjustment: float = 0.8,
    elo_k: float = 32.0,
    league_elo_step: float = 75.0,
    team_context: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, FeatureBuilder]:
    """Create leakage-free features: every row sees only earlier matches."""
    builder = FeatureBuilder(
        home_elo_bonus=home_elo_bonus,
        home_goal_adjustment=home_goal_adjustment,
        elo_k=elo_k,
        league_elo_step=league_elo_step,
        team_context=team_context,
    )
    rows: list[dict[str, float | str]] = []

    for match in matches.itertuples(index=False):
        season = str(getattr(match, "season", "")) or None
        league_level = int(getattr(match, "league_level", 3))
        row: dict[str, float | str] = builder.features_for(
            match.home_team,
            match.away_team,
            season=season,
            league_level=league_level,
        )
        if match.home_goals > match.away_goals:
            row["result"] = "H"
        elif match.home_goals < match.away_goals:
            row["result"] = "A"
        else:
            row["result"] = "D"
        rows.append(row)
        builder.update(
            match.home_team,
            match.away_team,
            int(match.home_goals),
            int(match.away_goals),
            season=season,
            league_level=league_level,
            **{
                name: float(getattr(match, name, 0.0))
                for name in (
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
            },
        )

    builder.apply_latest_team_context()
    return pd.DataFrame(rows), builder
