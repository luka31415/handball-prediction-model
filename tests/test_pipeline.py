from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from handball_prediction.calibrate_league_step import (
    DEFAULT_CALIBRATED_LEAGUE_ELO_STEP,
    estimate_league_elo_step,
)
from handball_prediction.data import (
    discover_training_files,
    load_many_matches,
    load_matches,
    load_team_context,
)
from handball_prediction.features import (
    FEATURE_COLUMNS,
    FeatureBuilder,
    PLAYER_FEATURE_COLUMNS,
    build_training_table,
)
from handball_prediction.handball4all import parse_completed_matches
from handball_prediction.model import HandballModel
from handball_prediction.reports import aggregate_report, parse_discipline_events


ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA = ROOT / "data" / "demo_matches.csv"


class PipelineTest(unittest.TestCase):
    def test_verified_historical_tve_data_is_loadable_and_unique(self) -> None:
        path = ROOT / "data" / "tv_eppelheim_2023_24_verified.csv"
        matches = load_matches(path)

        self.assertEqual(len(matches), 26)
        self.assertTrue(
            (
                (matches["home_team"] == "TV Eppelheim")
                | (matches["away_team"] == "TV Eppelheim")
            ).all()
        )
        self.assertEqual(len(matches.drop_duplicates()), len(matches))
        opponents = pd.concat(
            [
                matches.loc[matches["home_team"] == "TV Eppelheim", "away_team"],
                matches.loc[matches["away_team"] == "TV Eppelheim", "home_team"],
            ],
            ignore_index=True,
        )
        self.assertEqual(opponents.nunique(), 13)
        self.assertTrue((opponents.value_counts() == 2).all())

    def test_full_curated_dataset_is_unique(self) -> None:
        files = discover_training_files(ROOT / "data")
        matches = load_many_matches(files)

        self.assertEqual(len(files), 19)
        self.assertEqual(len(matches), 1_629)
        self.assertEqual(
            len(
                matches.drop_duplicates(
                    ["date", "home_team", "away_team", "home_goals", "away_goals"]
                )
            ),
            len(matches),
        )

    def test_demo_data_trains_and_predicts(self) -> None:
        matches = load_matches(DEMO_DATA)
        model = HandballModel()
        evaluation = model.fit(matches)
        prediction = model.predict("TV Eppelheim", "TSV Rot-Malsch 2")

        self.assertGreater(evaluation.matches, 0)
        self.assertAlmostEqual(
            prediction.home_win + prediction.draw + prediction.away_win,
            1.0,
            places=7,
        )

    def test_first_match_uses_neutral_pre_match_state(self) -> None:
        matches = load_matches(DEMO_DATA)
        table, _ = build_training_table(matches.head(1))
        self.assertEqual(table.loc[0, "elo_diff"], 0.0)
        self.assertEqual(table.loc[0, "recent_opponent_elo_diff"], 0.0)
        self.assertEqual(table.loc[0, "form_diff"], 0.0)
        self.assertEqual(table.loc[0, "has_home_advantage"], 1.0)

    def test_home_elo_bonus_is_configurable(self) -> None:
        matches = load_matches(DEMO_DATA)
        table, builder = build_training_table(matches.head(1), home_elo_bonus=35.0)
        self.assertEqual(table.loc[0, "elo_diff"], 0.0)
        self.assertEqual(builder.home_elo_bonus, 35.0)

    def test_recent_opponent_elo_uses_pre_match_rating_without_leakage(self) -> None:
        matches = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "home_team": "Strong",
                    "away_team": "Opponent",
                    "home_goals": 30,
                    "away_goals": 20,
                },
                {
                    "date": "2026-01-02",
                    "home_team": "Tracked",
                    "away_team": "Opponent",
                    "home_goals": 25,
                    "away_goals": 20,
                },
                {
                    "date": "2026-01-03",
                    "home_team": "Tracked",
                    "away_team": "Fresh",
                    "home_goals": 25,
                    "away_goals": 20,
                },
            ]
        )

        table, builder = build_training_table(matches)

        expected_home = 1.0 / (1.0 + 10.0 ** (-55.0 / 400.0))
        opponent_rating_before_second_match = 1500.0 - 32.0 * (1.0 - expected_home)
        self.assertAlmostEqual(
            builder.states["Tracked"].opponent_elos[0],
            opponent_rating_before_second_match,
        )
        self.assertAlmostEqual(
            table.loc[2, "recent_opponent_elo_diff"],
            opponent_rating_before_second_match - 1500.0,
        )

    def test_recent_opponent_elo_is_an_active_feature(self) -> None:
        self.assertIn("recent_opponent_elo_diff", FEATURE_COLUMNS)
        self.assertIn("promoted_diff", FEATURE_COLUMNS)
        self.assertIn("relegated_diff", FEATURE_COLUMNS)
        self.assertEqual(len(FEATURE_COLUMNS), 16)

    def test_starting_elo_depends_on_league_level(self) -> None:
        builder = FeatureBuilder()

        builder.features_for("High A", "High B", season="2025_26", league_level=6)
        builder.features_for("Low A", "Low B", season="2025_26", league_level=1)

        self.assertEqual(builder.states["High A"].elo, 1920.0)
        self.assertEqual(builder.states["Low A"].elo, 1220.0)

    def test_league_step_is_reproducible_from_transition_data(self) -> None:
        rows = pd.read_csv(ROOT / "data" / "league_transition_calibration.csv")
        calibration = estimate_league_elo_step(rows)

        self.assertEqual(calibration.teams, 22)
        self.assertEqual(calibration.seasons, 5)
        self.assertAlmostEqual(calibration.raw_elo_step, 137.8264, places=3)
        self.assertEqual(
            calibration.rounded_elo_step, DEFAULT_CALIBRATED_LEAGUE_ELO_STEP
        )

    def test_promotion_and_relegation_are_set_on_season_transition(self) -> None:
        builder = FeatureBuilder()
        builder.features_for("Promoted", "Stable", season="2025_26", league_level=3)
        promoted = builder.features_for(
            "Promoted", "New", season="2026_27", league_level=4
        )
        builder.features_for("Relegated", "Stable 2", season="2025_26", league_level=4)
        relegated = builder.features_for(
            "Relegated", "New 2", season="2026_27", league_level=3
        )

        self.assertEqual(promoted["promoted_diff"], 1.0)
        self.assertEqual(promoted["relegated_diff"], 0.0)
        self.assertEqual(relegated["promoted_diff"], 0.0)
        self.assertEqual(relegated["relegated_diff"], 1.0)

    def test_current_team_context_covers_teams_without_a_completed_game(self) -> None:
        context = load_team_context(ROOT / "data" / "team_season_context.csv")
        builder = FeatureBuilder(team_context=context)
        builder.apply_latest_team_context()

        self.assertEqual(builder.states["TV Friedrichsf"].elo, 1780.0)
        self.assertEqual(builder.states["TV Schriesheim 2"].promoted, 1.0)
        self.assertEqual(builder.states["TV Eppelheim 2"].relegated, 1.0)

    def test_eppelheim_2025_26_transition_uses_verified_previous_leagues(self) -> None:
        context = load_team_context(ROOT / "data" / "team_season_context.csv")
        eppelheim = context[
            (context["season"] == "2025_26")
            & context["team"].isin(["TV Eppelheim", "TV Eppelheim 2"])
        ].set_index("team")

        self.assertEqual(eppelheim.loc["TV Eppelheim", "previous_league_level"], 6)
        self.assertEqual(eppelheim.loc["TV Eppelheim", "movement"], "relegated")
        self.assertEqual(eppelheim.loc["TV Eppelheim 2", "previous_league_level"], 4)
        self.assertEqual(eppelheim.loc["TV Eppelheim 2", "movement"], "same")

    def test_home_advantage_is_an_explicit_model_feature(self) -> None:
        self.assertIn("has_home_advantage", FEATURE_COLUMNS)
        model = HandballModel()
        self.assertFalse(model.pipeline.named_steps["classifier"].fit_intercept)

    def test_expected_goals_use_validated_home_adjustment(self) -> None:
        matches = load_matches(DEMO_DATA)
        _, builder = build_training_table(matches.head(0))
        home, away = builder.expected_goals("Home", "Away")
        self.assertAlmostEqual(home, 30.8)
        self.assertAlmostEqual(away, 29.2)

    def test_pdf_statistics_are_active_model_features(self) -> None:
        self.assertIn("two_minute_diff", FEATURE_COLUMNS)
        self.assertIn("card_diff", FEATURE_COLUMNS)
        self.assertIn("seven_meter_efficiency_diff", FEATURE_COLUMNS)

    def test_player_statistics_are_active_model_features(self) -> None:
        self.assertEqual(len(PLAYER_FEATURE_COLUMNS), 4)
        self.assertTrue(set(PLAYER_FEATURE_COLUMNS).issubset(FEATURE_COLUMNS))

    def test_model_clips_extreme_standardized_features(self) -> None:
        model = HandballModel()
        self.assertIn("preprocess", model.pipeline.named_steps)

    def test_handball4all_parser_excludes_forfeit(self) -> None:
        payload = {
            "content": {
                "actualGames": [],
                "futureGames": {
                    "games": [
                        {
                            "gID": "1",
                            "gDate": "20.09.25",
                            "gHomeTeam": "TV Eppelheim",
                            "gGuestTeam": "TV Brühl",
                            "gHomeGoals": "23",
                            "gGuestGoals": "24",
                            "gComment": " ",
                        },
                        {
                            "gID": "2",
                            "gDate": "18.04.26",
                            "gHomeTeam": "TV Eppelheim",
                            "gGuestTeam": "S3L Handball 3",
                            "gHomeGoals": "0",
                            "gGuestGoals": "0",
                            "gComment": "(2:0), gg. Gast, Absage Gast",
                        },
                    ]
                },
            }
        }

        matches, excluded = parse_completed_matches(payload)

        self.assertEqual(len(matches), 1)
        self.assertEqual(excluded, 1)
        self.assertEqual(matches.loc[0, "date"], "2025-09-20")

    def test_multiple_csv_files_are_combined_without_duplicates(self) -> None:
        matches = load_many_matches([DEMO_DATA, DEMO_DATA])
        original = load_matches(DEMO_DATA)

        self.assertEqual(len(matches), len(original))
        self.assertTrue(matches["date"].is_monotonic_increasing)

    def test_discovery_finds_all_curated_league_files(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first_stats.csv").write_text("x\n", encoding="utf-8")
            (root / "second_stats.csv").write_text("x\n", encoding="utf-8")
            (root / "older_verified.csv").write_text("x\n", encoding="utf-8")
            (root / "raw.csv").write_text("x\n", encoding="utf-8")

            files = discover_training_files(root)

        self.assertEqual(
            {path.name for path in files},
            {"first_stats.csv", "second_stats.csv", "older_verified.csv"},
        )

    def test_combining_core_and_stats_rows_prefers_enriched_game(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            core = pd.read_csv(DEMO_DATA).head(1)
            stats = core.assign(home_red_cards=1)
            core.to_csv(root / "core.csv", index=False)
            stats.to_csv(root / "stats.csv", index=False)

            matches = load_many_matches([root / "core.csv", root / "stats.csv"])

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches.loc[0, "home_red_cards"], 1)

    def test_trained_model_can_be_saved_and_loaded(self) -> None:
        matches = load_matches(DEMO_DATA)
        model = HandballModel()
        model.fit(matches)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"
            model.save(path)
            prediction = HandballModel.load(path).predict(
                "TV Eppelheim", "TSV Rot-Malsch 2"
            )

        self.assertAlmostEqual(
            prediction.home_win + prediction.draw + prediction.away_win,
            1.0,
            places=7,
        )

    def test_report_parser_distinguishes_red_and_blue_cards(self) -> None:
        text = """
        18:18:24 29:06 Disqualifikation für Max Rot (8, TV Eppelheim)
        18:37:10 35:42 Disqualifikation mit Bericht für Ben Blau (27, TV Brühl)
        18:43:32 39:09 2-min Strafe für Ben Blau (27, TV Brühl)
        18:50:00 44:10 7m-Tor durch Max Rot (8, TV Eppelheim)
        18:51:00 45:10 7m, KEIN Tor durch Max Rot (8, TV Eppelheim)
        18:52:00 46:10 Tor durch Max Rot (8, TV Eppelheim)
        18:53:00 47:10 Tor durch Mia Tor (10, TV Eppelheim)
        """
        events = parse_discipline_events(
            text, home_team="TV Eppelheim", away_team="TV Brühl", report_id="42"
        )
        stats = aggregate_report(events)

        self.assertEqual(stats["home_red_cards"], 1)
        self.assertEqual(stats["away_red_cards"], 1)
        self.assertEqual(stats["away_blue_cards"], 1)
        self.assertEqual(stats["away_two_minute_penalties"], 1)
        self.assertEqual(stats["home_seven_meter_goals"], 1)
        self.assertEqual(stats["home_seven_meter_attempts"], 2)
        self.assertEqual(stats["home_distinct_scorers"], 2)
        self.assertEqual(stats["home_top_scorer_goals"], 2)
        self.assertEqual(stats["home_disciplinary_players"], 1)


if __name__ == "__main__":
    unittest.main()
