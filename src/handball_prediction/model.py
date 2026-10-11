from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .calibrate_league_step import DEFAULT_CALIBRATED_LEAGUE_ELO_STEP
from .features import (
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    HOME_ADVANTAGE_FEATURE_COLUMNS,
    FeatureBuilder,
    build_training_table,
)


def _scale_home_advantage(values: np.ndarray) -> np.ndarray:
    # This feature replaces the unpenalized classifier intercept. A large
    # numerical scale makes its effective L2 penalty negligible while the raw
    # model input remains the intuitive binary value 0/1.
    return values * 100.0


def build_model_pipeline(*, regularization_c: float = 0.015) -> Pipeline:
    continuous_pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clip",
                FunctionTransformer(
                    np.clip,
                    kw_args={"a_min": -3.0, "a_max": 3.0},
                ),
            ),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("continuous", continuous_pipeline, CONTINUOUS_FEATURE_COLUMNS),
            (
                "home_advantage",
                FunctionTransformer(_scale_home_advantage),
                HOME_ADVANTAGE_FEATURE_COLUMNS,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2_000,
                    C=regularization_c,
                    fit_intercept=False,
                    random_state=42,
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class Evaluation:
    matches: int
    accuracy: float
    log_loss: float


@dataclass(frozen=True)
class Prediction:
    home_team: str
    away_team: str
    home_win: float
    draw: float
    away_win: float
    expected_home_goals: float
    expected_away_goals: float

    @property
    def pick(self) -> str:
        values = {"H": self.home_win, "D": self.draw, "A": self.away_win}
        return max(values, key=values.get)


class HandballModel:
    def __init__(
        self,
        *,
        home_elo_bonus: float = 55.0,
        home_goal_adjustment: float = 0.8,
        elo_k: float = 32.0,
        league_elo_step: float = DEFAULT_CALIBRATED_LEAGUE_ELO_STEP,
        regularization_c: float = 0.015,
    ) -> None:
        if league_elo_step < 0:
            raise ValueError("league_elo_step must be non-negative")
        self.home_elo_bonus = float(home_elo_bonus)
        self.home_goal_adjustment = float(home_goal_adjustment)
        self.elo_k = float(elo_k)
        self.league_elo_step = float(league_elo_step)
        self.regularization_c = float(regularization_c)
        self.pipeline = build_model_pipeline(regularization_c=self.regularization_c)
        self.builder: FeatureBuilder | None = None
        self.training_matches = 0

    def fit(
        self, matches: pd.DataFrame, team_context: pd.DataFrame | None = None
    ) -> Evaluation:
        self.training_matches = len(matches)
        table, _ = build_training_table(
            matches,
            home_elo_bonus=self.home_elo_bonus,
            home_goal_adjustment=self.home_goal_adjustment,
            elo_k=self.elo_k,
            league_elo_step=self.league_elo_step,
            team_context=team_context,
        )
        if table["result"].nunique() < 3:
            raise ValueError("Training data must contain home wins, draws, and away wins")

        split = max(1, int(len(table) * 0.75))
        split = min(split, len(table) - 1)
        train = table.iloc[:split]
        test = table.iloc[split:]

        self.pipeline.fit(train[FEATURE_COLUMNS], train["result"])
        probabilities = self.pipeline.predict_proba(test[FEATURE_COLUMNS])
        predictions = self.pipeline.predict(test[FEATURE_COLUMNS])
        evaluation = Evaluation(
            matches=len(test),
            accuracy=float(accuracy_score(test["result"], predictions)),
            log_loss=float(
                log_loss(test["result"], probabilities, labels=self.pipeline.classes_)
            ),
        )

        full_table, self.builder = build_training_table(
            matches,
            home_elo_bonus=self.home_elo_bonus,
            home_goal_adjustment=self.home_goal_adjustment,
            elo_k=self.elo_k,
            league_elo_step=self.league_elo_step,
            team_context=team_context,
        )
        self.pipeline.fit(full_table[FEATURE_COLUMNS], full_table["result"])
        return evaluation

    def predict(self, home_team: str, away_team: str) -> Prediction:
        if self.builder is None:
            raise RuntimeError("Call fit() before predict()")

        row = pd.DataFrame([self.builder.features_for(home_team, away_team)])
        raw = self.pipeline.predict_proba(row[FEATURE_COLUMNS])[0]
        by_class = dict(zip(self.pipeline.classes_, raw, strict=True))
        expected_home, expected_away = self.builder.expected_goals(home_team, away_team)
        return Prediction(
            home_team=home_team,
            away_team=away_team,
            home_win=float(by_class.get("H", 0.0)),
            draw=float(by_class.get("D", 0.0)),
            away_win=float(by_class.get("A", 0.0)),
            expected_home_goals=expected_home,
            expected_away_goals=expected_away,
        )

    def save(self, path: str | Path) -> Path:
        if self.builder is None:
            raise RuntimeError("Call fit() before save()")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "HandballModel":
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("Saved file does not contain a HandballModel")
        return model
