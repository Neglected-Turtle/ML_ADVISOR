from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


MODEL_NAMES = (
    "logistic",
    "decision_tree",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
)


class FeatureCap(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=500):
        self.max_features = max_features

    def fit(self, features, target=None):
        return self

    def transform(self, features):
        return features[:, : self.max_features]


def build_preprocessor(features):
    numeric = list(features.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [column for column in features.columns if column not in numeric]

    return ColumnTransformer(
        [
            (
                "numeric",
                SimpleImputer(strategy="median", add_indicator=True),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_probe_models(seed):
    return {
        "logistic": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=150,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=30,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=30,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=35,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }


def sample_indices(target, size, seed, minimum=3):
    if size >= len(target):
        return np.arange(len(target))

    groups = [
        np.flatnonzero(target.to_numpy() == value)
        for value in target.unique()
    ]
    required = sum(min(minimum, len(group)) for group in groups)

    if required > size:
        minimum = max(1, size // len(groups))

    generator = np.random.default_rng(seed)
    selected = []

    for group in groups:
        count = min(minimum, len(group))
        selected.extend(generator.choice(group, count, replace=False).tolist())

    remaining = size - len(selected)
    pool = np.setdiff1d(np.arange(len(target)), selected)

    if remaining:
        selected.extend(
            generator.choice(pool, remaining, replace=False).tolist()
        )

    return generator.permutation(selected)


def dataset_features(features, target):
    rows, columns = features.shape
    numeric = features.select_dtypes(include=[np.number, "bool"])
    categorical = features.drop(columns=numeric.columns)
    class_ratios = target.value_counts(normalize=True)
    missing = features.isna().mean()
    unique = features.nunique(dropna=True) / max(rows, 1)
    skew = numeric.skew().replace([np.inf, -np.inf], np.nan)

    return {
        "n_rows": float(rows),
        "n_features": float(columns),
        "log_rows": float(np.log1p(rows)),
        "log_features": float(np.log1p(columns)),
        "rows_per_feature": float(rows / max(columns, 1)),
        "numeric_ratio": float(len(numeric.columns) / max(columns, 1)),
        "categorical_ratio": float(len(categorical.columns) / max(columns, 1)),
        "missing_ratio": float(missing.mean()),
        "max_feature_missing": float(missing.max() if columns else 0),
        "features_with_missing": float((missing > 0).mean() if columns else 0),
        "mean_unique_ratio": float(unique.mean() if columns else 0),
        "max_unique_ratio": float(unique.max() if columns else 0),
        "high_cardinality_ratio": float(
            (features.nunique(dropna=True) >= 50).mean()
            if columns
            else 0
        ),
        "mean_abs_skew": float(skew.abs().mean() if len(skew) else 0),
        "max_abs_skew": float(skew.abs().max() if len(skew) else 0),
        "n_classes": float(target.nunique()),
        "minority_ratio": float(class_ratios.min()),
        "majority_ratio": float(class_ratios.max()),
        "class_entropy": float(
            -(class_ratios * np.log2(class_ratios + 1e-12)).sum()
        ),
        "dimensionality_ratio": float(columns / max(rows, 1)),
    }


def prepare_frame(frame, target_column, max_rows, seed):
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    frame = frame.dropna(subset=[target_column]).reset_index(drop=True)
    target = frame[target_column]

    if target.nunique() < 2:
        raise ValueError("The target must contain at least two classes.")

    if target.value_counts().min() < 2:
        raise ValueError("This target has labels that appear only once. Choose a repeated categorical outcome such as class, type, status, rating, churn, or survived.")

    if len(frame) > max_rows:
        indices = sample_indices(target, max_rows, seed)
        frame = frame.iloc[indices].reset_index(drop=True)

    features = frame.drop(columns=[target_column])

    if features.shape[1] > 500:
        features = features.iloc[:, :500]

    return features, frame[target_column]


def run_probes(features, target, sample_rows, seed):
    indices = sample_indices(
        target,
        min(sample_rows, len(target)),
        seed + 11,
    )
    sample_features = features.iloc[indices].reset_index(drop=True)
    sample_target = target.iloc[indices].reset_index(drop=True)
    splitter = StratifiedShuffleSplit(
        n_splits=2,
        test_size=0.25,
        random_state=seed,
    )
    output = dataset_features(features, target)
    scores = {}

    for name, estimator in build_probe_models(seed).items():
        fold_scores = []

        for train_indices, valid_indices in splitter.split(
            sample_features,
            sample_target,
        ):
            model = Pipeline(
                [
                    (
                        "preprocessor",
                        clone(build_preprocessor(sample_features)),
                    ),
                    ("feature_cap", FeatureCap()),
                    ("model", clone(estimator)),
                ]
            )
            model.fit(
                sample_features.iloc[train_indices],
                sample_target.iloc[train_indices],
            )
            predictions = model.predict(
                sample_features.iloc[valid_indices]
            )
            fold_scores.append(
                balanced_accuracy_score(
                    sample_target.iloc[valid_indices],
                    predictions,
                )
            )

        scores[name] = float(np.mean(fold_scores))
        output[f"probe_{name}"] = scores[name]
        output[f"probe_{name}_std"] = float(np.std(fold_scores))
        output[f"probe_{name}_seconds"] = 0.0

    values = np.asarray(list(scores.values()))
    output["probe_spread"] = float(values.max() - values.min())
    output["probe_std_across_models"] = float(values.std())

    for first, second in artifact_pairs():
        output[f"probe_gap_{first}__{second}"] = (
            scores[first] - scores[second]
        )

    return output


def artifact_pairs():
    return [
        (first, second)
        for index, first in enumerate(MODEL_NAMES)
        for second in MODEL_NAMES[index + 1 :]
    ]


class ModelSelector:
    def __init__(self, artifact_dir="artifacts"):
        artifact_path = Path(artifact_dir)
        self.selector = joblib.load(artifact_path / "selector.joblib")
        self.calibrator = joblib.load(
            artifact_path / "risk_calibrator.joblib"
        )

    def recommend(
        self,
        frame,
        target_column,
        max_rows=12000,
        sample_rows=2000,
        seed=42,
    ):
        features, target = prepare_frame(
            frame,
            target_column,
            max_rows,
            seed,
        )
        probe_values = run_probes(
            features,
            target,
            sample_rows,
            seed,
        )
        probe_frame = pd.DataFrame([probe_values])
        meta = probe_frame.reindex(
            columns=self.selector["columns"],
            fill_value=0,
        ).replace([np.inf, -np.inf], np.nan).fillna(0)
        pair_scores = pd.DataFrame(
            0.0,
            index=[0],
            columns=MODEL_NAMES,
        )

        for (first, second), model in self.selector[
            "pair_models"
        ].items():
            probability = model.predict_proba(meta)[:, 1]
            pair_scores[first] += probability
            pair_scores[second] += 1 - probability

        ranking = pair_scores.iloc[0].sort_values(ascending=False)
        selector_margin = ranking.iloc[0] - ranking.iloc[1]
        risk_features = meta.copy()

        for name in MODEL_NAMES:
            risk_features[f"pair_score_{name}"] = pair_scores[name]

        risk_features["selector_margin"] = selector_margin
        risk_features = risk_features.reindex(
            columns=self.selector["risk_columns"],
            fill_value=0,
        )
        raw_risk = self.selector["risk_model"].predict_proba(
            risk_features
        )[:, 1]
        calibrated_risk = float(self.calibrator.predict(raw_risk)[0])

        if calibrated_risk <= 0.10:
            action = "top_1"
            result_count = 1
        elif calibrated_risk <= 0.25:
            action = "top_2"
            result_count = 2
        else:
            action = "top_3"
            result_count = 3

        score_min = ranking.min()
        score_range = max(ranking.max() - score_min, 1e-9)

        recommendations = [
            {
                "rank": index + 1,
                "model": model_name,
                "score": round(float((score - score_min) / score_range), 4),
                "probe_score": round(
                    float(probe_values[f"probe_{model_name}"]),
                    4,
                ),
            }
            for index, (model_name, score) in enumerate(
                ranking.head(result_count).items()
            )
        ]

        return {
            "action": action,
            "risk": round(calibrated_risk, 4),
            "recommendations": recommendations,
            "dataset": {
                "rows": int(len(features)),
                "features": int(features.shape[1]),
                "classes": int(target.nunique()),
            },
        }


def recommend_csv(csv_path, target_column, artifact_dir="artifacts"):
    frame = pd.read_csv(csv_path, low_memory=False)
    selector = ModelSelector(artifact_dir)
    return selector.recommend(frame, target_column)
