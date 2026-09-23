from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "assets" / "images"
FIGURES.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "win_pct",
    "run_diff_per_game",
    "runs_per_game",
    "ops",
    "home_runs_per_game",
    "stolen_bases_per_game",
    "era",
    "whip",
]


def load_and_prepare():
    teams = pd.read_csv(DATA / "teams_2005_2024.csv")
    teams = teams.drop_duplicates(subset=["yearID", "teamID"], keep="last").copy()
    playoff_flag = teams[["DivWin", "WCWin", "LgWin", "WSWin"]].eq("Y").any(axis=1)
    df = teams.loc[playoff_flag].copy()

    df["world_series_winner"] = df["WSWin"].eq("Y").astype(int)
    df["win_pct"] = df["W"] / df["G"]
    df["run_diff_per_game"] = (df["R"] - df["RA"]) / df["G"]
    df["runs_per_game"] = df["R"] / df["G"]
    singles = df["H"] - df["2B"] - df["3B"] - df["HR"]
    df["obp"] = (df["H"] + df["BB"] + df["HBP"]) / (df["AB"] + df["BB"] + df["HBP"] + df["SF"])
    df["slg"] = (singles + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]) / df["AB"]
    df["ops"] = df["obp"] + df["slg"]
    df["home_runs_per_game"] = df["HR"] / df["G"]
    df["stolen_bases_per_game"] = df["SB"] / df["G"]
    df["era"] = pd.to_numeric(df["ERA"], errors="coerce")
    innings = df["IPouts"] / 3
    df["whip"] = (df["BBA"] + df["HA"]) / innings

    keep = ["yearID", "teamID", "name", "W", "L", "world_series_winner"] + FEATURES
    df = df[keep].sort_values(["yearID", "win_pct"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(DATA / "mlb_playoff_teams_2005_2024.csv", index=False)
    return df


def metrics(y_true, pred, prob):
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
    }


def fit_and_evaluate(df):
    train = df[df["yearID"] <= 2019].copy()
    test = df[df["yearID"] >= 2020].copy()
    X_train, y_train = train[FEATURES], train["world_series_winner"]
    X_test, y_test = test[FEATURES], test["world_series_winner"]

    prep = ColumnTransformer([("numbers", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), FEATURES)])
    logit = Pipeline([("prep", prep), ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))])
    forest = Pipeline([("impute", SimpleImputer(strategy="median")), ("model", RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced", random_state=42))])
    models = {"Logistic regression": logit, "Random forest": forest}

    rows = []
    predictions = test[["yearID", "teamID", "name", "world_series_winner"]].copy()
    # Because exactly one team wins each season, each method selects the single
    # highest-scoring playoff team within that season. The baseline uses win pct.
    def pick_one_per_season(frame, scores):
        scored = frame[["yearID"]].copy()
        scored["score"] = np.asarray(scores)
        chosen = scored.groupby("yearID")["score"].idxmax()
        pred = pd.Series(0, index=frame.index, dtype=int)
        pred.loc[chosen] = 1
        return pred.to_numpy()

    baseline_prob = test["win_pct"].to_numpy()
    baseline_pred = pick_one_per_season(test, baseline_prob)
    rows.append({"model": "Baseline (best win percentage)", **metrics(y_test, baseline_pred, baseline_prob)})

    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = pick_one_per_season(test, prob)
        rows.append({"model": name, **metrics(y_test, pred, prob)})
        predictions[name] = prob
        fitted[name] = model

    results = pd.DataFrame(rows)
    results.to_csv(DATA / "model_results.csv", index=False)
    predictions.to_csv(DATA / "test_predictions_2020_2024.csv", index=False)

    logit_coefs = fitted["Logistic regression"].named_steps["model"].coef_[0]
    importance = pd.DataFrame({"feature": FEATURES, "logistic_coefficient": logit_coefs})
    importance["random_forest_importance"] = fitted["Random forest"].named_steps["model"].feature_importances_
    importance.to_csv(DATA / "feature_importance.csv", index=False)
    return train, test, results, predictions, importance


def make_figures(df, test, results, predictions, importance):
    sns.set_theme(style="whitegrid", context="talk")
    red, navy, gray = "#CE1141", "#13274F", "#7C879E"

    plt.figure(figsize=(9, 6))
    sns.boxplot(data=df, x="world_series_winner", y="win_pct", color=navy)
    sns.stripplot(data=df, x="world_series_winner", y="win_pct", color=red, alpha=.65, jitter=.18)
    plt.xticks([0, 1], ["Other playoff teams", "World Series winners"])
    plt.xlabel("")
    plt.ylabel("Regular-season win percentage")
    plt.title("World Series winners usually had strong records—but not always the best")
    plt.tight_layout()
    plt.savefig(FIGURES / "win_pct_by_outcome.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=df, x="win_pct", y="run_diff_per_game", hue="world_series_winner", palette={0: navy, 1: red}, s=90, alpha=.78)
    plt.xlabel("Regular-season win percentage")
    plt.ylabel("Run differential per game")
    plt.title("Champions overlap heavily with other playoff teams")
    plt.legend(title="World Series winner", labels=["No", "Yes"])
    plt.tight_layout()
    plt.savefig(FIGURES / "win_pct_run_diff.png", dpi=180)
    plt.close()

    plot = results.set_index("model")[["balanced_accuracy", "roc_auc", "pr_auc"]]
    plot.plot(kind="bar", figsize=(10, 6), color=[navy, red, gray])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.xlabel("")
    plt.title("Model performance on the 2020–2024 holdout seasons")
    plt.xticks(rotation=12, ha="right")
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.savefig(FIGURES / "model_comparison.png", dpi=180)
    plt.close()

    imp = importance.sort_values("random_forest_importance")
    plt.figure(figsize=(9, 6))
    plt.barh(imp["feature"], imp["random_forest_importance"], color=red)
    plt.xlabel("Random-forest feature importance")
    plt.ylabel("")
    plt.title("Which regular-season statistics mattered most?")
    plt.tight_layout()
    plt.savefig(FIGURES / "feature_importance.png", dpi=180)
    plt.close()

    scored = predictions[["yearID"]].copy()
    scored["score"] = predictions["Logistic regression"].to_numpy()
    chosen = scored.groupby("yearID")["score"].idxmax()
    logit_class = pd.Series(0, index=test.index, dtype=int)
    logit_class.loc[chosen] = 1
    cm = confusion_matrix(test["world_series_winner"], logit_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(red, as_cmap=True), cbar=False)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.xticks([.5, 1.5], ["Non-winner", "Winner"])
    plt.yticks([.5, 1.5], ["Non-winner", "Winner"], rotation=0)
    plt.title("Logistic-regression confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    frame = load_and_prepare()
    train, test, results, predictions, importance = fit_and_evaluate(frame)
    make_figures(frame, test, results, predictions, importance)
    print(f"Playoff team-seasons: {len(frame)}; winners: {frame.world_series_winner.sum()}")
    print(f"Train: {len(train)} rows ({train.yearID.min()}-{train.yearID.max()}); Test: {len(test)} rows ({test.yearID.min()}-{test.yearID.max()})")
    print(results.to_string(index=False))
    print("\nMean win percentage:")
    print(frame.groupby("world_series_winner")["win_pct"].mean())
    print("\nFeature importance:")
    print(importance.sort_values("random_forest_importance", ascending=False).to_string(index=False))
