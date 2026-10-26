"""Model training, consistent evaluation, and local explanations."""
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.base import clone


def get_models(seed=42):
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", RandomForestClassifier(
                n_estimators=200, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
            )),
        ]),
    }
    try:
        from xgboost import XGBClassifier
        # scale_pos_weight is configured per training split in evaluate_models.
        models["XGBoost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=.05,
                subsample=.85, colsample_bytree=.85, reg_lambda=1,
                eval_metric="logloss", random_state=seed, n_jobs=-1,
            )),
        ])
    except Exception as exc:
        warnings.warn(f"XGBoost unavailable: {exc}")
    return models


def _metric_row(name, y_true, probabilities, threshold=.5):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    try:
        roc_auc = roc_auc_score(y_true, probabilities)
    except Exception:
        roc_auc = np.nan
    try:
        pr_auc = average_precision_score(y_true, probabilities)
    except Exception:
        pr_auc = np.nan
    return {
        "Model": name,
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced accuracy": balanced_accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
    }, predictions


def evaluate_models(X, y, seed=42):
    """Train on one stratified split and derive every metric from stored outputs."""
    X = X.copy().reset_index(drop=True)
    y = pd.Series(y).astype(int).reset_index(drop=True)
    if set(y.unique()) != {0, 1}:
        raise ValueError(f"Expected binary target labels 0 and 1, got {sorted(y.unique())}")
    if y.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two examples for stratified evaluation.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=seed, stratify=y
    )
    results, fitted = [], {}
    for name, pipeline in get_models(seed).items():
        try:
            model = clone(pipeline)
            if name == "XGBoost":
                # Correct for the rare bankruptcy class without resampling the test set.
                estimator = model.named_steps["model"]
                estimator.set_params(scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)))
            model.fit(X_train, y_train)
            probabilities = model.predict_proba(X_test)[:, 1]
            row, predictions = _metric_row(name, y_test, probabilities, threshold=.5)
            results.append(row)
            fitted[name] = {
                "pipeline": model,
                "X_train": X_train.copy(),
                "X_test": X_test.copy(),
                "y_train": y_train.copy(),
                "y_test": y_test.copy(),
                "prob": probabilities.copy(),
                "pred": predictions.copy(),
            }
        except Exception as exc:
            warnings.warn(f"{name} failed: {exc}")
    metrics = pd.DataFrame(results)
    if metrics.empty:
        raise RuntimeError("No model trained successfully; check the dataset and dependencies.")
    # For rare bankruptcy cases, PR-AUC is more informative than accuracy alone.
    metrics = metrics.sort_values(["PR-AUC", "F1", "ROC-AUC"], ascending=False, na_position="last").reset_index(drop=True)
    return metrics, fitted


def metrics_for_threshold(name, result, threshold=.5):
    row, predictions = _metric_row(name, result["y_test"], result["prob"], threshold)
    return row, predictions


def risk_band(probability):
    # Illustrative only; these cutoffs are not calibrated or regulatory ratings.
    return "LOW" if probability < .30 else ("MEDIUM" if probability < .60 else "HIGH")



def explain_prediction(pipeline, X_one, background_X, max_features=12):
    """
    Explain one bankruptcy prediction.

    Uses SHAP when available. If SHAP fails, returns a clearly labelled
    one-feature-at-a-time sensitivity analysis instead.
    """
    if X_one.empty:
        raise ValueError("The selected company record is empty.")

    if background_X.empty:
        raise ValueError("The explanation background dataset is empty.")

    # Keep the background sample small for practical local execution.
    background = background_X.sample(
        n=min(20, len(background_X)),
        random_state=42
    )

    base_probability = float(
        pipeline.predict_proba(X_one)[:, 1][0]
    )

    try:
        import shap

        explainer = shap.Explainer(
            pipeline.predict_proba,
            background
        )

        explanation = explainer(
            X_one,
            max_evals=max(2 * X_one.shape[1] + 1, 100)
        )

        values = explanation.values

        # SHAP may return a list or arrays with different class dimensions.
        if isinstance(values, list):
            values = np.asarray(values[1] if len(values) > 1 else values[0])
            contributions = values[0] if values.ndim > 1 else values
        else:
            values = np.asarray(values)

            if values.ndim == 3:
                contributions = values[0, :, 1]
            elif values.ndim == 2:
                contributions = values[0, :]
            elif values.ndim == 1:
                contributions = values
            else:
                raise ValueError(
                    f"Unexpected SHAP output shape: {values.shape}"
                )

        if len(contributions) != X_one.shape[1]:
            raise ValueError(
                "SHAP output does not match the input feature count."
            )

        output = pd.DataFrame({
            "Feature": X_one.columns,
            "Contribution": np.asarray(contributions, dtype=float)
        })

        output["AbsContribution"] = output["Contribution"].abs()

        return (
            output.sort_values(
                "AbsContribution", ascending=False
            ).head(max_features),
            "SHAP"
        )

    except Exception as exc:
        # This is a sensitivity analysis, not a SHAP explanation.
        rows = []

        for column in X_one.columns:
            changed = X_one.copy()
            changed[column] = background_X[column].median()

            changed_probability = float(
                pipeline.predict_proba(changed)[:, 1][0]
            )

            rows.append({
                "Feature": column,
                "Contribution": base_probability - changed_probability,
                "AbsContribution": abs(
                    base_probability - changed_probability
                )
            })

        fallback = (
            pd.DataFrame(rows)
            .sort_values("AbsContribution", ascending=False)
            .head(max_features)
        )

        method = (
            "Sensitivity fallback (not SHAP; "
            f"{type(exc).__name__})"
        )

        return fallback, method
