import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from src_data import load_uci_bankruptcy, make_synthetic_demo
from modeling import evaluate_models, metrics_for_threshold, risk_band, explain_prediction

st.set_page_config(page_title="FinSight AI | Financial Risk", page_icon="📊", layout="wide")
st.title("FinSight AI")
st.subheader("Explainable AI-Based Financial Risk Assessment & Investment Decision Support")
st.write("Academic prototype for company-bankruptcy risk classification, model comparison, and feature-level explanations.")

@st.cache_data(show_spinner=False)
def load_data():
    try:
        return load_uci_bankruptcy(cohort_year=3)
    except Exception as exc:
        X, y, label = make_synthetic_demo()
        return X, y, label + f" | UCI load unavailable: {type(exc).__name__}"

with st.spinner("Loading dataset..."):
    X, y, data_label = load_data()
if data_label.startswith("SYNTHETIC"):
    st.warning("DEMO MODE: synthetic data is being used. Do not report these metrics as real-dataset results.")
else:
    st.success("Real-data mode: UCI Polish Companies Bankruptcy — single 3rd-year cohort loaded.")

with st.expander("Dataset and methodology notes"):
    st.write(f"**Data source:** {data_label}")
    st.write(f"**Rows:** {len(X):,} | **Financial features:** {X.shape[1]} | **Positive class:** 1 = bankrupt")
    st.write("A single cohort is used so the target forecasting horizon is not mixed across years.")
    st.write("The dataset is highly imbalanced; therefore recall and PR-AUC are important alongside accuracy and ROC-AUC.")
    st.write("Low/Medium/High risk bands are illustrative probability thresholds, not validated credit ratings.")
    st.write("This is a retrospective academic prototype. It is not investment, lending, or financial advice.")

@st.cache_resource(show_spinner=True)
def train_models(X, y):
    return evaluate_models(X, y)

try:
    _initial_metrics, fitted = train_models(X, y)
except Exception as exc:
    st.error(f"Training failed: {exc}")
    st.stop()

# Rebuild the table from the exact stored probabilities and labels used by the
# confusion matrix. This makes the reported metrics and matrix consistent.
metric_rows = []
for model_name, result in fitted.items():
    row, _ = metrics_for_threshold(model_name, result, threshold=.5)
    metric_rows.append(row)
metrics = pd.DataFrame(metric_rows).sort_values(
    ["PR-AUC", "F1", "ROC-AUC"], ascending=False, na_position="last"
).reset_index(drop=True)

class_counts = pd.Series(y).value_counts().reindex([0, 1], fill_value=0)
normal_count, bankrupt_count = int(class_counts.loc[0]), int(class_counts.loc[1])

overview, evaluation, assessment = st.tabs(["Overview", "Model Evaluation", "Company Assessment"])
with overview:
    a, b, c, d = st.columns(4)
    a.metric("Observations", f"{len(X):,}")
    b.metric("Financial features", str(X.shape[1]))
    c.metric("Non-bankrupt records", f"{normal_count:,}")
    d.metric("Bankrupt records", f"{bankrupt_count:,}")
    st.markdown("### System pipeline")
    st.markdown("**UCI data → cohort selection → missing-value imputation → LR / RF / XGBoost → risk probability → explanation**")
    st.markdown("### Demonstrated capabilities")
    st.write("- Cohort-specific dataset loading and missing-value handling")
    st.write("- Comparison of Logistic Regression, Random Forest and XGBoost where installed")
    st.write("- Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC and PR-AUC")
    st.write("- Company-level probability prediction and explicitly illustrative risk band")
    st.write("- SHAP explanation when available, with a clearly labelled sensitivity fallback")
    st.markdown("### Class distribution")
    distribution = pd.DataFrame({"Class": ["0 — Non-bankrupt", "1 — Bankrupt"], "Records": [normal_count, bankrupt_count]})
    st.bar_chart(distribution.set_index("Class"))
    st.caption("The target is imbalanced; high accuracy alone can be misleading.")

with evaluation:
    st.markdown("### Holdout performance")
    st.caption("The table below uses a probability threshold of 0.50. Positive class = bankrupt (1). All metrics are derived from the same stored holdout predictions used below.")
    display = metrics.copy()
    for col in ["Accuracy", "Balanced accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]:
        display[col] = display[col].map(lambda value: f"{value:.3f}" if pd.notna(value) else "N/A")
    display["Threshold"] = display["Threshold"].map(lambda value: f"{value:.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    best_name = metrics.iloc[0]["Model"]
    best = fitted[best_name]
    st.info(f"Baseline selection rule: highest PR-AUC, then F1, then ROC-AUC. Selected model: **{best_name}**.")
    threshold = st.slider("Decision threshold for the confusion matrix", min_value=0.05, max_value=0.95, value=0.50, step=0.05,
                          help="Lower thresholds generally flag more companies as at risk; this may increase recall but also false positives.")
    threshold_row, threshold_predictions = metrics_for_threshold(best_name, best, threshold=threshold)
    st.markdown("### Selected model at chosen threshold")
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Precision", f"{threshold_row['Precision']:.3f}")
    mc2.metric("Recall", f"{threshold_row['Recall']:.3f}")
    mc3.metric("F1-score", f"{threshold_row['F1']:.3f}")
    mc4.metric("Balanced accuracy", f"{threshold_row['Balanced accuracy']:.3f}")

    cm = confusion_matrix(best["y_test"], threshold_predictions, labels=[0, 1])
    cm_frame = pd.DataFrame(cm, index=["Actual: non-bankrupt (0)", "Actual: bankrupt (1)"],
                            columns=["Predicted: non-bankrupt (0)", "Predicted: bankrupt (1)"])
    st.markdown("### Confusion matrix")
    st.dataframe(cm_frame, use_container_width=True)
    st.caption("Confusion matrix and threshold-specific metrics are calculated from the same holdout labels and predictions. The holdout is a baseline evaluation, not proof of future-market performance.")
    
    st.markdown("### ROC and precision–recall analysis")

    # ROC curve: compare the true-positive and false-positive rates.
    fig_roc, ax_roc = plt.subplots(figsize=(7, 4.5))

    for model_name, result in fitted.items():
        y_true = np.asarray(result["y_test"])
        probabilities = np.asarray(result["prob"])

        fpr, tpr, _ = roc_curve(y_true, probabilities)
        auc_value = roc_auc_score(y_true, probabilities)

        ax_roc.plot(
            fpr, tpr,
            label=f"{model_name} (AUC = {auc_value:.3f})"
        )

    ax_roc.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC curve comparison")
    ax_roc.legend()
    ax_roc.grid(alpha=0.25)
    fig_roc.tight_layout()
    st.pyplot(fig_roc)
    plt.close(fig_roc)

    # Precision–recall curve: particularly useful for imbalanced data.
    fig_pr, ax_pr = plt.subplots(figsize=(7, 4.5))

    for model_name, result in fitted.items():
        y_true = np.asarray(result["y_test"])
        probabilities = np.asarray(result["prob"])

        precision_values, recall_values, _ = precision_recall_curve(
            y_true, probabilities
        )
        ap_value = average_precision_score(y_true, probabilities)

        ax_pr.plot(
            recall_values, precision_values,
            label=f"{model_name} (AP = {ap_value:.3f})"
        )

    positive_rate = float(np.mean(np.asarray(best["y_test"])))
    ax_pr.axhline(
        positive_rate,
        linestyle="--",
        label=f"Positive-class baseline = {positive_rate:.3f}"
    )
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision–recall curve comparison")
    ax_pr.legend()
    ax_pr.grid(alpha=0.25)
    fig_pr.tight_layout()
    st.pyplot(fig_pr)
    plt.close(fig_pr)

    st.caption(
        "ROC-AUC measures ranking discrimination. Average precision "
        "summarises the precision–recall curve. Both should be interpreted "
        "alongside the confusion matrix and the class imbalance."
    )

    

with assessment:
    st.markdown("### Assess a held-out company record")
    best_name = metrics.iloc[0]["Model"]
    best = fitted[best_name]
    X_test = best["X_test"]
    row_idx = st.slider("Select held-out record", min_value=0, max_value=max(0, len(X_test) - 1), value=0)
    X_one = X_test.iloc[[row_idx]].copy()
    st.write("**Selected record (first 12 financial features)**")
    st.dataframe(X_one.iloc[:, :12], use_container_width=True)
    if st.button("Predict risk and explain", type="primary"):
        probability = float(best["pipeline"].predict_proba(X_one)[:, 1][0])
        band = risk_band(probability)
        c1, c2 = st.columns(2)
        c1.metric("Illustrative risk band", band)
        c2.metric("Predicted bankruptcy probability", f"{probability:.1%}")
        st.progress(min(max(probability, 0), 1))
        st.caption("Illustrative bands: <30% Low, 30–60% Medium, ≥60% High. These are not calibrated ratings.")
        with st.spinner("Preparing feature explanation..."):
            explanation, method = explain_prediction(best["pipeline"], X_one, X_test)
        st.markdown(f"### Feature explanation — {method}")
        st.caption("Explanations describe model behaviour, not causation. A sensitivity fallback is not a SHAP explanation.")
        explanation_display = explanation[["Feature", "Contribution"]].copy()
        st.dataframe(explanation_display, use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(9, max(3.5, len(explanation_display) * .32)))
        values = explanation_display["Contribution"].to_numpy()
        colors = ["#238b57" if value < 0 else "#c74c4c" for value in values]
        ax.barh(explanation_display["Feature"].astype(str)[::-1], values[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_xlabel("Contribution / sensitivity")
        ax.set_title("Most influential features for selected record")
        fig.tight_layout()
        st.pyplot(fig)
        st.warning("Academic demonstration only. Do not use for real investment or lending decisions.")

st.divider()
st.caption("FinSight AI | Devanshi Singh (24BDE0110) • Suhani Sharma (24BDE0115)")
