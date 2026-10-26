# FinSight AI — Explainable Company Bankruptcy Risk Prototype

Academic prototype that compares machine-learning models for company-bankruptcy classification using the UCI Polish Companies Bankruptcy dataset. The Streamlit dashboard shows evaluation metrics, a threshold-adjustable confusion matrix, a held-out company prediction, and a feature explanation when SHAP is available.

## Dataset and target

- Source: UCI Machine Learning Repository, **Polish Companies Bankruptcy** (dataset ID 365; DOI: 10.24432/C5F600).
- The loader keeps only the **3rd-year cohort** when the combined download includes a `year` field. This avoids mixing records whose target labels refer to different forecast horizons.
- Target convention: `0 = non-bankrupt`, `1 = bankrupt`.
- The dataset is highly imbalanced. Accuracy should not be interpreted alone; review recall, F1, ROC-AUC, PR-AUC and the confusion matrix.
- The low/medium/high probability bands are illustrative thresholds, not calibrated or validated ratings.

## Run on Windows

1. Extract the ZIP into a normal folder.
2. Open Command Prompt in that extracted folder (File Explorer address bar → type `cmd` → Enter).
3. Run `python --version` (Python 3.10+ recommended).
4. Run `run_windows.bat` and wait for installation and startup.
5. Open `http://localhost:8501` in your browser. Keep the terminal open while using the app.

If the UCI data download fails, the app switches to explicitly labelled synthetic demo data. **Never report synthetic-demo metrics as real research results.**

## Evaluation method

A stratified 75/25 holdout is used for a reproducible baseline. Models are ranked by average precision (PR-AUC), then F1, then ROC-AUC because bankrupt cases are rare. The evaluation table uses a 0.50 probability threshold. The confusion-matrix slider recalculates threshold-specific metrics and the matrix from the same stored predictions.

This is a baseline, not a production financial system. Further work should include repeated or time-aware validation, calibration analysis, threshold selection justified by costs, and comparison with published research.

## Citation

Tomczak, S. (2016). *Polish Companies Bankruptcy* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F600

Zieba, M., Tomczak, S. K., & Tomczak, J. M. (2016). Ensemble boosted trees with synthetic features generation in application to bankruptcy prediction. *Expert Systems with Applications, 58*, 93–101. https://doi.org/10.1016/j.eswa.2016.03.006
