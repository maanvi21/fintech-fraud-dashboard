"""
=============================================================================
Insurance Fraud Detection — Isolation Forest (Unsupervised)
=============================================================================

WHY ISOLATION FOREST IGNORES claim_status:
  Isolation Forest is fully unsupervised. It never sees any label column.
  It learns what "normal" looks like from the feature distributions alone,
  then flags records that are statistically hard to explain — i.e., anomalies.
  This makes it ideal for real-time scoring where labels don't exist yet.

OUTPUT:
  - fraud_isolation_forest_results.csv  →  original df + anomaly_score + fraud_flag
  - isolation_forest_model.pkl          →  serialized pipeline (scaler + model)

=============================================================================
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1. CONFIGURATION
# =============================================================================

DATA_PATH   = "final_fraud_dataset.csv"   # ← update path if needed
OUTPUT_CSV  = "fraud_isolation_forest_results.csv"
MODEL_PATH  = "isolation_forest_model.pkl"
RANDOM_SEED = 42

# Features that carry genuine fraud signal.
# Excludes ID columns, dates, free-text, and postal/routing numbers
# (those are identifiers, not behavioural signals).
FEATURE_COLS = [
    "claim_amount",
    "premium_amount",
    "policy_age_days",
    "tenure",                    # customer_tenure_days equivalent
    "report_delay_days",
    "incident_hour_of_the_day",  # incident_hour equivalent
    "vendor_claim_count",
    "processing_delay",
    "agent_experience_days",
    "age",
    "no_of_family_members",
    "any_injury",
    "police_report_available",
]


# =============================================================================
# 2. DATA LOADING
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """Load dataset and return a clean copy."""
    print(f"\n{'='*60}")
    print("STEP 1 — Loading data")
    print(f"{'='*60}")

    df = pd.read_csv(path)
    print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
    return df


# =============================================================================
# 3. FEATURE SELECTION — numerical only, no labels
# =============================================================================

def select_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Keep only the numerical feature columns defined above.
    Falls back gracefully if any column is missing from this dataset version.
    NOTE: claim_status is deliberately excluded — IF is unsupervised.
    """
    print(f"\n{'='*60}")
    print("STEP 2 — Feature selection (numerical only)")
    print(f"{'='*60}")

    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]

    if missing:
        print(f"  ⚠  Columns not found (skipped): {missing}")

    # Also auto-pick any remaining numeric columns not already in list
    # (catches engineered columns added later without editing this script)
    auto_numeric = [
        c for c in df.select_dtypes(include="number").columns
        if c not in available and c not in [
            "postal_code_x", "postal_code_y", "postal_code",
            "routing_number", "emp_routing_number"   # identifiers, not signals
        ]
    ]
    if auto_numeric:
        print(f"  ℹ  Auto-added numeric columns: {auto_numeric}")
        available += auto_numeric

    print(f"  Features selected ({len(available)}): {available}")
    return df[available].copy()


# =============================================================================
# 4. MISSING VALUE HANDLING
# =============================================================================

def handle_missing(X: pd.DataFrame) -> pd.DataFrame:
    """
    Median imputation — robust to outliers (important for fraud data
    where extreme values are intentional signals, not errors).
    """
    print(f"\n{'='*60}")
    print("STEP 3 — Missing value handling")
    print(f"{'='*60}")

    null_counts = X.isnull().sum()
    has_nulls   = null_counts[null_counts > 0]

    if has_nulls.empty:
        print("  ✓ No missing values found.")
    else:
        print(f"  Imputing {len(has_nulls)} column(s) with median:")
        for col, cnt in has_nulls.items():
            print(f"    {col}: {cnt} nulls")
        X = X.fillna(X.median(numeric_only=True))

    return X


# =============================================================================
# 5. CONTAMINATION ESTIMATION
# =============================================================================

def estimate_contamination(df_full: pd.DataFrame) -> float:
    """
    Estimate contamination (expected fraud rate) intelligently.

    Strategy (in priority order):
      1. Use claim_status if present: 'D' (Denied) ≈ fraud proxy.
         This gives a data-driven prior even though IF itself is unsupervised.
      2. Fall back to industry benchmark: 5–10% insurance fraud rate → 0.07.

    The contamination param tells IF roughly how many anomalies to expect,
    which calibrates the decision threshold. It does NOT feed into training.
    """
    print(f"\n{'='*60}")
    print("STEP 4 — Contamination estimation")
    print(f"{'='*60}")

    if "claim_status" in df_full.columns:
        # 'D' = Denied is the closest available fraud proxy label
        fraud_proxy = (df_full["claim_status"] == "D").mean()
        # Clamp between 0.01 and 0.30 (sklearn hard limits)
        contamination = float(np.clip(fraud_proxy, 0.01, 0.30))
        print(f"  claim_status 'D' rate   : {fraud_proxy:.4f} ({fraud_proxy*100:.2f}%)")
        print(f"  Contamination set to    : {contamination:.4f}")
        print("  ℹ  Note: IF does NOT train on claim_status — this only")
        print("     calibrates the anomaly threshold, nothing more.")
    else:
        contamination = 0.07
        print(f"  claim_status not found → using industry benchmark: {contamination}")

    return contamination


# =============================================================================
# 6. TRAIN ISOLATION FOREST
# =============================================================================

def train_isolation_forest(X_scaled: np.ndarray, contamination: float) -> IsolationForest:
    """
    Train Isolation Forest.

    Key hyperparameters:
      n_estimators=200    : more trees → more stable anomaly scores
      max_samples='auto'  : uses min(256, n_samples) — fast and effective
      contamination       : data-driven threshold from step 4
      random_state        : reproducibility
    """
    print(f"\n{'='*60}")
    print("STEP 5 — Training Isolation Forest")
    print(f"{'='*60}")

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        max_features=1.0,        # use all features per tree
        bootstrap=False,
        n_jobs=-1,               # use all CPU cores
        random_state=RANDOM_SEED,
        verbose=0,
    )

    model.fit(X_scaled)
    print("  ✓ Model trained successfully.")
    return model


# =============================================================================
# 7. PREDICT & ATTACH RESULTS
# =============================================================================

def predict_and_attach(df_full: pd.DataFrame, X_scaled: np.ndarray,
                        model: IsolationForest) -> pd.DataFrame:
    """
    Attach anomaly_score and fraud_flag to the original dataframe.

    sklearn IsolationForest convention:
      decision_function() → negative = more anomalous
      predict()           → -1 = anomaly, +1 = normal

    We convert:
      predict() == -1  →  fraud_flag = 1  (fraud)
      predict() == +1  →  fraud_flag = 0  (normal)
    """
    print(f"\n{'='*60}")
    print("STEP 6 — Generating predictions")
    print(f"{'='*60}")

    # Raw anomaly scores (lower = more anomalous)
    df_full["anomaly_score"] = model.decision_function(X_scaled)

    # Binary fraud flag: convert sklearn's -1/+1 to intuitive 1/0
    raw_preds             = model.predict(X_scaled)          # -1 or +1
    df_full["fraud_flag"] = np.where(raw_preds == -1, 1, 0)  # 1=fraud, 0=normal

    return df_full


# =============================================================================
# 8. RESULTS SUMMARY
# =============================================================================

def print_summary(df_full: pd.DataFrame) -> None:
    """Print detection summary and cross-tab against claim_status if available."""
    print(f"\n{'='*60}")
    print("STEP 7 — Results Summary")
    print(f"{'='*60}")

    n_total   = len(df_full)
    n_fraud   = df_full["fraud_flag"].sum()
    n_normal  = n_total - n_fraud
    pct_fraud = (n_fraud / n_total) * 100

    print(f"  Total records     : {n_total:,}")
    print(f"  Flagged as fraud  : {n_fraud:,}  ({pct_fraud:.2f}%)")
    print(f"  Flagged as normal : {n_normal:,}")

    # Cross-tab vs claim_status proxy (informational only — IF never saw this)
    if "claim_status" in df_full.columns:
        print(f"\n  Cross-tab vs claim_status (proxy label):")
        ct = pd.crosstab(
            df_full["claim_status"],
            df_full["fraud_flag"],
            rownames=["claim_status"],
            colnames=["fraud_flag (IF)"],
            margins=True
        )
        print(ct.to_string(index=True))
        print("\n  ℹ  'D'=Denied (proxy fraud), 'A'=Approved (proxy normal)")
        print("     IF was trained WITHOUT this label — overlap shows signal quality.")

    # Anomaly score distribution
    print(f"\n  Anomaly score stats:")
    print(df_full["anomaly_score"].describe().round(4).to_string())


# =============================================================================
# 9. FEATURE IMPORTANCE (via anomaly score correlation)
# =============================================================================

def feature_importance(X: pd.DataFrame, df_full: pd.DataFrame) -> None:
    """
    Isolation Forest has no native feature_importances_.
    We approximate importance by Pearson correlation of each feature
    with the anomaly score (negative score = more anomalous).

    Features with high NEGATIVE correlation with anomaly_score drive fraud flags.
    """
    print(f"\n{'='*60}")
    print("STEP 8 — Feature Importance (correlation with anomaly score)")
    print(f"{'='*60}")

    corr = (
        X.copy()
        .assign(anomaly_score=df_full["anomaly_score"].values)
        .corr()["anomaly_score"]
        .drop("anomaly_score")
        .sort_values()           # most negative = strongest fraud driver at top
    )

    print("\n  Feature correlation with anomaly_score")
    print("  (negative = pushes toward fraud flag, positive = pushes toward normal)\n")
    print(f"  {'Feature':<30} {'Correlation':>12}")
    print(f"  {'-'*44}")
    for feat, val in corr.items():
        bar = "▓" * int(abs(val) * 20)
        direction = "← fraud" if val < 0 else "→ normal"
        print(f"  {feat:<30} {val:>+.4f}   {bar} {direction}")


# =============================================================================
# 10. SAVE OUTPUTS
# =============================================================================

def save_outputs(df_full: pd.DataFrame, scaler: StandardScaler,
                  model: IsolationForest) -> None:
    """Save results CSV and serialized pipeline for serving."""
    print(f"\n{'='*60}")
    print("STEP 9 — Saving outputs")
    print(f"{'='*60}")

    # Results
    df_full.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Results saved  → {OUTPUT_CSV}")

    # Serialized pipeline (scaler + model together for safe serving)
    pipeline = {"scaler": scaler, "model": model}
    joblib.dump(pipeline, MODEL_PATH)
    print(f"  ✓ Model saved    → {MODEL_PATH}")


# =============================================================================
# 11. INFERENCE FUNCTION (FastAPI / Kafka ready)
# =============================================================================

def score_single_record(record: dict, pipeline_path: str = MODEL_PATH) -> dict:
    """
    Score a single insurance claim record in real-time.
    Designed for FastAPI endpoint or Kafka consumer.

    Args:
        record       : dict of feature key-value pairs (one claim)
        pipeline_path: path to saved joblib pipeline

    Returns:
        dict with anomaly_score and fraud_flag

    FastAPI usage:
        @app.post("/score")
        def score(claim: ClaimSchema):
            return score_single_record(claim.dict())

    Kafka usage:
        for msg in consumer:
            result = score_single_record(json.loads(msg.value))
            producer.send("fraud-alerts", result)
    """
    pipeline    = joblib.load(pipeline_path)
    scaler      = pipeline["scaler"]
    model       = pipeline["model"]

    # Build a single-row DataFrame matching training feature order
    feature_order = scaler.feature_names_in_
    X = pd.DataFrame([record])[feature_order].fillna(
        pd.Series(scaler.mean_, index=feature_order)  # median fallback at serve time
    )

    X_scaled       = scaler.transform(X)
    anomaly_score  = float(model.decision_function(X_scaled)[0])
    raw_pred       = model.predict(X_scaled)[0]
    fraud_flag     = 1 if raw_pred == -1 else 0

    return {
        "anomaly_score": round(anomaly_score, 6),
        "fraud_flag"   : fraud_flag,
        "risk_level"   : "HIGH" if fraud_flag == 1 else "NORMAL",
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    # --- Load ---
    df = load_data(DATA_PATH)

    # --- Features (no label column) ---
    X = select_features(df, FEATURE_COLS)

    # --- Missing values ---
    X = handle_missing(X)

    # --- Contamination (uses claim_status as calibration proxy only) ---
    contamination = estimate_contamination(df)

    # --- Scale ---
    print(f"\n{'='*60}")
    print("STEP 4b — Scaling features (StandardScaler)")
    print(f"{'='*60}")
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("  ✓ Features scaled.")

    # --- Train ---
    model = train_isolation_forest(X_scaled, contamination)

    # --- Predict ---
    df = predict_and_attach(df, X_scaled, model)

    # --- Summary ---
    print_summary(df)

    # --- Feature importance ---
    feature_importance(X, df)

    # --- Save ---
    save_outputs(df, scaler, model)

    print(f"\n{'='*60}")
    print("✅ Isolation Forest training complete.")
    print(f"   Next step: train XGBoost on the same data using")
    print(f"   fraud_flag from this output as a soft label,")
    print(f"   then combine both in the ensemble script.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()