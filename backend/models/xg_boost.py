"""
XGBoost Fraud Detection v2 — Insurance Claims
============================================================
Improvements over v1:
  1. Categorical features encoded (insurance_type, incident_severity,
     employment_status, risk_segmentation, social_class, house_type,
     marital_status, customer_education_level, authority_contacted)
  2. Interaction features added (delay_x_amount, high_claim_no_injury,
     vendor_concentration, claim_to_premium_ratio)
  3. Mild SMOTE (sampling_strategy=0.3) instead of full 1:1 oversampling
  4. scale_pos_weight used alongside SMOTE for better calibration
  5. Threshold tuned for precision @ min 40% recall (not raw F1)
  6. Full feature set saved to model pickle for inference

OUTPUT FILES (saved in same folder as this script):
  - xgb_model_v2.pkl
  - xgb_fraud_results_v2.csv
  - xgb_v2_summary.json
  - xgb_v2_performance_dashboard.png
  - xgb_v2_score_distribution.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Paths — all relative to THIS script's directory ──────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "final_fraud_dataset.csv")
IF_RESULTS = os.path.join(BASE_DIR, "fraud_isolation_forest_results.csv")
OUT_CSV    = os.path.join(BASE_DIR, "xgb_fraud_results_v2.csv")
OUT_MODEL  = os.path.join(BASE_DIR, "xgb_model_v2.pkl")
OUT_DIR    = BASE_DIR   # all plots + JSON saved next to the script

print(f"\n  📁 Working directory : {BASE_DIR}")
print(f"  📄 Data path         : {DATA_PATH}")
print(f"  💾 Model will save to: {OUT_MODEL}")

# ── Feature lists ─────────────────────────────────────────────────────────────
NUM_FEATURES = [
    "claim_amount", "premium_amount", "policy_age_days", "tenure",
    "report_delay_days", "incident_hour_of_the_day", "vendor_claim_count",
    "processing_delay", "agent_experience_days", "age",
    "no_of_family_members", "any_injury", "police_report_available",
]

CAT_FEATURES = [
    "insurance_type", "incident_severity", "employment_status",
    "risk_segmentation", "social_class", "house_type",
    "marital_status", "customer_education_level", "authority_contacted",
]

# Minimum recall floor for precision-tuned threshold
MIN_RECALL = 0.40

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 1 — Loading data")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — Building labels")
print("=" * 60)

df["label"] = (df["claim_status"] == "D").astype(int)
print(f"  Fraud: {df['label'].sum()}  Normal: {(df['label']==0).sum()}  ({df['label'].mean()*100:.2f}% fraud)")

if os.path.exists(IF_RESULTS):
    if_df = pd.read_csv(IF_RESULTS)
    if "fraud_flag" in if_df.columns and len(if_df) == len(df):
        df["if_flag"] = if_df["fraud_flag"].values
        df["label_soft"] = ((df["label"] == 1) | (df["if_flag"] == 1)).astype(int)
        print(f"  ✓ IF results merged  |  Soft-label fraud: {df['label_soft'].sum()}")
        TARGET = "label_soft"
    else:
        print("  ⚠  IF file mismatch — using claim_status only")
        TARGET = "label"
else:
    print("  ℹ  No IF results found — using claim_status label only")
    TARGET = "label"

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Categorical encoding")
print("=" * 60)

cat_cols_present = [c for c in CAT_FEATURES if c in df.columns]
missing_cats     = [c for c in CAT_FEATURES if c not in df.columns]
if missing_cats:
    print(f"  ⚠  Columns not found (skipped): {missing_cats}")

for col in cat_cols_present:
    df[col] = df[col].fillna("Unknown").astype(str)

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
cat_encoded = enc.fit_transform(df[cat_cols_present])
cat_df = pd.DataFrame(cat_encoded, columns=cat_cols_present, index=df.index)
print(f"  ✓ Encoded {len(cat_cols_present)} categorical features: {cat_cols_present}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 4 — Interaction features")
print("=" * 60)

df["claim_to_premium_ratio"] = df["claim_amount"] / (df["premium_amount"] + 1)
df["delay_x_amount"]         = df["report_delay_days"] * df["claim_amount"]
df["high_claim_no_injury"]   = (
    (df["claim_amount"] > df["claim_amount"].quantile(0.75)) &
    (df["any_injury"] == 0)
).astype(int)
df["vendor_concentration"]   = df["vendor_claim_count"] / (df["tenure"] + 1)

INTERACTION_FEATURES = [
    "claim_to_premium_ratio", "delay_x_amount",
    "high_claim_no_injury", "vendor_concentration",
]
print(f"  ✓ Added interaction features: {INTERACTION_FEATURES}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 5 — Assembling feature matrix")
print("=" * 60)

ALL_FEATURES = NUM_FEATURES + cat_cols_present + INTERACTION_FEATURES

num_df = df[NUM_FEATURES + INTERACTION_FEATURES].copy()
X_full = pd.concat([num_df, cat_df], axis=1)[ALL_FEATURES]
y      = df[TARGET].copy()

for col in NUM_FEATURES + INTERACTION_FEATURES:
    if X_full[col].isna().any():
        X_full[col].fillna(X_full[col].median(), inplace=True)
        print(f"  Imputed NaN in: {col}")

print(f"  Total features : {len(ALL_FEATURES)}")
print(f"  Numerical      : {len(NUM_FEATURES + INTERACTION_FEATURES)}")
print(f"  Categorical    : {len(cat_cols_present)}")
print(f"  Samples        : {len(X_full):,}  |  Fraud: {y.sum()} ({y.mean()*100:.2f}%)")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 6 — Train / Test split (stratified 80/20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"  Train fraud: {y_train.sum()}  |  Test fraud: {y_test.sum()}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 7 — Scaling (numerical + interaction only)")
print("=" * 60)

num_interaction_cols = NUM_FEATURES + INTERACTION_FEATURES
scaler = StandardScaler()

X_train_arr = X_train.values.copy().astype(float)
X_test_arr  = X_test.values.copy().astype(float)

num_idx = [ALL_FEATURES.index(c) for c in num_interaction_cols]
X_train_arr[:, num_idx] = scaler.fit_transform(X_train_arr[:, num_idx])
X_test_arr[:, num_idx]  = scaler.transform(X_test_arr[:, num_idx])
print(f"  ✓ Scaled {len(num_idx)} numerical/interaction columns")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 8 — Mild SMOTE (sampling_strategy=0.3)")
print("=" * 60)

smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.3)
X_res, y_res = smote.fit_resample(X_train_arr, y_train)
print(f"  Before → Fraud: {y_train.sum():,}  Normal: {(y_train==0).sum():,}")
print(f"  After  → Fraud: {y_res.sum():,}   Normal: {(y_res==0).sum():,}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 9 — XGBoost training (RandomizedSearchCV, precision-optimised)")
print("=" * 60)

neg_count = int((y_res == 0).sum())
pos_count = int(y_res.sum())
spw       = round(neg_count / pos_count, 2)
print(f"  scale_pos_weight = {spw}")

base_xgb = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=spw,
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
)

param_dist = {
    "n_estimators"    : [100, 200, 300],
    "max_depth"       : [3, 5, 7],
    "learning_rate"   : [0.05, 0.1, 0.2],
    "subsample"       : [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma"           : [0, 0.1, 0.3],
    "reg_alpha"       : [0, 0.1, 0.5],
    "reg_lambda"      : [1, 1.5, 2],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    base_xgb,
    param_dist,
    n_iter=20,
    scoring="average_precision",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    refit=True,
    random_state=42,
)

search.fit(X_res, y_res)
best_model = search.best_estimator_

print(f"\n  ✓ Best params  : {search.best_params_}")
print(f"  ✓ Best CV AUPRC: {search.best_score_:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"STEP 10 — Threshold tuning (max precision @ recall ≥ {MIN_RECALL})")
print("=" * 60)

y_prob = best_model.predict_proba(X_test_arr)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

mask = recalls[:-1] >= MIN_RECALL
if mask.any():
    best_idx = np.where(mask)[0][np.argmax(precisions[:-1][mask])]
    best_thr = float(thresholds[best_idx])
else:
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx  = int(np.argmax(f1_scores))
    best_thr  = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    print(f"  ⚠  No threshold met recall ≥ {MIN_RECALL} — falling back to max F1")

print(f"  Best threshold : {best_thr:.4f}")
print(f"  Precision      : {precisions[best_idx]:.4f}")
print(f"  Recall         : {recalls[best_idx]:.4f}")

y_pred = (y_prob >= best_thr).astype(int)

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 11 — Evaluation")
print("=" * 60)

roc_auc = roc_auc_score(y_test, y_prob)
pr_auc  = average_precision_score(y_test, y_prob)
cm      = confusion_matrix(y_test, y_pred)
report  = classification_report(y_test, y_pred, target_names=["Normal", "Fraud"])

print(f"\n  ROC-AUC  : {roc_auc:.4f}")
print(f"  PR-AUC   : {pr_auc:.4f}")
print(f"\n  Confusion Matrix:\n{cm}")
print(f"\n  Classification Report:\n{report}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 12 — Feature importance")
print("=" * 60)

importances = pd.Series(
    best_model.feature_importances_, index=ALL_FEATURES
).sort_values(ascending=False)

print(f"\n  {'Feature':<35} {'Importance':>10}")
print("  " + "-" * 47)
for feat, imp in importances.items():
    bar = "▓" * int(imp * 60)
    print(f"  {feat:<35} {imp:>8.4f}  {bar}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 13 — Saving plots")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("XGBoost Fraud Detection v2 — Performance Dashboard",
             fontsize=14, fontweight="bold")

ax = axes[0, 0]
fpr, tpr, _ = roc_curve(y_test, y_prob)
ax.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(recalls, precisions, color="#16a34a", lw=2, label=f"PR (AUC={pr_auc:.3f})")
ax.axvline(recalls[best_idx], color="red", ls="--", lw=1.2,
           label=f"Threshold={best_thr:.2f}")
ax.axhline(precisions[best_idx], color="orange", ls=":", lw=1.2,
           label=f"Precision={precisions[best_idx]:.2f}")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve"); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1, 0]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix (thr={best_thr:.2f})")

ax = axes[1, 1]
top15  = importances.head(15)
colors = ["#dc2626" if v > top15.median() else "#60a5fa" for v in top15.values]
ax.barh(top15.index[::-1], top15.values[::-1], color=colors[::-1])
ax.set_xlabel("Feature Importance (gain)")
ax.set_title("Top 15 Features"); ax.grid(alpha=0.3, axis="x")

plt.tight_layout()
dashboard_path = os.path.join(OUT_DIR, "xgb_v2_performance_dashboard.png")
plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Dashboard saved → {dashboard_path}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(y_prob[y_test == 0], bins=50, alpha=0.6, color="#3b82f6", label="Normal")
ax.hist(y_prob[y_test == 1], bins=50, alpha=0.7, color="#ef4444", label="Fraud")
ax.axvline(best_thr, color="black", ls="--", lw=1.5,
           label=f"Threshold ({best_thr:.2f})")
ax.set_xlabel("Fraud Probability"); ax.set_ylabel("Count")
ax.set_title("Score Distribution — Normal vs Fraud")
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
score_dist_path = os.path.join(OUT_DIR, "xgb_v2_score_distribution.png")
plt.savefig(score_dist_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Score dist saved → {score_dist_path}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 14 — Saving outputs")
print("=" * 60)

X_full_arr = X_full.values.copy().astype(float)
X_full_arr[:, num_idx] = scaler.transform(X_full_arr[:, num_idx])

df["xgb_fraud_prob"] = best_model.predict_proba(X_full_arr)[:, 1]
df["xgb_fraud_flag"] = (df["xgb_fraud_prob"] >= best_thr).astype(int)
df["ground_truth"]   = df["label"]

df.to_csv(OUT_CSV, index=False)
print(f"  ✓ Results CSV saved  → {OUT_CSV}")
print(f"  Flagged fraud: {df['xgb_fraud_flag'].sum()} / {len(df)}")

def to_py(obj):
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, dict): return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_py(v) for v in obj]
    return obj

model_bundle = {
    "model"            : best_model,
    "scaler"           : scaler,
    "encoder"          : enc,
    "threshold"        : best_thr,
    "all_features"     : ALL_FEATURES,
    "num_features"     : NUM_FEATURES,
    "cat_features"     : cat_cols_present,
    "interaction_feats": INTERACTION_FEATURES,
    "num_idx"          : num_idx,
    "best_params"      : search.best_params_,
}
with open(OUT_MODEL, "wb") as f:
    pickle.dump(model_bundle, f)
print(f"  ✓ Model bundle saved → {OUT_MODEL}")

summary = {
    "roc_auc"         : round(roc_auc, 4),
    "pr_auc"          : round(pr_auc, 4),
    "best_threshold"  : round(best_thr, 4),
    "precision_fraud" : round(float(precisions[best_idx]), 4),
    "recall_fraud"    : round(float(recalls[best_idx]), 4),
    "total_features"  : len(ALL_FEATURES),
    "best_params"     : search.best_params_,
    "total_records"   : len(df),
    "flagged_fraud"   : int(df["xgb_fraud_flag"].sum()),
}
summary_path = os.path.join(OUT_DIR, "xgb_v2_summary.json")
with open(summary_path, "w") as f:
    json.dump(to_py(summary), f, indent=2)
print(f"  ✓ Summary JSON saved → {summary_path}")

print("\n" + "=" * 60)
print("✅ XGBoost v2 training complete.")
print(f"   ROC-AUC: {roc_auc:.4f}  |  PR-AUC: {pr_auc:.4f}")
print(f"   Fraud Precision: {precisions[best_idx]:.4f}  |  Recall: {recalls[best_idx]:.4f}")
print(f"\n   📁 All files saved to: {BASE_DIR}")
print(f"      • xgb_model_v2.pkl")
print(f"      • xgb_fraud_results_v2.csv")
print(f"      • xgb_v2_summary.json")
print(f"      • xgb_v2_performance_dashboard.png")
print(f"      • xgb_v2_score_distribution.png")
print("=" * 60 + "\n")