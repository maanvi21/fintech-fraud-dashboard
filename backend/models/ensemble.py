"""
Ensemble Fraud Detection — XGBoost + Isolation Forest
Goal: Maximize F1 (balance precision and recall equally)
=====================================================================
WHAT THIS SCRIPT DOES (plain English):
  1. Loads your XGBoost model (already trained)
  2. Re-trains Isolation Forest fresh from the data
  3. Combines both models in 6 different ways (strategies)
  4. For EACH strategy, tries 1000 different threshold values
     and picks the one with the best F1 score
  5. Compares all strategies and picks the overall winner
  6. Saves results, charts, and the best model

WHY THRESHOLD MATTERS:
  A threshold is the cutoff score above which we call something "fraud".
  - Low threshold (e.g. 0.2) → catches more fraud but also flags more
    innocent claims (high recall, low precision)
  - High threshold (e.g. 0.8) → only flags very suspicious claims but
    misses some real fraud (high precision, low recall)
  - Best F1 threshold → the sweet spot that balances both

OUTPUT FILES:
  - ensemble_results.csv
  - ensemble_summary.json
  - ensemble_performance_dashboard.png
  - threshold_search_curves.png       ← shows F1 vs threshold for all strategies
  - ensemble_model.pkl
  - isolation_forest_model_fixed.pkl
"""

import os, warnings, pickle, json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve, f1_score
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — change these if your file names differ
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH       = os.path.join(BASE_DIR, "final_fraud_dataset.csv")
XGB_MODEL_PATH  = os.path.join(BASE_DIR, "xgb_model_v2.pkl")
IF_MODEL_FIXED  = os.path.join(BASE_DIR, "isolation_forest_model_fixed.pkl")
OUT_CSV         = os.path.join(BASE_DIR, "ensemble_results.csv")
OUT_MODEL       = os.path.join(BASE_DIR, "ensemble_model.pkl")
OUT_DIR         = BASE_DIR
RANDOM_SEED     = 42

# How many threshold values to try per strategy
# 1000 means we try 0.001, 0.002, ... 0.999
N_THRESHOLDS    = 1000

print(f"\n  Working dir : {BASE_DIR}")
print(f"  Threshold search: trying {N_THRESHOLDS} values per strategy")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA AND XGBOOST MODEL
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1 — Loading data and XGBoost model")
print("="*60)

df = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

fraud_count  = (df["claim_status"] == "D").sum()
normal_count = len(df) - fraud_count
print(f"  Fraud claims  : {fraud_count:,}  ({fraud_count/len(df)*100:.1f}%)")
print(f"  Normal claims : {normal_count:,}  ({normal_count/len(df)*100:.1f}%)")
print(f"  → Imbalanced dataset: F1 matters more than accuracy here.")

with open(XGB_MODEL_PATH, "rb") as f:
    xgb_bundle = pickle.load(f)
print(f"\n  ✓ XGBoost model loaded")
print(f"    Number of features : {len(xgb_bundle['all_features'])}")
print(f"    Original threshold : {xgb_bundle['threshold']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2 — REBUILD XGBOOST FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2 — Rebuilding XGBoost features")
print("="*60)
print("  (Must match exactly how the model was originally trained)")

xgb_model         = xgb_bundle["model"]
xgb_scaler        = xgb_bundle["scaler"]
xgb_enc           = xgb_bundle["encoder"]
xgb_thr           = xgb_bundle["threshold"]
ALL_FEATURES      = xgb_bundle["all_features"]
NUM_FEATURES      = xgb_bundle["num_features"]
CAT_FEATURES      = xgb_bundle["cat_features"]
INTERACTION_FEATS = xgb_bundle["interaction_feats"]
num_idx           = xgb_bundle["num_idx"]

df["label"] = (df["claim_status"] == "D").astype(int)

for col in CAT_FEATURES:
    df[col] = df[col].fillna("Unknown").astype(str)
cat_encoded = xgb_enc.transform(df[CAT_FEATURES])
cat_df = pd.DataFrame(cat_encoded, columns=CAT_FEATURES, index=df.index)

df["claim_to_premium_ratio"] = df["claim_amount"] / (df["premium_amount"] + 1)
df["delay_x_amount"]         = df["report_delay_days"] * df["claim_amount"]
df["high_claim_no_injury"]   = (
    (df["claim_amount"] > df["claim_amount"].quantile(0.75)) &
    (df["any_injury"] == 0)
).astype(int)
df["vendor_concentration"] = df["vendor_claim_count"] / (df["tenure"] + 1)

num_df = df[NUM_FEATURES + INTERACTION_FEATS].copy()
X_full = pd.concat([num_df, cat_df], axis=1)[ALL_FEATURES]
for col in NUM_FEATURES + INTERACTION_FEATS:
    if X_full[col].isna().any():
        X_full[col].fillna(X_full[col].median(), inplace=True)

X_arr = X_full.values.copy().astype(float)
X_arr[:, num_idx] = xgb_scaler.transform(X_arr[:, num_idx])
print(f"  ✓ Feature matrix: {X_arr.shape[0]:,} rows × {X_arr.shape[1]} features")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3 — XGBOOST PROBABILITIES
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3 — XGBoost fraud probabilities")
print("="*60)

xgb_probs      = xgb_model.predict_proba(X_arr)[:, 1]
df["xgb_prob"] = xgb_probs
df["xgb_flag"] = (xgb_probs >= xgb_thr).astype(int)
print(f"  Score range: {xgb_probs.min():.4f} – {xgb_probs.max():.4f}  "
      f"(median: {np.median(xgb_probs):.4f})")
print(f"  Flagged at original threshold: {df['xgb_flag'].sum():,} claims")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4 — RETRAIN ISOLATION FOREST
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4 — Training Isolation Forest")
print("="*60)
print("  Unsupervised model — never sees labels.")
print("  Learns what 'normal' looks like, then flags statistical outliers.")

IF_FEATURE_COLS = [
    "claim_amount", "premium_amount", "policy_age_days", "tenure",
    "report_delay_days", "incident_hour_of_the_day", "vendor_claim_count",
    "processing_delay", "agent_experience_days", "age",
    "no_of_family_members", "any_injury", "police_report_available",
]
IF_FEATURES = [c for c in IF_FEATURE_COLS if c in df.columns]
missing_if  = [c for c in IF_FEATURE_COLS if c not in df.columns]
if missing_if:
    print(f"  ⚠  Skipping missing columns: {missing_if}")

X_if_df     = df[IF_FEATURES].copy().fillna(df[IF_FEATURES].median(numeric_only=True))
if_scaler   = StandardScaler()
X_if_scaled = if_scaler.fit_transform(X_if_df)

fraud_proxy   = (df["claim_status"] == "D").mean()
contamination = float(np.clip(fraud_proxy, 0.01, 0.30))
print(f"  Contamination: {contamination:.4f}  (derived from fraud rate)")

if_model = IsolationForest(
    n_estimators=200, max_samples="auto", contamination=contamination,
    max_features=1.0, bootstrap=False, n_jobs=-1,
    random_state=RANDOM_SEED, verbose=0,
)
if_model.fit(X_if_scaled)
joblib.dump({"scaler": if_scaler, "model": if_model, "features": IF_FEATURES},
            IF_MODEL_FIXED)

raw_scores          = if_model.decision_function(X_if_scaled)
df["if_score_raw"]  = -raw_scores   # negate: higher = more anomalous
mms = MinMaxScaler()
df["if_score_norm"] = mms.fit_transform(df[["if_score_raw"]]).ravel()
df["if_flag"]       = np.where(if_model.predict(X_if_scaled) == -1, 1, 0)

print(f"  ✓ Trained  |  IF flagged: {df['if_flag'].sum():,} claims  "
      f"|  Saved → isolation_forest_model_fixed.pkl")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5 — TRAIN / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5 — Train / test split  (80% train, 20% test)")
print("="*60)

y = df["label"].values
idx_train, idx_test = train_test_split(
    np.arange(len(df)), test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {len(idx_train):,} claims  (fraud: {y[idx_train].sum():,})")
print(f"  Test : {len(idx_test):,}  claims  (fraud: {y[idx_test].sum():,})")
print(f"  stratify=y ensures same fraud ratio in both splits")

xgb_prob_test = df["xgb_prob"].values[idx_test]
if_score_test = df["if_score_norm"].values[idx_test]
y_test        = y[idx_test]

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6 — STACKING META-LEARNER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 6 — Training stacking meta-learner")
print("="*60)

meta_X_train = np.column_stack([df["xgb_prob"].values[idx_train],
                                 df["if_score_norm"].values[idx_train]])
meta_X_test  = np.column_stack([xgb_prob_test, if_score_test])
y_train      = y[idx_train]

lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=500)
lr.fit(meta_X_train, y_train)
stack_probs = lr.predict_proba(meta_X_test)[:, 1]

xgb_w = abs(lr.coef_[0][0])
if_w  = abs(lr.coef_[0][1])
total = xgb_w + if_w
print(f"  LR learned: XGBoost {xgb_w/total*100:.0f}%  |  IF {if_w/total*100:.0f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7 — EXHAUSTIVE THRESHOLD SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print(f"STEP 7 — Threshold search ({N_THRESHOLDS} values × 6 strategies)")
print("="*60)

def find_best_threshold_f1(probs, labels, n_steps=N_THRESHOLDS):
    """
    Try n_steps evenly-spaced thresholds across the full score range.
    At each threshold, compute F1 on the fraud class.
    Return: best_threshold, best_f1, all_thresholds, all_f1_scores
    """
    thresholds = np.linspace(probs.min() + 1e-6, probs.max() - 1e-6, n_steps)
    best_f1    = -1.0
    best_thr   = 0.5
    f1_curve   = np.zeros(n_steps)

    for i, thr in enumerate(thresholds):
        preds = (probs >= thr).astype(int)
        if preds.sum() == 0:    # no fraud predicted — skip
            continue
        f1 = f1_score(labels, preds, zero_division=0)
        f1_curve[i] = f1
        if f1 > best_f1:
            best_f1  = f1
            best_thr = thr

    return best_thr, best_f1, thresholds, f1_curve

def evaluate(name, probs, labels):
    best_thr, best_f1, thr_grid, f1_curve = find_best_threshold_f1(probs, labels)
    preds = (probs >= best_thr).astype(int)
    roc   = roc_auc_score(labels, probs)
    pr    = average_precision_score(labels, probs)
    cm    = confusion_matrix(labels, preds)
    rep   = classification_report(labels, preds,
                target_names=["Normal", "Fraud"], output_dict=True)
    tn, fp, fn, tp = cm.ravel()

    print(f"  ┌─ {name}")
    print(f"  │  Threshold  : {best_thr:.4f}")
    print(f"  │  Precision  : {rep['Fraud']['precision']:.3f}  "
          f"({tp} correct flags / {tp+fp} total flags)")
    print(f"  │  Recall     : {rep['Fraud']['recall']:.3f}  "
          f"({tp} caught / {tp+fn} real fraud)")
    print(f"  └  F1         : {rep['Fraud']['f1-score']:.3f}\n")

    return {
        "name"     : name,
        "threshold": best_thr,
        "roc_auc"  : roc,
        "pr_auc"   : pr,
        "precision": rep["Fraud"]["precision"],
        "recall"   : rep["Fraud"]["recall"],
        "f1"       : rep["Fraud"]["f1-score"],
        "cm"       : cm,
        "probs"    : probs,
        "preds"    : preds,
        "thr_grid" : thr_grid,
        "f1_curve" : f1_curve,
    }

results = {}
results["XGBoost alone"]  = evaluate("XGBoost alone",
                                       xgb_prob_test, y_test)
results["Weighted 80/20"] = evaluate("Weighted (XGB 80% + IF 20%)",
                                       0.80*xgb_prob_test + 0.20*if_score_test, y_test)
results["Weighted 70/30"] = evaluate("Weighted (XGB 70% + IF 30%)",
                                       0.70*xgb_prob_test + 0.30*if_score_test, y_test)
results["Weighted 60/40"] = evaluate("Weighted (XGB 60% + IF 40%)",
                                       0.60*xgb_prob_test + 0.40*if_score_test, y_test)
results["OR rule"]        = evaluate("OR rule (max of XGB, IF)",
                                       np.maximum(xgb_prob_test, if_score_test), y_test)
results["Stacking LR"]    = evaluate("Stacking (LR meta-learner)",
                                       stack_probs, y_test)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8 — PICK THE WINNER
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 8 — Strategy ranking")
print("="*60)

ranked = sorted(results.values(), key=lambda x: x["f1"], reverse=True)
best   = ranked[0]

print(f"\n  {'Rank':<5} {'Strategy':<36} {'F1':>5}  {'Prec':>5}  {'Rec':>5}  {'Threshold':>9}")
print(f"  {'-'*70}")
for i, r in enumerate(ranked):
    tag = " ← BEST" if i == 0 else ""
    print(f"  {i+1:<5} {r['name']:<36} {r['f1']:>5.3f}  "
          f"{r['precision']:>5.3f}  {r['recall']:>5.3f}  "
          f"{r['threshold']:>9.4f}{tag}")

tn, fp, fn, tp = best["cm"].ravel()
print(f"\n  At threshold {best['threshold']:.4f} the best model:")
print(f"    ✓ Correctly caught  : {tp:>4} fraud cases")
print(f"    ✗ Missed            : {fn:>4} fraud cases  (false negatives)")
print(f"    ✗ False alarms      : {fp:>4} innocent claims flagged  (false positives)")
print(f"    ✓ Correctly cleared : {tn:>4} innocent claims")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9 — CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 9 — Saving charts")
print("="*60)

colors = ["#2563eb", "#16a34a", "#d97706", "#dc2626", "#7c3aed", "#0891b2"]

# Chart 1 — F1 vs Threshold curves (one panel per strategy)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(f"F1 Score vs Threshold — {N_THRESHOLDS}-point search\n"
             f"(dashed line = chosen threshold)",
             fontsize=13, fontweight="bold")

for ax, (name, res), col in zip(axes.flat, results.items(), colors):
    ax.plot(res["thr_grid"], res["f1_curve"], color=col, lw=2)
    ax.axvline(res["threshold"], color="black", ls="--", lw=1.8,
               label=f"thr={res['threshold']:.3f}  F1={res['f1']:.3f}")
    ax.fill_between(res["thr_grid"], res["f1_curve"], alpha=0.10, color=col)
    ax.set_title(res["name"].split("(")[0].strip(), fontsize=9, fontweight="bold")
    ax.set_xlabel("Threshold", fontsize=8)
    ax.set_ylabel("F1 (fraud class)", fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    ymax = max(res["f1_curve"]) if max(res["f1_curve"]) > 0 else 0.1
    ax.set_ylim(0, ymax * 1.25)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "threshold_search_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Threshold curves  → threshold_search_curves.png")

# Chart 2 — Performance dashboard
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(f"Ensemble Performance Dashboard\n"
             f"Winner: {best['name']}  "
             f"(F1={best['f1']:.3f}, P={best['precision']:.3f}, R={best['recall']:.3f})",
             fontsize=13, fontweight="bold")

# ROC
ax = axes[0, 0]
for (name, res), col in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res["probs"])
    ax.plot(fpr, tpr, color=col, lw=1.5,
            label=f"{res['name'].split('(')[0].strip()} ({res['roc_auc']:.3f})")
ax.plot([0,1],[0,1],"k--",lw=0.8)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (higher-right = better)")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Precision-Recall
ax = axes[0, 1]
for (name, res), col in zip(results.items(), colors):
    prec, rec, _ = precision_recall_curve(y_test, res["probs"])
    ax.plot(rec, prec, color=col, lw=1.5,
            label=f"{res['name'].split('(')[0].strip()} ({res['pr_auc']:.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves (higher = better)")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# F1 bar chart
ax = axes[0, 2]
names_s    = [r["name"].split("(")[0].strip()[:16] for r in ranked]
f1s        = [r["f1"] for r in ranked]
bar_colors = [colors[0] if r["name"] == best["name"] else "#cbd5e1" for r in ranked]
bars = ax.bar(names_s, f1s, color=bar_colors, edgecolor="white", alpha=0.9)
ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
ax.set_xticklabels(names_s, rotation=30, ha="right", fontsize=8)
ax.set_title("F1 Score by Strategy  (blue = winner)")
ax.set_ylim(0, max(f1s) * 1.2)
ax.grid(alpha=0.3, axis="y")

# Confusion matrix
ax = axes[1, 0]
sns.heatmap(best["cm"], annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Normal","Fraud"], yticklabels=["Normal","Fraud"],
            annot_kws={"size": 14})
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best['name'].split('(')[0].strip()}")

# Score distribution
ax = axes[1, 1]
ax.hist(best["probs"][y_test==0], bins=50, alpha=0.6, color="#3b82f6",
        label=f"Normal ({(y_test==0).sum():,})", density=True)
ax.hist(best["probs"][y_test==1], bins=50, alpha=0.7, color="#ef4444",
        label=f"Fraud ({(y_test==1).sum():,})", density=True)
ax.axvline(best["threshold"], color="black", ls="--", lw=2,
           label=f"Threshold = {best['threshold']:.3f}")
ax.set_xlabel("Fraud Score"); ax.set_ylabel("Density")
ax.set_title("Score Distribution\n(more overlap = harder problem)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# P / R / F1 grouped bars
ax = axes[1, 2]
x = np.arange(len(ranked)); w = 0.25
ax.bar(x-w, [r["precision"] for r in ranked], w,
       label="Precision", color="#2563eb", alpha=0.85)
ax.bar(x,   [r["recall"]    for r in ranked], w,
       label="Recall",    color="#16a34a", alpha=0.85)
ax.bar(x+w, [r["f1"]        for r in ranked], w,
       label="F1",        color="#d97706", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([r["name"].split("(")[0].strip()[:14]
                    for r in ranked], rotation=30, ha="right", fontsize=8)
ax.set_title("Precision / Recall / F1 per Strategy")
ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0, 1.1)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ensemble_performance_dashboard.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Performance dashboard → ensemble_performance_dashboard.png")

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10 — SCORE FULL DATASET AND SAVE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 10 — Scoring full dataset and saving outputs")
print("="*60)

name = best["name"]
if "80/20" in name:
    df["ensemble_score"] = 0.80*df["xgb_prob"] + 0.20*df["if_score_norm"]
elif "70/30" in name:
    df["ensemble_score"] = 0.70*df["xgb_prob"] + 0.30*df["if_score_norm"]
elif "60/40" in name:
    df["ensemble_score"] = 0.60*df["xgb_prob"] + 0.40*df["if_score_norm"]
elif "OR" in name:
    df["ensemble_score"] = np.maximum(df["xgb_prob"], df["if_score_norm"])
elif "Stacking" in name:
    full_meta = np.column_stack([df["xgb_prob"], df["if_score_norm"]])
    df["ensemble_score"] = lr.predict_proba(full_meta)[:, 1]
else:
    df["ensemble_score"] = df["xgb_prob"]

df["ensemble_flag"] = (df["ensemble_score"] >= best["threshold"]).astype(int)
df["risk_tier"]     = pd.cut(
    df["ensemble_score"],
    bins=[0, 0.3, 0.5, 0.7, 1.001],
    labels=["Low", "Medium", "High", "Critical"]
)

df.to_csv(OUT_CSV, index=False)
print(f"  ✓ Saved → ensemble_results.csv")
print(f"  Flagged as fraud : {df['ensemble_flag'].sum():,} / {len(df):,}")
print(f"\n  Risk tier breakdown:")
for tier, count in df["risk_tier"].value_counts().sort_index().items():
    bar = "█" * int(count / len(df) * 100 / 2)
    print(f"    {tier:<10}: {count:>5,}  ({count/len(df)*100:4.1f}%)  {bar}")

def to_py(obj):
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, dict): return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, list): return [to_py(v) for v in obj]
    return obj

bundle = {
    "strategy"  : best["name"],
    "threshold" : best["threshold"],
    "xgb_bundle": xgb_bundle,
    "if_scaler" : mms,
    "lr_meta"   : lr if "Stacking" in best["name"] else None,
}
with open(OUT_MODEL, "wb") as f:
    pickle.dump(bundle, f)
print(f"\n  ✓ Saved → ensemble_model.pkl")

summary = {
    "best_strategy"      : best["name"],
    "threshold"          : round(best["threshold"], 4),
    "roc_auc"            : round(best["roc_auc"],   4),
    "pr_auc"             : round(best["pr_auc"],    4),
    "fraud_precision"    : round(best["precision"], 4),
    "fraud_recall"       : round(best["recall"],    4),
    "fraud_f1"           : round(best["f1"],        4),
    "improvement_vs_xgb" : {
        "precision": round(best["precision"] - results["XGBoost alone"]["precision"], 4),
        "recall"   : round(best["recall"]    - results["XGBoost alone"]["recall"],    4),
        "f1"       : round(best["f1"]        - results["XGBoost alone"]["f1"],        4),
    },
    "all_strategies": [
        {"name"     : r["name"],
         "threshold": round(r["threshold"], 4),
         "f1"       : round(r["f1"],        4),
         "precision": round(r["precision"], 4),
         "recall"   : round(r["recall"],    4)}
        for r in ranked
    ],
}
with open(os.path.join(OUT_DIR, "ensemble_summary.json"), "w") as f:
    json.dump(to_py(summary), f, indent=2)
print(f"  ✓ Saved → ensemble_summary.json")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("✅  DONE")
print("="*60)
print(f"\n  Best strategy  : {best['name']}")
print(f"  Threshold      : {best['threshold']:.4f}")
print(f"  Fraud F1       : {best['f1']:.4f}")
print(f"  Fraud Precision: {best['precision']:.4f}  "
      f"(~{best['precision']*100:.0f} of every 100 flags are real fraud)")
print(f"  Fraud Recall   : {best['recall']:.4f}  "
      f"(catching ~{best['recall']*100:.0f}% of all fraud cases)")
print(f"  vs XGBoost F1  : {results['XGBoost alone']['f1']:.4f}  "
      f"(ensemble gain: {best['f1'] - results['XGBoost alone']['f1']:+.4f})")
print(f"\n  Files saved:")
print(f"    ensemble_results.csv               all claims with scores + risk tier")
print(f"    ensemble_summary.json              metrics for all strategies")
print(f"    ensemble_performance_dashboard.png ROC, PR, confusion matrix, dist")
print(f"    threshold_search_curves.png        F1 vs threshold for all strategies")
print(f"    ensemble_model.pkl                 model bundle for inference")
print(f"    isolation_forest_model_fixed.pkl   corrected IF model")
print("="*60 + "\n")