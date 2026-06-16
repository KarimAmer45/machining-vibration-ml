"""Train and compare pure ML and physics-guided milling wear models.

Improvements:
- 5-fold group-aware cross-validation: metrics are mean +/- std across held-out runs.
- GradientBoosting added alongside RandomForest for comparison.
- CV summary bar chart saved as cv_comparison.png.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

# Reuse the shared feature and training modules from the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.append(str(REPOSITORY_ROOT))

import joblib, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from data_loader import MillingRun, load_milling_runs
from feature_extraction import WEAR_CLASS_ORDER, extract_dataset_features, select_feature_columns, wear_class_from_vb
from physics_features import spindle_frequency_hz, tooth_passing_frequency_hz
from train_baseline import build_model, make_group_split

N_CV_FOLDS = 5

def _build_gb_model(random_state=42):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingClassifier(n_estimators=200, max_depth=4,
                  learning_rate=0.08, subsample=0.8, random_state=random_state)),
    ])

def _cv_evaluate(features, *, include_physics, model_factory, random_state, n_folds=N_CV_FOLDS):
    data = features.reset_index(drop=True)
    feature_columns = select_feature_columns(data, include_physics=include_physics)
    X = data[feature_columns].to_numpy()
    y = data["wear_class"].to_numpy()
    groups = (data["case"].astype(str) + "_" + data["run"].astype(str)).to_numpy()
    splitter = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
    fold_acc, fold_f1 = [], []
    for tr, te in splitter.split(X, y, groups):
        m = model_factory(random_state=random_state)
        m.fit(X[tr], y[tr]); preds = m.predict(X[te])
        fold_acc.append(float(accuracy_score(y[te], preds)))
        fold_f1.append(float(f1_score(y[te], preds, average="macro", zero_division=0)))
    final = model_factory(random_state=random_state)
    tr_i, te_i = make_group_split(data, random_state=random_state)
    final.fit(X[tr_i.to_numpy()], y[tr_i.to_numpy()])
    fp = final.predict(X[te_i.to_numpy()])
    return {
        "model": final, "feature_columns": feature_columns,
        "y_test": pd.Series(y[te_i.to_numpy()]).reset_index(drop=True),
        "predictions": pd.Series(fp).reset_index(drop=True),
        "accuracy": float(accuracy_score(y[te_i.to_numpy()], fp)),
        "macro_f1": float(f1_score(y[te_i.to_numpy()], fp, average="macro", zero_division=0)),
        "classification_report": classification_report(y[te_i.to_numpy()], fp, output_dict=True, zero_division=0),
        "cv_accuracy_mean": float(np.mean(fold_acc)), "cv_accuracy_std": float(np.std(fold_acc)),
        "cv_f1_mean": float(np.mean(fold_f1)), "cv_f1_std": float(np.std(fold_f1)),
        "cv_folds": len(fold_acc), "n_features": len(feature_columns),
    }

def _compact_metrics(r):
    return {
        "accuracy": r["accuracy"], "macro_f1": r["macro_f1"],
        "cv_f1_mean": r.get("cv_f1_mean",0.), "cv_f1_std": r.get("cv_f1_std",0.),
        "cv_accuracy_mean": r.get("cv_accuracy_mean",0.), "cv_accuracy_std": r.get("cv_accuracy_std",0.),
        "cv_folds": r.get("cv_folds",0), "n_features": r.get("n_features",0),
        "classification_report": r["classification_report"],
    }

def _ordered_labels(*series):
    present = set()
    for s in series:
        for v in s.dropna().unique(): present.add(v)
    ordered = [l for l in WEAR_CLASS_ORDER if l in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered

def _save_models(out_dir, baseline, physics):
    joblib.dump({"model": baseline["model"], "feature_columns": baseline["feature_columns"], "include_physics": False}, out_dir / "baseline_model.joblib")
    joblib.dump({"model": physics["model"],  "feature_columns": physics["feature_columns"],  "include_physics": True},  out_dir / "physics_guided_model.joblib")

def _plot_confusion_matrices(path, baseline, physics):
    labels = _ordered_labels(baseline["y_test"], baseline["predictions"], physics["predictions"])
    fig, axes = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    for ax, title, r in [(axes[0],"Pure ML",baseline),(axes[1],"Physics-guided",physics)]:
        ConfusionMatrixDisplay.from_predictions(r["y_test"], r["predictions"], labels=labels,
            display_labels=labels, cmap="Blues", colorbar=False, ax=ax)
        ax.set_title(title)
    fig.savefig(path, dpi=180); plt.close(fig)

def _plot_feature_importance(path, result, top_n=15):
    rf = result["model"].named_steps["model"]
    imp = pd.Series(rf.feature_importances_, index=result["feature_columns"]).sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
    imp.sort_values().plot(kind="barh", ax=ax, color="#2e7d6f")
    ax.set_title("Physics-guided model — top feature importances")
    ax.set_xlabel("Random Forest importance")
    fig.savefig(path, dpi=180); plt.close(fig)

def _choose_example_runs(runs):
    examples = []
    for label in WEAR_CLASS_ORDER:
        m = [r for r in runs if wear_class_from_vb(r.vb) == label]
        if m: examples.append(m[len(m)//2])
    return examples or runs[:1]

def _plot_fft_examples(path, runs):
    examples = _choose_example_runs(runs)
    fig, axes = plt.subplots(len(examples), 1, figsize=(9, 2.8*len(examples)), constrained_layout=True)
    if len(examples) == 1: axes = [axes]
    for ax, run in zip(axes, examples):
        sig_name = "vib_spindle" if "vib_spindle" in run.signals else run.signals.columns[0]
        sig = run.signals[sig_name].to_numpy(dtype=float)
        freqs = np.fft.rfftfreq(sig.size, d=1.0/run.sample_rate_hz)
        amp = np.abs(np.fft.rfft(sig - np.mean(sig)))
        mask = freqs <= min(500., run.sample_rate_hz/2.)
        ax.plot(freqs[mask], amp[mask], color="#263238", linewidth=1.0)
        spf = spindle_frequency_hz(run.speed_rpm)
        tpf = tooth_passing_frequency_hz(run.speed_rpm, run.tooth_count)
        if spf: ax.axvline(spf, color="#d14f3f", linestyle="--", linewidth=1., label="spindle")
        if tpf: ax.axvline(tpf, color="#3266a8", linestyle="--", linewidth=1., label="tooth-pass")
        ax.set_title(f"case {run.case}, run {run.run}, VB={run.vb:.3f}, class={wear_class_from_vb(run.vb)}")
        ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Amplitude"); ax.legend(loc="upper right")
    fig.savefig(path, dpi=180); plt.close(fig)

def _plot_prediction_vs_true(path, test_rows, baseline, physics):
    labels = _ordered_labels(baseline["y_test"], baseline["predictions"], physics["predictions"])
    lc = {l: i for i, l in enumerate(labels)}
    order = (test_rows.reset_index(drop=True).assign(_r=np.arange(len(test_rows)))
             .sort_values(["case","run","window_id"])["_r"].to_numpy())
    x = np.arange(len(order))
    tc = [lc[l] for l in baseline["y_test"].iloc[order]]
    bc = [lc[l] for l in baseline["predictions"].iloc[order]]
    pc = [lc[l] for l in physics["predictions"].iloc[order]]
    fig, ax = plt.subplots(figsize=(10,4), constrained_layout=True)
    ax.plot(x, tc, "o-", label="true",        linewidth=1.2, markersize=3)
    ax.plot(x, bc, "s--",label="RF baseline", linewidth=1.0, markersize=3)
    ax.plot(x, pc, "^--",label="RF physics",  linewidth=1.0, markersize=3)
    ax.set_yticks(list(lc.values()), labels=list(lc.keys()))
    ax.set_xlabel("Test windows sorted by case/run"); ax.set_ylabel("Wear class")
    ax.set_title("Prediction vs true class"); ax.legend()
    fig.savefig(path, dpi=180); plt.close(fig)

def _plot_cv_comparison(path, rf_b, rf_p, gb_b, gb_p):
    names  = ["RF Baseline","RF Physics","GB Baseline","GB Physics"]
    means  = [r["cv_f1_mean"] for r in [rf_b, rf_p, gb_b, gb_p]]
    stds   = [r["cv_f1_std"]  for r in [rf_b, rf_p, gb_b, gb_p]]
    colors = ["#607D8B","#2e7d6f","#7B68EE","#1a5f7a"]
    fig, ax = plt.subplots(figsize=(7,4), constrained_layout=True)
    bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors, alpha=0.88, error_kw={"linewidth":1.5})
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x()+bar.get_width()/2, mean+std+0.012,
                f"{mean:.3f}+/-{std:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.10); ax.set_ylabel(f"{N_CV_FOLDS}-Fold CV Macro-F1")
    ax.set_title(f"{N_CV_FOLDS}-Fold Group-Aware CV: Macro-F1 by Model Variant")
    ax.axhline(0.9, color="gray", linestyle=":", linewidth=0.8, label="0.90 target")
    ax.legend(fontsize=8); fig.savefig(path, dpi=180); plt.close(fig)

def run_evaluation(*, data_path, demo, output_dir="results", window_seconds=1.0, overlap=0.5, random_state=42):
    runs = load_milling_runs(data_path, demo=demo or data_path is None)
    features = extract_dataset_features(runs, include_physics=True,
                window_seconds=window_seconds, overlap=overlap).dropna(subset=["wear_class"]).reset_index(drop=True)
    rf_b = _cv_evaluate(features, include_physics=False, model_factory=build_model,     random_state=random_state)
    rf_p = _cv_evaluate(features, include_physics=True,  model_factory=build_model,     random_state=random_state)
    gb_b = _cv_evaluate(features, include_physics=False, model_factory=_build_gb_model, random_state=random_state)
    gb_p = _cv_evaluate(features, include_physics=True,  model_factory=_build_gb_model, random_state=random_state)
    baseline, physics = rf_b, rf_p
    tr, te = make_group_split(features, random_state=random_state)
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    _save_models(out_dir, baseline, physics)
    _plot_confusion_matrices(out_dir/"confusion_matrix.png", baseline, physics)
    _plot_feature_importance(out_dir/"feature_importance.png", physics)
    _plot_fft_examples(out_dir/"fft_examples.png", runs)
    _plot_prediction_vs_true(out_dir/"prediction_vs_true.png", features.loc[te], baseline, physics)
    _plot_cv_comparison(out_dir/"cv_comparison.png", rf_b, rf_p, gb_b, gb_p)
    pd.DataFrame({"case": features.loc[te,"case"].to_numpy(), "run": features.loc[te,"run"].to_numpy(),
        "window_id": features.loc[te,"window_id"].to_numpy(), "true": baseline["y_test"].to_numpy(),
        "baseline_pred": baseline["predictions"].to_numpy(), "physics_guided_pred": physics["predictions"].to_numpy(),
    }).to_csv(out_dir/"predictions.csv", index=False)
    metrics = {"rf_baseline": _compact_metrics(rf_b), "rf_physics": _compact_metrics(rf_p),
               "gb_baseline": _compact_metrics(gb_b), "gb_physics": _compact_metrics(gb_p),
               "baseline": _compact_metrics(rf_b), "physics_guided": _compact_metrics(rf_p),
               "n_windows": int(len(features))}
    (out_dir/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-path", default=None); p.add_argument("--demo", action="store_true")
    p.add_argument("--output-dir", default="results"); p.add_argument("--window-seconds", type=float, default=1.0)
    p.add_argument("--overlap", type=float, default=0.5)
    return p.parse_args()

def main():
    args = parse_args()
    m = run_evaluation(data_path=args.data_path, demo=args.demo, output_dir=args.output_dir,
                       window_seconds=args.window_seconds, overlap=args.overlap)
    summary = {k: {x:v for x,v in vals.items() if x != "classification_report"}
               for k, vals in m.items() if k in {"rf_baseline","rf_physics","gb_baseline","gb_physics"}}
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
