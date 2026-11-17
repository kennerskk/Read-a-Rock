# -*- coding: utf-8 -*-
# =====================================================================
#  generate_figures.py — สร้างรูปสำหรับสไลด์นำเสนอ
#  ผลิต 4 รูป + สรุปตัวเลขสำหรับสไลด์:
#    1) confusion_matrix_best.png   — confusion matrix ของโมเดลที่ดีที่สุด
#    2) roc_comparison.png          — ROC curve เทียบทุกโมเดล (macro-average OvR)
#    3) roc_per_class_best.png      — ROC แยกตามคลาสของโมเดลที่ดีที่สุด
#    4) model_comparison_bar.png    — Accuracy + 95% CI ของทุกโมเดล
#
#  หมายเหตุ: ใช้ผลทำนาย out-of-sample จาก walk-forward validation เท่านั้น
#            (ไม่มี data leakage) ป้ายคลาสในรูปใช้ภาษาอังกฤษเพื่อเลี่ยงปัญหา
#            ฟอนต์ไทยใน matplotlib ที่จะแสดงเป็นสี่เหลี่ยมว่าง
#
#  วิธีรัน:  python generate_figures.py
#            python generate_figures.py --folds 10 --outdir reports/figures
# =====================================================================
from __future__ import annotations

import os
import sys
import math
import argparse
import datetime
from typing import Dict, List

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from telemetry import read_telemetry
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, log_loss, confusion_matrix,
                             roc_curve, auc, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize

from model_pipeline import FeatureStore

# ป้ายคลาส: ค้อน=0, กรรไกร=1, กระดาษ=2 (ตาม FeatureStore.MOVE_TO_INT)
CLASS_LABELS_EN = ["Rock", "Scissors", "Paper"]
CLASS_LABELS_TH = ["ค้อน", "กรรไกร", "กระดาษ"]
N_CLASSES = 3

# จานสีที่อ่านออกทั้งจอโปรเจกเตอร์และขาวดำ
COLORS = ["#2563eb", "#dc2626", "#059669"]


def make_models(seed: int = 42) -> Dict[str, object]:
    return {
        "LogisticRegression": LogisticRegression(max_iter=500, C=1.0),
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed),
        "GradientBoosting": HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=seed),
    }


def collect_oos_predictions(csv_path: str, n_folds: int = 10, seed: int = 42):
    """
    เก็บผลทำนาย out-of-sample ด้วย walk-forward (expanding window)
    คืน dict: ชื่อโมเดล -> {y_true, y_pred, y_proba, fold_acc[], fold_ll[]}
    """
    df = read_telemetry(csv_path)
    X, y, feat_names = FeatureStore.build_features(df)
    X = np.asarray(X)
    y = np.asarray(y)
    total = len(X)
    min_train = int(total * 0.40)
    val_size = max(25, int((total - min_train) / n_folds))

    names = list(make_models(seed).keys())
    store = {n: {"y_true": [], "y_pred": [], "y_proba": [], "fold_acc": [], "fold_ll": []}
             for n in names}

    models = make_models(seed)
    used_folds = 0
    for fold in range(n_folds):
        tr_end = min_train + fold * val_size
        va_end = min(total, tr_end + val_size)
        if va_end - tr_end < 10:
            break
        X_tr, y_tr = X[:tr_end], y[:tr_end]
        X_va, y_va = X[tr_end:va_end], y[tr_end:va_end]
        if len(np.unique(y_tr)) < N_CLASSES:
            continue
        for name, model in models.items():
            model.fit(X_tr, y_tr)
            pred = model.predict(X_va)
            proba = model.predict_proba(X_va)
            store[name]["y_true"].extend(y_va.tolist())
            store[name]["y_pred"].extend(pred.tolist())
            store[name]["y_proba"].extend(proba.tolist())
            store[name]["fold_acc"].append(float(accuracy_score(y_va, pred)))
            store[name]["fold_ll"].append(float(log_loss(y_va, proba, labels=[0, 1, 2])))
        used_folds += 1

    for n in names:
        store[n]["y_true"] = np.asarray(store[n]["y_true"])
        store[n]["y_pred"] = np.asarray(store[n]["y_pred"])
        store[n]["y_proba"] = np.asarray(store[n]["y_proba"])

    meta = {"n_samples": total, "n_features": len(feat_names), "feature_names": feat_names,
            "min_train": min_train, "val_size": val_size, "used_folds": used_folds,
            "class_counts": {CLASS_LABELS_EN[c]: int(np.sum(y == c)) for c in range(N_CLASSES)}}
    return store, meta


def macro_roc(y_true: np.ndarray, y_proba: np.ndarray):
    """ROC แบบ One-vs-Rest แล้วเฉลี่ยแบบ macro (ถ่วงทุกคลาสเท่ากัน)"""
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    per_class = {}
    for c in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
        per_class[c] = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
    grid = np.linspace(0.0, 1.0, 200)
    mean_tpr = np.zeros_like(grid)
    for c in range(N_CLASSES):
        mean_tpr += np.interp(grid, per_class[c]["fpr"], per_class[c]["tpr"])
    mean_tpr /= N_CLASSES
    return grid, mean_tpr, float(auc(grid, mean_tpr)), per_class


def mean_ci(x) -> tuple:
    a = np.asarray(x, dtype=float)
    n = a.size
    m = float(a.mean())
    if n < 2:
        return m, 0.0
    sd = float(a.std(ddof=1))
    if sd == 0:
        return m, 0.0
    return m, float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n))


# =====================================================================
#  รูปที่ 1: Confusion Matrix
# =====================================================================
def plot_confusion(store, best, outdir):
    y_true = store[best]["y_true"]
    y_pred = store[best]["y_pred"]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    cm_pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

    ax.set_xticks(range(N_CLASSES), CLASS_LABELS_EN)
    ax.set_yticks(range(N_CLASSES), CLASS_LABELS_EN)
    ax.set_xlabel("Predicted player move", fontsize=11, labelpad=8)
    ax.set_ylabel("Actual player move", fontsize=11, labelpad=8)
    ax.set_title(f"Confusion Matrix — {best}\n(out-of-sample, walk-forward validation, n={len(y_true)})",
                 fontsize=12, pad=14)

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            txt = f"{cm[i, j]}\n{cm_pct[i, j]:.1f}%"
            ax.text(j, i, txt, ha="center", va="center", fontsize=11,
                    color="white" if cm_pct[i, j] > 55 else "#1f2937",
                    fontweight="bold" if i == j else "normal")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("% of actual class (row-normalized)", fontsize=9)
    ax.set_xticks(np.arange(-0.5, N_CLASSES, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_CLASSES, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    path = os.path.join(outdir, "confusion_matrix_best.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(N_CLASSES)), zero_division=0)
    return path, cm, cm_pct, {"precision": prec, "recall": rec, "f1": f1, "support": sup}


# =====================================================================
#  รูปที่ 2: ROC เทียบทุกโมเดล
# =====================================================================
def plot_roc_comparison(store, outdir):
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    aucs = {}
    for idx, (name, d) in enumerate(store.items()):
        grid, mean_tpr, macro_auc, _ = macro_roc(d["y_true"], d["y_proba"])
        aucs[name] = macro_auc
        ax.plot(grid, mean_tpr, color=COLORS[idx % len(COLORS)], linewidth=2.2,
                label=f"{name}  (AUC = {macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=1.4,
            label="Random classifier (AUC = 0.500)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve Comparison of All Models\n(macro-average One-vs-Rest, 3 classes, out-of-sample)",
                 fontsize=12, pad=14)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_aspect("equal")
    fig.tight_layout()
    path = os.path.join(outdir, "roc_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path, aucs


# =====================================================================
#  รูปที่ 3: ROC แยกคลาสของโมเดลที่ดีที่สุด
# =====================================================================
def plot_roc_per_class(store, best, outdir):
    _, _, macro_auc, per_class = macro_roc(store[best]["y_true"], store[best]["y_proba"])
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    for c in range(N_CLASSES):
        ax.plot(per_class[c]["fpr"], per_class[c]["tpr"], color=COLORS[c], linewidth=2.2,
                label=f"{CLASS_LABELS_EN[c]}  (AUC = {per_class[c]['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", linewidth=1.4, label="Random (AUC = 0.500)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"Per-Class ROC — {best}\n(One-vs-Rest, macro AUC = {macro_auc:.3f})",
                 fontsize=12, pad=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_aspect("equal")
    fig.tight_layout()
    path = os.path.join(outdir, "roc_per_class_best.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path, per_class, macro_auc


# =====================================================================
#  รูปที่ 4: แท่งเทียบ Accuracy + 95% CI
# =====================================================================
def plot_model_bars(store, outdir):
    names = list(store.keys())
    means, errs = [], []
    for n in names:
        m, h = mean_ci(store[n]["fold_acc"])
        means.append(m * 100)
        errs.append(h * 100)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xs = np.arange(len(names))
    bars = ax.bar(xs, means, yerr=errs, capsize=7, width=0.55,
                  color=[COLORS[i % len(COLORS)] for i in range(len(names))],
                  edgecolor="#1f2937", linewidth=0.8, alpha=0.9)
    ax.axhline(100.0 / 3.0, linestyle="--", color="#6b7280", linewidth=1.5)
    ax.text(len(names) - 0.42, 100.0 / 3.0 + 1.2, "Random baseline (33.3%)",
            fontsize=9, color="#4b5563", ha="right")

    for b, m, e in zip(bars, means, errs):
        ax.text(b.get_x() + b.get_width() / 2, m + e + 1.4, f"{m:.2f}%",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(xs, names, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, max(means) + max(errs) + 12)
    ax.set_title("Model Accuracy Comparison with 95% Confidence Intervals\n"
                 "(walk-forward validation, error bars = 95% CI)", fontsize=12, pad=14)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(outdir, "model_comparison_bar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path, dict(zip(names, zip(means, errs)))


# =====================================================================
#  สรุปตัวเลขสำหรับสไลด์
# =====================================================================
def write_summary(outdir, meta, store, best, aucs, cm, cm_pct, prf, bars, per_class, macro_auc, paths):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    L.append("# ตัวเลขและรูปสำหรับสไลด์นำเสนอ")
    L.append("")
    L.append(f"**สร้างเมื่อ:** {now}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## สไลด์ข้อ 3 — Data Sets & Class Labels")
    L.append("")
    L.append("### Class labels (3 คลาส)")
    L.append("")
    L.append("| index | ภาษาไทย | English | จำนวนตัวอย่าง |")
    L.append("|---:|:---|:---|---:|")
    for c in range(N_CLASSES):
        L.append(f"| {c} | {CLASS_LABELS_TH[c]} | {CLASS_LABELS_EN[c]} | "
                 f"{meta['class_counts'][CLASS_LABELS_EN[c]]} |")
    L.append("")
    L.append("### การแบ่งข้อมูล (Walk-Forward Time-Series Split)")
    L.append("")
    L.append("| รายการ | ค่า |")
    L.append("|:---|---:|")
    L.append(f"| ข้อมูลทั้งหมด (ตา) | {meta['n_samples']} |")
    L.append(f"| จำนวนฟีเจอร์ | {meta['n_features']} |")
    L.append(f"| Train เริ่มต้น | {meta['min_train']} ตา (40%) |")
    L.append(f"| Validation ต่อ fold | {meta['val_size']} ตา |")
    L.append(f"| จำนวน fold | {meta['used_folds']} |")
    L.append(f"| ตัวอย่าง out-of-sample รวม | {len(store[best]['y_true'])} |")
    L.append("")
    L.append("> ใช้ walk-forward (expanding window) แทนการ random split "
             "เพราะข้อมูลเป็นอนุกรมเวลา การสุ่มสลับจะทำให้โมเดลเห็นอนาคต (data leakage)")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## สไลด์ข้อ 4 — Confusion Matrix & ROC")
    L.append("")
    L.append(f"**โมเดลที่ดีที่สุด (เลือกจาก macro AUC): `{best}` — AUC = {macro_auc:.3f}**")
    L.append("")
    L.append("### ตารางเทียบทุกโมเดล")
    L.append("")
    L.append("| โมเดล | Accuracy (95% CI) | Macro AUC | สรุป |")
    L.append("|:---|---:|---:|:---|")
    for n in store:
        m, e = bars[n]
        tag = "**ดีที่สุด**" if n == best else ""
        L.append(f"| {n} | {m:.2f}% ± {e:.2f}% | {aucs[n]:.3f} | {tag} |")
    L.append("")
    L.append("### Confusion Matrix (ค่าดิบ)")
    L.append("")
    L.append("| actual \\ predicted | " + " | ".join(CLASS_LABELS_EN) + " | total |")
    L.append("|:---|" + "---:|" * (N_CLASSES + 1))
    for i in range(N_CLASSES):
        row = " | ".join(str(int(cm[i, j])) for j in range(N_CLASSES))
        L.append(f"| **{CLASS_LABELS_EN[i]}** | {row} | {int(cm[i].sum())} |")
    L.append("")
    L.append("### Precision / Recall / F1 ต่อคลาส")
    L.append("")
    L.append("| คลาส | Precision | Recall | F1 | Support |")
    L.append("|:---|---:|---:|---:|---:|")
    for c in range(N_CLASSES):
        L.append(f"| {CLASS_LABELS_EN[c]} ({CLASS_LABELS_TH[c]}) | {prf['precision'][c]:.3f} | "
                 f"{prf['recall'][c]:.3f} | {prf['f1'][c]:.3f} | {int(prf['support'][c])} |")
    L.append("")
    L.append("### AUC แยกตามคลาสของโมเดลที่ดีที่สุด")
    L.append("")
    L.append("| คลาส | AUC |")
    L.append("|:---|---:|")
    for c in range(N_CLASSES):
        L.append(f"| {CLASS_LABELS_EN[c]} | {per_class[c]['auc']:.3f} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## ไฟล์รูปที่สร้าง")
    L.append("")
    for k, v in paths.items():
        L.append(f"- `{v}` — {k}")
    L.append("")
    L.append("> ป้ายคลาสในรูปใช้ภาษาอังกฤษ (Rock / Scissors / Paper) "
             "เพราะฟอนต์เริ่มต้นของ matplotlib ไม่รองรับภาษาไทย จะแสดงเป็นสี่เหลี่ยมว่าง")
    L.append("")
    path = os.path.join(outdir, "figures_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return path


def main():
    ap = argparse.ArgumentParser(description="สร้างรูป confusion matrix และ ROC สำหรับสไลด์")
    ap.add_argument("--csv", default="data/players")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="reports/figures")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("=" * 80)
    print("  สร้างรูปสำหรับสไลด์นำเสนอ")
    print("=" * 80)

    print("  [1/5] เก็บผลทำนาย out-of-sample ด้วย walk-forward validation...")
    store, meta = collect_oos_predictions(args.csv, args.folds, args.seed)
    print(f"        ข้อมูล {meta['n_samples']} ตา | {meta['n_features']} ฟีเจอร์ | "
          f"{meta['used_folds']} folds | out-of-sample {len(store[list(store)[0]]['y_true'])} ตัวอย่าง")

    print("  [2/5] วาด ROC เทียบทุกโมเดล...")
    roc_path, aucs = plot_roc_comparison(store, args.outdir)
    best = max(aucs, key=aucs.get)
    print(f"        โมเดลที่ดีที่สุดตาม macro AUC: {best} (AUC = {aucs[best]:.3f})")

    print("  [3/5] วาด confusion matrix ของโมเดลที่ดีที่สุด...")
    cm_path, cm, cm_pct, prf = plot_confusion(store, best, args.outdir)

    print("  [4/5] วาด ROC แยกคลาส และกราฟแท่งเทียบ accuracy...")
    rocc_path, per_class, macro_auc = plot_roc_per_class(store, best, args.outdir)
    bar_path, bars = plot_model_bars(store, args.outdir)

    print("  [5/5] เขียนสรุปตัวเลขสำหรับสไลด์...")
    paths = {
        "Confusion matrix ของโมเดลที่ดีที่สุด (สไลด์ข้อ 4)": cm_path,
        "ROC เทียบทุกโมเดล (สไลด์ข้อ 4)": roc_path,
        "ROC แยกคลาส (สไลด์เสริม)": rocc_path,
        "Accuracy + 95% CI (สไลด์ข้อ 4)": bar_path,
    }
    sm_path = write_summary(args.outdir, meta, store, best, aucs, cm, cm_pct,
                            prf, bars, per_class, macro_auc, paths)

    print("")
    print("-" * 80)
    print(f"{'โมเดล':<24} | {'Accuracy (95% CI)':<22} | Macro AUC")
    print("-" * 80)
    for n in store:
        m, e = bars[n]
        star = "  <== ดีที่สุด" if n == best else ""
        print(f"{n:<24} | {m:6.2f}% +/- {e:4.2f}%        | {aucs[n]:.3f}{star}")
    print("-" * 80)
    print("")
    for k, v in paths.items():
        print(f"  {v}")
    print(f"  {sm_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
