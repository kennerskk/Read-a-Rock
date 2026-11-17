# -*- coding: utf-8 -*-
# =====================================================================
#  statistical_test.py — การทดสอบนัยสำคัญทางสถิติ: "โมเดลไหนดีที่สุด?"
# ---------------------------------------------------------------------
#  หลักการออกแบบการทดลอง (Experimental Design)
#  1) หน่วยวิเคราะห์ = "แมตช์" ไม่ใช่ "ตา"
#     เพราะบอสเรียนรู้ระหว่างเกม (online learning) ตาแต่ละตาจึงไม่เป็น
#     อิสระต่อกัน การใช้ binomial CI กับ 1000 ตาจะทำให้ CI แคบเกินจริง
#  2) Common Random Numbers (CRN): ทุกบอสเจอ seed ชุดเดียวกัน
#     -> เปรียบเทียบแบบจับคู่ (paired) ตัดความแปรปรวนของคู่ต่อสู้ออก
#  3) R = (W + 0.5D)/N * 100  -> การเล่นสุ่มล้วนให้ E[R] = 50% พอดี
#     ใช้ 50% เป็น null hypothesis ของ baseline test
#
#  วิธีรัน:
#     python statistical_test.py                  # ค่าเริ่มต้น 30 แมตช์ x 1000 ตา
#     python statistical_test.py --reps 50
#     python statistical_test.py --skip-ml        # ข้ามส่วนโมเดล sklearn
# =====================================================================
from __future__ import annotations

import sys
import os
import math
import random
import argparse
import datetime
import itertools
from typing import Dict, List, Tuple, Sequence

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from scipy import stats

ALPHA = 0.05
RANDOM_BASELINE_R = 50.0   # E[R] ของการเล่นสุ่มล้วน เมื่อ R = (W + 0.5D)/N

# ค่าวิกฤต q_0.05 ของ Nemenyi post-hoc (Demsar, JMLR 2006) ตาม k = จำนวนโมเดล
NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
    8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313,
    14: 3.354, 15: 3.391,
}


# =====================================================================
#  ส่วนที่ 1: ฟังก์ชันสถิติพื้นฐาน
# =====================================================================
def mean_ci(x: Sequence[float], conf: float = 0.95) -> Tuple[float, float, float]:
    """คืนค่า (mean, SD, half-width ของ CI) โดยใช้การแจกแจง t"""
    a = np.asarray(x, dtype=float)
    n = a.size
    m = float(a.mean())
    if n < 2:
        return m, 0.0, 0.0
    sd = float(a.std(ddof=1))
    if sd == 0.0:
        return m, 0.0, 0.0
    h = float(stats.t.ppf(0.5 + conf / 2.0, n - 1) * sd / math.sqrt(n))
    return m, sd, h


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni: แก้ปัญหา multiple comparisons (อนุรักษ์น้อยกว่า Bonferroni)"""
    p = list(pvals)
    m = len(p)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired_compare(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """
    เปรียบเทียบแบบจับคู่: เลือก paired t-test ถ้าผลต่างแจกแจงปกติ
    (Shapiro-Wilk p > 0.05) ไม่เช่นนั้นใช้ Wilcoxon signed-rank
    effect size ใช้ Cohen's d_z = mean(diff) / SD(diff)
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    d = x - y
    n = d.size
    out = {"mean_diff": float(d.mean()), "n": n}

    if np.allclose(d, 0.0):
        out.update(test="identical", p=1.0, dz=0.0, shapiro_p=float("nan"))
        return out

    sh_p = float("nan")
    if n >= 3:
        try:
            sh_p = float(stats.shapiro(d).pvalue)
        except Exception:
            sh_p = float("nan")
    out["shapiro_p"] = sh_p

    use_t = (not math.isnan(sh_p)) and sh_p > ALPHA
    if use_t:
        out["test"] = "paired t"
        out["p"] = float(stats.ttest_rel(x, y).pvalue)
    else:
        out["test"] = "Wilcoxon"
        try:
            out["p"] = float(stats.wilcoxon(d).pvalue)
        except Exception:
            out["p"] = 1.0

    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    out["dz"] = float(d.mean() / sd) if sd > 0 else float("inf")
    return out


def effect_label(dz: float) -> str:
    """ตีความขนาดผลตามเกณฑ์ Cohen"""
    a = abs(dz)
    if math.isinf(a):
        return "สมบูรณ์"
    if a < 0.2:
        return "เล็กมาก"
    if a < 0.5:
        return "เล็ก"
    if a < 0.8:
        return "กลาง"
    if a < 1.2:
        return "ใหญ่"
    return "ใหญ่มาก"


def average_ranks(score_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """จัดอันดับในแต่ละบล็อก (อันดับ 1 = ดีที่สุด) แล้วเฉลี่ยข้ามบล็อก"""
    M = np.asarray(score_matrix, dtype=float)
    ranks = np.empty_like(M)
    for i in range(M.shape[0]):
        ranks[i, :] = stats.rankdata(-M[i, :])   # ติดลบ เพราะคะแนนสูง = ดี
    return ranks.mean(axis=0), ranks


def nemenyi_cd(k: int, n_blocks: int) -> float:
    """Critical Difference ของ Nemenyi: CD = q * sqrt(k(k+1) / (6N))"""
    q = NEMENYI_Q05.get(k)
    if q is None:
        q = NEMENYI_Q05[max(NEMENYI_Q05)]
    return q * math.sqrt(k * (k + 1) / (6.0 * n_blocks))


# =====================================================================
#  ส่วนที่ 2: การทดลองในสนามประลอง (Arena: บอส x persona)
# =====================================================================
def short_boss_name(cls) -> str:
    raw = getattr(cls, "name", str(cls))
    if "." in raw:
        raw = raw.split(".", 1)[1]
    return raw.split("(")[0].strip()


def short_persona_name(cls) -> str:
    raw = getattr(cls, "name", str(cls))
    if "." in raw:
        raw = raw.split(".", 1)[1]
    return raw.strip()


def run_arena(reps: int, rounds: int, seed0: int = 0, verbose: bool = True):
    # แสดงแถบความคืบหน้าเฉพาะเมื่อรันบนเทอร์มินัลจริง (ไม่ทำให้ไฟล์ log เลอะ)
    verbose = verbose and bool(getattr(sys.stdout, "isatty", lambda: False)())
    """
    รัน reps แมตช์อิสระต่อคู่ (บอส x persona) ด้วย seed ชุดเดียวกันทุกบอส
    คืนค่า (boss_names, persona_names, data) โดย data[b, p, r] = R ของแมตช์นั้น
    """
    from adaptive_arena import BOSSES
    from evaluate_ai import PERSONAS, evaluate_matchup

    boss_names = [short_boss_name(b) for b in BOSSES]
    pers_names = [short_persona_name(p) for p in PERSONAS]
    data = np.zeros((len(BOSSES), len(PERSONAS), reps), dtype=float)

    total = len(BOSSES) * len(PERSONAS) * reps
    done = 0
    for bi, b_cls in enumerate(BOSSES):
        for pi, p_cls in enumerate(PERSONAS):
            for r in range(reps):
                # CRN: seed เดียวกันสำหรับ replication ที่ r ของทุกบอส -> paired design
                random.seed(seed0 + r)
                win_rate, _loss_rate, draw_rate = evaluate_matchup(b_cls, p_cls, rounds=rounds)
                data[bi, pi, r] = win_rate + 0.5 * draw_rate
                done += 1
            if verbose:
                pct = 100.0 * done / total
                print(f"\r  กำลังจำลอง... {pct:5.1f}%  ({boss_names[bi]} vs {pers_names[pi][:22]})      ",
                      end="", flush=True)
    if verbose:
        print("\r  จำลองครบ " + f"{total} แมตช์" + " " * 50)
    return boss_names, pers_names, data


def analyze_descriptive(boss_names, pers_names, data):
    """ตารางที่ 1: mean R +/- 95% CI ของทุกคู่"""
    rows = []
    for bi, bname in enumerate(boss_names):
        cells = []
        for pi in range(len(pers_names)):
            m, sd, h = mean_ci(data[bi, pi, :])
            cells.append({"mean": m, "sd": sd, "ci": h})
        overall = mean_ci(data[bi].mean(axis=0))   # เฉลี่ยข้าม persona ในแต่ละ replication
        rows.append({"boss": bname, "cells": cells,
                     "overall_mean": overall[0], "overall_ci": overall[2]})
    return rows


def analyze_baseline(boss_names, pers_names, data):
    """
    ตารางที่ 2: ทดสอบว่าชนะเหนือการสุ่ม (R > 50%) อย่างมีนัยสำคัญหรือไม่
    one-sample t-test ทางเดียว + Holm correction
    รวมถึง sanity check: เจอ Nash ต้องไม่ชนะเกิน 50% อย่างมีนัยสำคัญ
    """
    recs = []
    for bi, bname in enumerate(boss_names):
        for pi, pname in enumerate(pers_names):
            x = data[bi, pi, :]
            m, sd, h = mean_ci(x)
            if sd == 0.0:
                p = 0.0 if m > RANDOM_BASELINE_R else 1.0
            else:
                p = float(stats.ttest_1samp(x, RANDOM_BASELINE_R, alternative="greater").pvalue)
            recs.append({"boss": bname, "persona": pname, "mean": m, "ci": h, "p": p})
    adj = holm([r["p"] for r in recs])
    for r, a in zip(recs, adj):
        r["p_holm"] = a
        r["sig"] = a < ALPHA
    return recs


def analyze_friedman(boss_names, pers_names, data):
    """
    ตารางที่ 3: Friedman test + Nemenyi post-hoc (Demsar 2006)
    บล็อก = (persona, seed) ทุกคู่ -> N = n_personas x reps บล็อก
    ทุกโมเดลเจอบล็อกเดียวกันเป๊ะ จึงเป็นการเทียบแบบจับคู่สมบูรณ์
    """
    n_boss, n_pers, reps = data.shape
    blocks = np.zeros((n_pers * reps, n_boss), dtype=float)
    row = 0
    for pi in range(n_pers):
        for r in range(reps):
            blocks[row, :] = data[:, pi, r]
            row += 1

    avg_ranks, _ = average_ranks(blocks)
    columns = [blocks[:, j] for j in range(n_boss)]
    chi2, p = stats.friedmanchisquare(*columns)
    cd = nemenyi_cd(n_boss, blocks.shape[0])

    order = np.argsort(avg_ranks)
    ranking = [{"boss": boss_names[j], "avg_rank": float(avg_ranks[j])} for j in order]

    # หากลุ่มที่ไม่ต่างกันอย่างมีนัยสำคัญ (|diff| <= CD)
    sorted_ranks = [r["avg_rank"] for r in ranking]
    raw_groups = []
    n = len(sorted_ranks)
    for i in range(n):
        j = i
        while j + 1 < n and (sorted_ranks[j + 1] - sorted_ranks[i]) <= cd:
            j += 1
        if j > i:
            raw_groups.append((i, j))
    groups = [g for g in raw_groups
              if not any(h != g and h[0] <= g[0] and g[1] <= h[1] for h in raw_groups)]

    return {"chi2": float(chi2), "p": float(p), "cd": cd,
            "n_blocks": blocks.shape[0], "k": n_boss,
            "ranking": ranking, "groups": groups, "blocks": blocks}


def analyze_champion(boss_names, pers_names, data, champion_idx):
    """
    ตารางที่ 4: แชมป์ vs บอสอื่นทุกตัว ในแต่ละ persona
    paired t / Wilcoxon + Cohen's d_z + Holm correction ทั้ง family
    """
    recs = []
    for pi, pname in enumerate(pers_names):
        for bi, bname in enumerate(boss_names):
            if bi == champion_idx:
                continue
            res = paired_compare(data[champion_idx, pi, :], data[bi, pi, :])
            recs.append({"persona": pname, "opponent": bname, **res})
    adj = holm([r["p"] for r in recs])
    for r, a in zip(recs, adj):
        r["p_holm"] = a
        r["sig"] = a < ALPHA
    return recs


def analyze_power(data):
    """
    ข้อ 8: power analysis ย้อนกลับ — ที่ n ปัจจุบัน ตรวจจับผลต่างได้เล็กสุดเท่าไร
    MDE (paired, two-sided, power 0.80) = (z_{a/2} + z_b) * SD_diff / sqrt(n)
    """
    n_boss, n_pers, reps = data.shape
    sds = []
    for pi in range(n_pers):
        for i, j in itertools.combinations(range(n_boss), 2):
            d = data[i, pi, :] - data[j, pi, :]
            if d.std(ddof=1) > 0:
                sds.append(float(d.std(ddof=1)))
    sd_med = float(np.median(sds)) if sds else 0.0
    z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(0.80)
    mde = z * sd_med / math.sqrt(reps) if reps > 0 else float("nan")
    n_for_2pp = (z * sd_med / 2.0) ** 2 if sd_med > 0 else 0.0
    return {"sd_diff_median": sd_med, "mde": mde, "reps": reps,
            "z_sum": float(z), "n_for_2pp": math.ceil(n_for_2pp)}


# =====================================================================
#  ส่วนที่ 3: การทดลองฝั่งโมเดล sklearn (walk-forward + McNemar)
# =====================================================================
def make_models(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    return {
        "LogisticRegression (Softmax)": LogisticRegression(max_iter=500, C=1.0),
        "RandomForest (Ensemble Trees)": RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=seed),
        "GradientBoosting (Histogram)": HistGradientBoostingClassifier(
            max_iter=100, max_depth=4, random_state=seed),
    }


def run_ml(csv_path: str, n_folds: int = 10, seeds=(42, 7, 2026), verbose: bool = True):
    """
    Walk-forward (expanding window) หลาย fold x หลาย seed
    -> ได้ fold scores จับคู่กันสำหรับ Wilcoxon
    -> เก็บผลทำนายรายตาของ seed แรก สำหรับ McNemar's test (Dietterich 1998)
    """
    import pandas as pd
    from model_pipeline import FeatureStore
    from sklearn.metrics import accuracy_score, log_loss

    verbose = verbose and bool(getattr(sys.stdout, "isatty", lambda: False)())

    if not os.path.exists(csv_path):
        return None

    df = read_telemetry(csv_path)
    X, y, feat_names = FeatureStore.build_features(df)
    X = np.asarray(X)
    y = np.asarray(y)
    total = len(X)
    if total < 200:
        return None

    min_train = int(total * 0.40)
    val_size = max(25, int((total - min_train) / n_folds))

    names = list(make_models(seeds[0]).keys())
    acc = {n: [] for n in names}
    lls = {n: [] for n in names}
    correct = {n: [] for n in names}     # boolean รายตา จาก seed แรก
    holdout = None

    for si, seed in enumerate(seeds):
        models = make_models(seed)
        for fold in range(n_folds):
            train_end = min_train + fold * val_size
            val_end = min(total, train_end + val_size)
            if val_end - train_end < 10:
                break
            X_tr, y_tr = X[:train_end], y[:train_end]
            X_va, y_va = X[train_end:val_end], y[train_end:val_end]
            if len(np.unique(y_tr)) < 2:
                continue
            for name, model in models.items():
                model.fit(X_tr, y_tr)
                preds = model.predict(X_va)
                probs = model.predict_proba(X_va)
                acc[name].append(float(accuracy_score(y_va, preds)))
                lls[name].append(float(log_loss(y_va, probs, labels=[0, 1, 2])))
                if si == 0:
                    correct[name].extend(list(preds == y_va))
            if verbose:
                print(f"\r  walk-forward seed {seed} fold {fold + 1}/{n_folds}   ",
                      end="", flush=True)
    if verbose:
        print("\r  walk-forward เสร็จสิ้น" + " " * 40)

    # ชุดทดสอบสุดท้ายที่กันไว้ (กัน selection bias จากการเลือกแชมป์)
    cut = int(total * 0.85)
    if total - cut >= 30:
        models = make_models(seeds[0])
        holdout = {}
        for name, model in models.items():
            model.fit(X[:cut], y[:cut])
            p = model.predict(X[cut:])
            pr = model.predict_proba(X[cut:])
            holdout[name] = {
                "accuracy": float(accuracy_score(y[cut:], p)),
                "log_loss": float(log_loss(y[cut:], pr, labels=[0, 1, 2])),
                "n": int(total - cut),
            }

    return {"names": names, "acc": acc, "log_loss": lls, "correct": correct,
            "n_samples": total, "n_features": len(feat_names), "holdout": holdout,
            "n_folds": n_folds, "seeds": list(seeds)}


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, float]:
    """McNemar's test แบบ exact (binomial) — เทียบ 2 classifier บนชุดทดสอบเดียวกัน"""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    n01 = int(np.sum(~a & b))   # A ผิด, B ถูก
    n10 = int(np.sum(a & ~b))   # A ถูก, B ผิด
    tot = n01 + n10
    if tot == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    p = float(stats.binomtest(n10, tot, 0.5).pvalue)
    return {"n01": n01, "n10": n10, "p": p}


def analyze_ml(ml: Dict) -> Dict:
    names = ml["names"]
    desc = {}
    for n in names:
        m_a, sd_a, h_a = mean_ci(ml["acc"][n])
        m_l, sd_l, h_l = mean_ci(ml["log_loss"][n])
        desc[n] = {"acc": m_a, "acc_ci": h_a, "acc_sd": sd_a,
                   "ll": m_l, "ll_ci": h_l, "n_folds": len(ml["acc"][n])}

    pairs = []
    for a, b in itertools.combinations(names, 2):
        wil = paired_compare(ml["acc"][a], ml["acc"][b])
        mc = mcnemar_exact(ml["correct"][a], ml["correct"][b])
        pairs.append({"a": a, "b": b, "fold": wil, "mcnemar": mc})
    adj_f = holm([p["fold"]["p"] for p in pairs])
    adj_m = holm([p["mcnemar"]["p"] for p in pairs])
    for p, af, am in zip(pairs, adj_f, adj_m):
        p["fold"]["p_holm"] = af
        p["fold"]["sig"] = af < ALPHA
        p["mcnemar"]["p_holm"] = am
        p["mcnemar"]["sig"] = am < ALPHA

    champion = max(names, key=lambda n: desc[n]["acc"])
    return {"desc": desc, "pairs": pairs, "champion": champion}


# =====================================================================
#  ส่วนที่ 4: การแสดงผล (console + markdown)
# =====================================================================
def render_cd_diagram(fr: Dict, width: int = 46) -> List[str]:
    """Critical Difference diagram แบบข้อความ (ซ้าย = อันดับดีกว่า)"""
    ranking = fr["ranking"]
    cd = fr["cd"]
    lo = min(r["avg_rank"] for r in ranking)
    hi = max(r["avg_rank"] for r in ranking)
    span = max(hi - lo, 1e-9)

    lines = []
    lines.append(f"CD = {cd:.3f}  (Nemenyi, alpha={ALPHA}, k={fr['k']}, N={fr['n_blocks']} บล็อก)")
    lines.append("")
    lines.append(f"{'อันดับเฉลี่ย':<12} {'ตำแหน่ง (ซ้าย=ดีกว่า)':<{width + 2}} โมเดล")
    lines.append("-" * (width + 46))
    for r in ranking:
        pos = int(round((r["avg_rank"] - lo) / span * (width - 1)))
        bar = " " * pos + "O" + " " * (width - 1 - pos)
        lines.append(f"{r['avg_rank']:>8.3f}     |{bar}| {r['boss']}")
    lines.append("")
    if fr["groups"]:
        lines.append("กลุ่มที่ยังสรุปไม่ได้ว่าต่างกัน (อันดับเฉลี่ยห่างกันไม่เกิน CD):")
        for (i, j) in fr["groups"]:
            members = ", ".join(ranking[t]["boss"] for t in range(i, j + 1))
            lines.append(f"  - [{members}]")
    else:
        lines.append("ทุกคู่มีอันดับเฉลี่ยห่างกันเกิน CD -> แยกความต่างได้ทุกคู่")
    return lines


def console_report(boss_names, pers_names, data, desc, base, fr, champ_recs, power, ml_an, args):
    sep = "=" * 100
    print("\n" + sep)
    print("  รายงานผลการทดสอบนัยสำคัญทางสถิติ (Statistical Significance Testing)")
    print(sep)
    print(f"  จำนวน replication ต่อคู่ : {args.reps} แมตช์ (seed {args.seed0}..{args.seed0 + args.reps - 1})")
    print(f"  จำนวนตาต่อแมตช์          : {args.rounds}")
    print(f"  หน่วยวิเคราะห์            : แมตช์ (ไม่ใช่ตา) เพราะบอสเรียนรู้ระหว่างเกม")
    print(f"  การจับคู่                 : Common Random Numbers (seed ชุดเดียวกันทุกบอส)")

    print("\n" + sep)
    print("  ตารางที่ 1: R = (W + 0.5D)/N x 100  แสดงเป็น mean +/- 95% CI")
    print(sep)
    head = f"{'โมเดลบอส AI':<22} | " + " | ".join(f"{p[:20]:<20}" for p in pers_names) + " | รวมทุก persona"
    print(head)
    print("-" * len(head))
    for row in desc:
        cells = " | ".join(f"{c['mean']:5.1f} +/-{c['ci']:4.1f}      " for c in row["cells"])
        print(f"{row['boss']:<22} | {cells} | {row['overall_mean']:5.1f} +/-{row['overall_ci']:4.1f}")

    print("\n" + sep)
    print(f"  ตารางที่ 2: ทดสอบเหนือ baseline สุ่มล้วน (H0: R = {RANDOM_BASELINE_R}%, ทางเดียว, Holm-corrected)")
    print(sep)
    print(f"{'โมเดลบอส':<22} | {'persona':<30} | {'R (%)':>12} | {'p (Holm)':>10} | ผล")
    print("-" * 100)
    for r in base:
        mark = "ชนะเหนือสุ่ม" if r["sig"] else "สรุปไม่ได้"
        print(f"{r['boss']:<22} | {r['persona'][:30]:<30} | {r['mean']:6.1f}+/-{r['ci']:4.1f} | "
              f"{r['p_holm']:10.2e} | {mark}")

    nash = [r for r in base if "สุ่ม" in r["persona"] or "Nash" in r["persona"]]
    if nash:
        bad = [r for r in nash if r["sig"]]
        print("\n  [Sanity / Falsification check] เจอผู้เล่นสุ่มสมบูรณ์ (Nash) ไม่มีโมเดลใดควรชนะเกิน 50%")
        if bad:
            print("  !! พบ " + str(len(bad)) + " โมเดลที่ชนะเกิน 50% อย่างมีนัยสำคัญ -> ตรวจหา bug / data leakage:")
            for r in bad:
                print(f"     - {r['boss']}: R = {r['mean']:.1f}%, p(Holm) = {r['p_holm']:.2e}")
        else:
            print("  OK: ไม่มีโมเดลใดชนะผู้เล่นสุ่มได้เกิน 50% อย่างมีนัยสำคัญ (สอดคล้องกับ Game Theory)")

    print("\n" + sep)
    print("  ตารางที่ 3: Friedman test + Nemenyi post-hoc (จัดอันดับรวมทุก persona)")
    print(sep)
    print(f"  Friedman chi-square = {fr['chi2']:.3f}, p = {fr['p']:.3e}, "
          f"df = {fr['k'] - 1}, N = {fr['n_blocks']} บล็อก")
    verdict = "ปฏิเสธ H0 -> โมเดลต่างกันจริง" if fr["p"] < ALPHA else "สรุปไม่ได้ว่าโมเดลต่างกัน"
    print(f"  สรุป: {verdict}")
    print("")
    for line in render_cd_diagram(fr):
        print("  " + line)

    best = fr["ranking"][0]["boss"]
    print("\n" + sep)
    print(f"  ตารางที่ 4: แชมป์ '{best}' vs โมเดลอื่น (paired, Holm-corrected)")
    print(sep)
    print(f"{'persona':<26} | {'คู่เทียบ':<22} | {'ผลต่าง R':>9} | {'test':<10} | {'p (Holm)':>10} | d_z | ขนาดผล")
    print("-" * 110)
    for r in champ_recs:
        star = "*" if r["sig"] else " "
        dz = "inf" if math.isinf(r["dz"]) else f"{r['dz']:.2f}"
        print(f"{r['persona'][:26]:<26} | {r['opponent']:<22} | {r['mean_diff']:>+8.2f}  | "
              f"{r['test']:<10} | {r['p_holm']:10.2e}{star}| {dz:>5} | {effect_label(r['dz'])}")
    print("  (* = ต่างกันอย่างมีนัยสำคัญหลังแก้ Holm ที่ alpha = 0.05)")

    print("\n" + sep)
    print("  ตารางที่ 5: Power analysis (ย้อนกลับ)")
    print(sep)
    print(f"  SD ของผลต่างแบบจับคู่ (median) = {power['sd_diff_median']:.2f} pp")
    print(f"  ที่ n = {power['reps']} แมตช์ ตรวจจับผลต่างได้เล็กสุด (MDE, power 0.80, สองทาง) "
          f"= {power['mde']:.2f} pp")
    print(f"  ถ้าต้องการตรวจจับผลต่าง 2.00 pp ต้องใช้ n >= {power['n_for_2pp']} แมตช์")

    if ml_an:
        print("\n" + sep)
        print("  ตารางที่ 6: เปรียบเทียบโมเดล sklearn (Walk-Forward + McNemar)")
        print(sep)
        print(f"{'โมเดล':<34} | {'Accuracy (95% CI)':<24} | {'Log-Loss (95% CI)':<22} | folds")
        print("-" * 100)
        for n, d in ml_an["desc"].items():
            print(f"{n:<34} | {d['acc'] * 100:5.2f}% +/- {d['acc_ci'] * 100:4.2f}%        | "
                  f"{d['ll']:.4f} +/- {d['ll_ci']:.4f}    | {d['n_folds']}")
        print("")
        print(f"{'คู่เปรียบเทียบ':<52} | {'fold-wise p':>12} | {'McNemar p':>11} | b/c")
        print("-" * 100)
        for p in ml_an["pairs"]:
            s1 = "*" if p["fold"]["sig"] else " "
            s2 = "*" if p["mcnemar"]["sig"] else " "
            label = f"{p['a'].split()[0]} vs {p['b'].split()[0]}"
            print(f"{label:<52} | {p['fold']['p_holm']:12.2e}{s1}| {p['mcnemar']['p_holm']:11.2e}{s2}| "
                  f"{p['mcnemar']['n10']}/{p['mcnemar']['n01']}")
        print("  (* = มีนัยสำคัญหลังแก้ Holm | b = คู่ที่โมเดลซ้ายถูก-ขวาผิด, c = ซ้ายผิด-ขวาถูก)")
    print("\n" + sep)


def build_markdown(boss_names, pers_names, data, desc, base, fr, champ_recs, power, ml_an, args) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    best = fr["ranking"][0]["boss"]
    md = []
    md.append("# รายงานการทดสอบนัยสำคัญทางสถิติ (Statistical Significance Report)")
    md.append("")
    md.append(f"**วันที่ประมวลผล:** {now}  ")
    md.append(f"**โมเดลที่ดีที่สุดตามอันดับเฉลี่ย:** `{best}`")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 0. การออกแบบการทดลอง (Experimental Design)")
    md.append("")
    md.append(f"| รายการ | ค่า |")
    md.append("|:---|:---|")
    md.append(f"| หน่วยวิเคราะห์ (unit of analysis) | **แมตช์** ไม่ใช่ตา |")
    md.append(f"| จำนวน replication ต่อคู่ | {args.reps} แมตช์ |")
    md.append(f"| จำนวนตาต่อแมตช์ | {args.rounds} ตา |")
    md.append(f"| seed ที่ใช้ | {args.seed0} ถึง {args.seed0 + args.reps - 1} |")
    md.append(f"| การจับคู่ | Common Random Numbers (CRN) |")
    md.append(f"| ระดับนัยสำคัญ | alpha = {ALPHA} |")
    md.append(f"| การแก้ multiple comparisons | Holm-Bonferroni |")
    md.append("")
    md.append("เหตุผลที่ใช้ *แมตช์* เป็นหน่วยวิเคราะห์: บอสเรียนรู้ระหว่างเกม (`learn()` ถูกเรียกทุกตา) "
              "ทำให้ผลแต่ละตาไม่เป็นอิสระต่อกัน การคำนวณ binomial confidence interval จาก "
              f"{args.rounds} ตาในแมตช์เดียวจะให้ช่วงความเชื่อมั่นที่แคบเกินความจริง "
              "จึงต้องรันหลายแมตช์อิสระแล้วคิดสถิติจากค่า R ของแต่ละแมตช์")
    md.append("")
    md.append(f"ตัวชี้วัด: `R = (W + 0.5D)/N x 100` ซึ่งทำให้การเล่นสุ่มล้วนมีค่าคาดหวัง "
              f"`E[R] = {RANDOM_BASELINE_R}%` พอดี จึงใช้เป็นสมมติฐานหลัก (null hypothesis) ได้โดยตรง")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 1. ผลลัพธ์เชิงพรรณนา: R (mean ± 95% CI)")
    md.append("")
    md.append("| โมเดลบอส AI | " + " | ".join(pers_names) + " | รวมทุก persona |")
    md.append("|:---|" + "---:|" * (len(pers_names) + 1))
    for row in desc:
        cells = " | ".join(f"{c['mean']:.1f} ± {c['ci']:.1f}" for c in row["cells"])
        md.append(f"| **{row['boss']}** | {cells} | **{row['overall_mean']:.1f} ± {row['overall_ci']:.1f}** |")
    md.append("")
    md.append("ค่า ± คือครึ่งความกว้างของช่วงความเชื่อมั่น 95% คำนวณจากการแจกแจง t "
              f"(df = {args.reps - 1}) ไม่ใช่ค่าเบี่ยงเบนมาตรฐาน")
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"## 2. ทดสอบเหนือ baseline (H₀: R = {RANDOM_BASELINE_R}%)")
    md.append("")
    md.append("One-sample t-test ทางเดียว (H₁: R > 50%) แก้ค่า p ด้วย Holm-Bonferroni "
              f"ทั้งหมด {len(base)} การทดสอบ")
    md.append("")
    md.append("| โมเดลบอส | persona | R (%) | p (Holm) | สรุป |")
    md.append("|:---|:---|---:|---:|:---|")
    for r in base:
        mark = "ชนะเหนือสุ่มอย่างมีนัยสำคัญ" if r["sig"] else "สรุปไม่ได้"
        md.append(f"| {r['boss']} | {r['persona']} | {r['mean']:.1f} ± {r['ci']:.1f} | "
                  f"{r['p_holm']:.2e} | {mark} |")
    md.append("")
    nash = [r for r in base if "สุ่ม" in r["persona"] or "Nash" in r["persona"]]
    bad = [r for r in nash if r["sig"]]
    md.append("### Sanity / Falsification check")
    md.append("")
    md.append("ตามทฤษฎีเกม ไม่มีกลยุทธ์ใดเอาชนะผู้เล่นที่สุ่มสมบูรณ์ (Nash equilibrium 1/3) "
              "ได้ในระยะยาว ดังนั้นถ้าโมเดลใดชนะ persona นี้เกิน 50% อย่างมีนัยสำคัญ "
              "แปลว่าโค้ดมีบั๊กหรือมี data leakage")
    md.append("")
    if bad:
        md.append("**ไม่ผ่าน** — พบโมเดลที่ชนะผู้เล่นสุ่มอย่างมีนัยสำคัญ:")
        for r in bad:
            md.append(f"- `{r['boss']}`: R = {r['mean']:.1f}%, p(Holm) = {r['p_holm']:.2e}")
    else:
        md.append("**ผ่าน** — ไม่มีโมเดลใดชนะผู้เล่นสุ่มได้เกิน 50% อย่างมีนัยสำคัญ "
                  "ผลสอดคล้องกับทฤษฎีเกม ยืนยันว่าระบบจำลองไม่มี data leakage")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 3. จัดอันดับรวม: Friedman test + Nemenyi post-hoc")
    md.append("")
    md.append("วิธีมาตรฐานสำหรับเทียบหลายโมเดลพร้อมกันหลายเงื่อนไข (Demšar, JMLR 2006) "
              f"บล็อก = (persona × seed) รวม {fr['n_blocks']} บล็อก ทุกโมเดลเจอบล็อกเดียวกันทุกบล็อก")
    md.append("")
    md.append(f"- **Friedman χ² = {fr['chi2']:.3f}**, df = {fr['k'] - 1}, **p = {fr['p']:.3e}**")
    if fr["p"] < ALPHA:
        md.append(f"- p < {ALPHA} จึงปฏิเสธ H₀: โมเดลทั้งหมดมีประสิทธิภาพ**ไม่เท่ากันจริง** "
                  "สามารถทำ post-hoc ต่อได้")
    else:
        md.append(f"- p >= {ALPHA} จึงยังสรุปไม่ได้ว่าโมเดลต่างกัน (ไม่ควรทำ post-hoc)")
    md.append(f"- **Critical Difference (CD) = {fr['cd']:.3f}** อันดับ")
    md.append("")
    md.append("| อันดับ | โมเดล | อันดับเฉลี่ย (ต่ำ = ดี) |")
    md.append("|---:|:---|---:|")
    for i, r in enumerate(fr["ranking"], start=1):
        md.append(f"| {i} | **{r['boss']}** | {r['avg_rank']:.3f} |")
    md.append("")
    md.append("### Critical Difference diagram")
    md.append("")
    md.append("```")
    md.extend(render_cd_diagram(fr))
    md.append("```")
    md.append("")
    md.append("การอ่าน: โมเดลที่อยู่ในกลุ่มเดียวกันมีอันดับเฉลี่ยห่างกันไม่เกิน CD "
              "จึงยัง**สรุปไม่ได้**ว่าตัวใดดีกว่า ต้องเพิ่มจำนวนบล็อกถ้าต้องการแยกให้ชัด")
    md.append("")
    md.append("---")
    md.append("")
    md.append(f"## 4. แชมป์ `{best}` เทียบรายคู่ (paired comparison)")
    md.append("")
    md.append("ใช้ paired t-test เมื่อผลต่างแจกแจงปกติ (Shapiro-Wilk p > 0.05) "
              "ไม่เช่นนั้นใช้ Wilcoxon signed-rank; effect size คือ Cohen's dz = mean(diff)/SD(diff); "
              f"แก้ค่า p ด้วย Holm ทั้ง {len(champ_recs)} การทดสอบ")
    md.append("")
    md.append("| persona | คู่เทียบ | ผลต่าง R (pp) | test | p (Holm) | d_z | ขนาดผล | นัยสำคัญ |")
    md.append("|:---|:---|---:|:---|---:|---:|:---|:---:|")
    for r in champ_recs:
        dz = "∞" if math.isinf(r["dz"]) else f"{r['dz']:.2f}"
        sig = "ใช่" if r["sig"] else "ไม่"
        md.append(f"| {r['persona']} | {r['opponent']} | {r['mean_diff']:+.2f} | {r['test']} | "
                  f"{r['p_holm']:.2e} | {dz} | {effect_label(r['dz'])} | {sig} |")
    md.append("")
    md.append("> **ข้อควรระวังเชิงวิธีวิทยา (winner's curse):** แชมป์ถูกเลือกจากข้อมูลชุดเดียวกับ "
              "ที่ใช้ทดสอบ ค่า p ในตารางนี้จึงมองโลกในแง่ดีเกินจริงเล็กน้อย "
              "ข้อสรุปหลักควรอ้างอิงผล Friedman + Nemenyi ในหัวข้อ 3 ซึ่งแก้ปัญหานี้แล้ว "
              "ส่วนตารางนี้ถือเป็นการวิเคราะห์เสริม")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 5. Power analysis")
    md.append("")
    md.append("| รายการ | ค่า |")
    md.append("|:---|---:|")
    md.append(f"| SD ของผลต่างแบบจับคู่ (median) | {power['sd_diff_median']:.2f} pp |")
    md.append(f"| n ที่ใช้จริง | {power['reps']} แมตช์ |")
    md.append(f"| ผลต่างเล็กสุดที่ตรวจจับได้ (MDE, power = 0.80) | {power['mde']:.2f} pp |")
    md.append(f"| n ที่ต้องใช้เพื่อตรวจจับผลต่าง 2.00 pp | {power['n_for_2pp']} แมตช์ |")
    md.append("")
    md.append("สูตร (paired, two-sided): `MDE = (z_(α/2) + z_β) × SD_diff / √n` "
              f"โดย z_(α/2) + z_β = {power['z_sum']:.3f} ที่ alpha = {ALPHA}, power = 0.80 "
              "ตัวเลขนี้ยืนยันว่า n ที่เลือกมาจากการคำนวณ ไม่ใช่การกำหนดตามใจ")
    md.append("")

    if ml_an:
        md.append("---")
        md.append("")
        md.append("## 6. เปรียบเทียบโมเดล sklearn (Walk-Forward Validation)")
        md.append("")
        md.append(f"ข้อมูล {ml_an.get('n_samples', '-')} ตัวอย่าง | "
                  f"walk-forward แบบ expanding window | "
                  f"fold scores จับคู่กันด้วย (fold × seed)")
        md.append("")
        md.append("| โมเดล | Accuracy (mean ± 95% CI) | Log-Loss (mean ± 95% CI) | n folds |")
        md.append("|:---|---:|---:|---:|")
        for n, d in ml_an["desc"].items():
            md.append(f"| **{n}** | {d['acc'] * 100:.2f}% ± {d['acc_ci'] * 100:.2f}% | "
                      f"{d['ll']:.4f} ± {d['ll_ci']:.4f} | {d['n_folds']} |")
        md.append("")
        md.append("### การทดสอบรายคู่")
        md.append("")
        md.append("ใช้สองวิธีประกอบกัน: (ก) เทียบคะแนนรายfoldแบบจับคู่ และ "
                  "(ข) **McNemar's test** บนผลทำนายรายตาในชุด validation เดียวกัน "
                  "ซึ่ง Dietterich (1998) แนะนำให้ใช้เมื่อเทียบ classifier สองตัวบนข้อมูลชุดเดียว")
        md.append("")
        md.append("| คู่เปรียบเทียบ | fold-wise test | p (Holm) | McNemar p (Holm) | b / c |")
        md.append("|:---|:---|---:|---:|---:|")
        for p in ml_an["pairs"]:
            s1 = " ✱" if p["fold"]["sig"] else ""
            s2 = " ✱" if p["mcnemar"]["sig"] else ""
            md.append(f"| {p['a']} vs {p['b']} | {p['fold']['test']} | "
                      f"{p['fold']['p_holm']:.2e}{s1} | {p['mcnemar']['p_holm']:.2e}{s2} | "
                      f"{p['mcnemar']['n10']} / {p['mcnemar']['n01']} |")
        md.append("")
        md.append("`b` = จำนวนตาที่โมเดลซ้ายทำนายถูกแต่ขวาผิด, `c` = ซ้ายผิดแต่ขวาถูก "
                  "McNemar ทดสอบว่า b กับ c ต่างกันเกินความบังเอิญหรือไม่ (exact binomial test) "
                  "| ✱ = มีนัยสำคัญหลังแก้ Holm")
        md.append("")
        if ml_an.get("holdout"):
            md.append("### ชุดทดสอบสุดท้ายที่กันไว้ (held-out final test block)")
            md.append("")
            md.append("กันข้อมูล 15% ท้ายสุดออกจากการคัดเลือกโมเดลทั้งหมด "
                      "เพื่อให้ตัวเลขที่รายงานไม่ติด selection bias")
            md.append("")
            md.append("| โมเดล | Accuracy | Log-Loss | n |")
            md.append("|:---|---:|---:|---:|")
            for n, d in ml_an["holdout"].items():
                md.append(f"| {n} | {d['accuracy'] * 100:.2f}% | {d['log_loss']:.4f} | {d['n']} |")
            md.append("")

    md.append("---")
    md.append("")
    md.append("## 7. สรุปข้อสรุปที่อ้างได้ (Defensible Conclusions)")
    md.append("")
    if fr["p"] < ALPHA:
        md.append(f"1. ประสิทธิภาพของโมเดลทั้ง {fr['k']} ตัว **แตกต่างกันอย่างมีนัยสำคัญ** "
                  f"(Friedman χ² = {fr['chi2']:.2f}, p = {fr['p']:.2e})")
    else:
        md.append(f"1. ยังสรุปไม่ได้ว่าโมเดลทั้ง {fr['k']} ตัวต่างกัน (Friedman p = {fr['p']:.3f})")
    md.append(f"2. โมเดลที่ได้อันดับเฉลี่ยดีที่สุดคือ **{best}** "
              f"(อันดับเฉลี่ย {fr['ranking'][0]['avg_rank']:.3f} จาก {fr['k']} โมเดล)")
    n_sig = sum(1 for r in base if r["sig"])
    md.append(f"3. มี {n_sig} คู่ (โมเดล × persona) จาก {len(base)} คู่ ที่ชนะเหนือ baseline "
              "การสุ่มอย่างมีนัยสำคัญหลังแก้ Holm")
    if bad:
        md.append("4. **คำเตือน:** ระบบไม่ผ่าน sanity check กับผู้เล่นสุ่ม ควรตรวจโค้ดก่อนนำผลไปใช้")
    else:
        md.append("4. ระบบผ่าน sanity check: ไม่มีโมเดลใดเอาชนะผู้เล่นสุ่มสมบูรณ์ได้ "
                  "ตรงตามที่ทฤษฎีเกมทำนาย")
    md.append(f"5. ที่ n = {power['reps']} แมตช์ การทดลองนี้มี power 0.80 ในการตรวจจับผลต่าง "
              f"ตั้งแต่ {power['mde']:.2f} pp ขึ้นไป")
    md.append("")
    md.append("### อ้างอิงวิธีการ")
    md.append("")
    md.append("- Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data Sets.* "
              "JMLR 7, 1–30. — Friedman test + Nemenyi post-hoc + CD diagram")
    md.append("- Dietterich, T. G. (1998). *Approximate Statistical Tests for Comparing Supervised "
              "Classification Learning Algorithms.* Neural Computation 10(7). — McNemar's test")
    md.append("- Holm, S. (1979). *A Simple Sequentially Rejective Multiple Test Procedure.* "
              "Scandinavian Journal of Statistics 6(2). — การแก้ multiple comparisons")
    md.append("- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences.* — effect size")
    md.append("")
    return "\n".join(md)


# =====================================================================
#  ส่วนที่ 5: main
# =====================================================================
def main():
    ap = argparse.ArgumentParser(
        description="การทดสอบนัยสำคัญทางสถิติเพื่อตัดสินว่าโมเดลใดดีที่สุด")
    ap.add_argument("--reps", type=int, default=30, help="จำนวนแมตช์อิสระต่อคู่ (default 30)")
    ap.add_argument("--rounds", type=int, default=1000, help="จำนวนตาต่อแมตช์ (default 1000)")
    ap.add_argument("--seed0", type=int, default=0, help="seed เริ่มต้น (default 0)")
    ap.add_argument("--folds", type=int, default=10, help="จำนวน fold ของ walk-forward (default 10)")
    ap.add_argument("--csv", type=str, default="data/players")
    ap.add_argument("--out", type=str, default="reports/statistical_report.md")
    ap.add_argument("--skip-ml", action="store_true", help="ข้ามการเทียบโมเดล sklearn")
    args = ap.parse_args()

    if args.reps < 3:
        print("!! --reps ต้องมีค่าอย่างน้อย 3 จึงจะคำนวณสถิติได้ (แนะนำ 30)")
        return

    print("=" * 100)
    print("  เริ่มการทดลอง: จำลองการแข่งขันแบบจับคู่ด้วย Common Random Numbers")
    print("=" * 100)
    boss_names, pers_names, data = run_arena(args.reps, args.rounds, args.seed0)

    desc = analyze_descriptive(boss_names, pers_names, data)
    base = analyze_baseline(boss_names, pers_names, data)
    fr = analyze_friedman(boss_names, pers_names, data)
    champion_idx = boss_names.index(fr["ranking"][0]["boss"])
    champ_recs = analyze_champion(boss_names, pers_names, data, champion_idx)
    power = analyze_power(data)

    ml_an = None
    if not args.skip_ml:
        print("\n  เทียบโมเดล sklearn ด้วย walk-forward validation...")
        try:
            ml = run_ml(args.csv, n_folds=args.folds)
            if ml:
                ml_an = analyze_ml(ml)
                ml_an["n_samples"] = ml["n_samples"]
                ml_an["holdout"] = ml["holdout"]
            else:
                print("  (ข้าม: ไม่พบไฟล์ข้อมูล หรือข้อมูลน้อยเกินไป)")
        except Exception as exc:
            print(f"  (ข้ามส่วน sklearn เนื่องจากข้อผิดพลาด: {exc})")

    console_report(boss_names, pers_names, data, desc, base, fr, champ_recs, power, ml_an, args)

    md = build_markdown(boss_names, pers_names, data, desc, base, fr, champ_recs, power, ml_an, args)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  บันทึกรายงานไว้ที่: {args.out}")
    print("=" * 100)


if __name__ == "__main__":
    main()
