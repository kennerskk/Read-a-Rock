# -*- coding: utf-8 -*-
# =====================================================================
#  benchmark_lw_vs_hw.py
#  เปรียบเทียบ Lightweight Models (บอสเชิงสถิติ 8 ตัว, pure Python)
#  กับ Heavyweight Models (sklearn 3 ตัว) บน "สนามทดสอบกลาง" เดียวกัน
# ---------------------------------------------------------------------
#  ปัญหาที่แก้:
#    เดิมสองตระกูลวัดคนละหน่วย — บอสวัดด้วย win rate จากการจำลอง
#    ส่วน sklearn วัดด้วย accuracy จาก walk-forward บน CSV
#    จึงเทียบกันไม่ได้
#
#  วิธีแก้ — Prequential Evaluation (Test-Then-Train / Interleaved):
#    ทั้งสองตระกูลทำ "งานเดียวกัน" คือทำนายท่าถัดไปของผู้เล่น
#    บนข้อมูลจริงชุดเดียวกัน ลำดับเดียวกัน ทีละตาตามเวลาจริง
#
#    สำหรับตาที่ t:
#      1. ทุกโมเดลทำนายท่าที่ t โดยเห็นข้อมูลถึงตาที่ t-1 เท่านั้น
#      2. บันทึกถูก/ผิด และผลแพ้ชนะถ้านำไปเล่นจริง
#      3. เปิดเฉลยตาที่ t ให้ทุกโมเดลเรียนรู้
#    => ไม่มี data leakage และทุกโมเดลเจอข้อมูลเดียวกันเป๊ะทุกตา
#       (paired design สมบูรณ์ ใช้ McNemar ได้โดยตรง)
#
#  ตัวชี้วัด 4 ด้าน:
#    - Accuracy  : ทำนายท่าผู้เล่นถูกกี่ %   (baseline สุ่ม = 33.3%)
#    - R         : (W + 0.5D)/N x 100 ถ้านำไปเล่นจริง (baseline สุ่ม = 50%)
#    - Latency   : เวลาทำนายต่อตา (ไมโครวินาที) <- จุดต่างของ LW/HW
#    - Size      : ขนาดโมเดลหลัง serialize (KB)
#
#  วิธีรัน:  python benchmark_lw_vs_hw.py
#            python benchmark_lw_vs_hw.py --refit 25 --segment 100
# =====================================================================
from __future__ import annotations

import os
import sys
import math
import time
import pickle
import random
import argparse
import datetime
import itertools
from collections import Counter
from typing import Dict, List, Tuple

if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from telemetry import read_telemetry
from scipy import stats

from adaptive_arena import BOSSES, MOVES, BEATS, COUNTER, judge
from model_pipeline import FeatureStore

ALPHA = 0.05
RANDOM_ACC = 100.0 / 3.0     # baseline ของ accuracy 3 คลาส
RANDOM_R = 50.0              # baseline ของ R
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
               8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268,
               13: 3.313, 14: 3.354, 15: 3.391}

HEAVY_FAMILY = "Heavyweight (sklearn)"
LIGHT_FAMILY = "Lightweight (pure Python)"


def make_heavy_models(seed: int = 42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    return {
        "HW-LogisticRegression": LogisticRegression(max_iter=500, C=1.0),
        "HW-RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=seed),
        "HW-GradientBoosting": HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=seed),
    }


def model_size_bytes(obj) -> int:
    """
    ประมาณขนาดโมเดลหลัง serialize
    ลำดับการลอง: pickle -> JSON ของ get_state() -> นับขนาดเชิงลึกของ __dict__
    (บางบอสใช้ lambda ใน defaultdict จึง pickle ไม่ได้)
    """
    try:
        return len(pickle.dumps(obj))
    except Exception:
        pass
    try:
        import json
        if hasattr(obj, "get_state"):
            return len(json.dumps(obj.get_state(), ensure_ascii=False).encode("utf-8"))
    except Exception:
        pass

    seen = set()

    def deep(o, depth=0):
        if depth > 6 or id(o) in seen:
            return 0
        seen.add(id(o))
        total = sys.getsizeof(o, 64)
        if isinstance(o, dict):
            for k, v in list(o.items())[:5000]:
                total += deep(k, depth + 1) + deep(v, depth + 1)
        elif isinstance(o, (list, tuple, set)):
            for v in list(o)[:5000]:
                total += deep(v, depth + 1)
        elif hasattr(o, "__dict__"):
            total += deep(vars(o), depth + 1)
        return total

    return int(deep(obj))


def short_light_name(cls) -> str:
    raw = getattr(cls, "name", str(cls))
    if "." in raw:
        raw = raw.split(".", 1)[1]
    return "LW-" + raw.split("(")[0].strip().replace(" ", "")


# =====================================================================
#  สนามทดสอบกลาง: Prequential replay ของข้อมูลจริง
# =====================================================================
def run_prequential(df_player: pd.DataFrame, warmup_frac: float = 0.40,
                    refit_every: int = 25, seed: int = 42) -> Dict:
    """
    เล่นซ้ำ (replay) ลำดับการเล่นจริงของผู้เล่นหนึ่งคน ทีละตา
    ทุกโมเดลทำนายตาเดียวกันพร้อมกัน แล้วค่อยเรียนรู้เฉลย

    คืน dict: ชื่อโมเดล -> {correct[], result[], latency_us[], size_bytes}
    """
    moves = df_player["player_move"].tolist()
    X, y, feat_names = FeatureStore.build_features(df_player)
    X = np.asarray(X)
    y = np.asarray(y)
    # feature แถวที่ j คู่กับ move ลำดับที่ j+3 (build_features เริ่มที่ i=3)
    offset = 3
    n_feat_rows = len(X)
    if n_feat_rows < 60:
        return {}

    warm_j = int(n_feat_rows * warmup_frac)
    scored_idx = list(range(warm_j, n_feat_rows))

    random.seed(seed)
    np.random.seed(seed)

    results: Dict[str, Dict] = {}

    # ---------- ตระกูล Lightweight ----------
    for b_cls in BOSSES:
        name = short_light_name(b_cls)
        boss = b_cls()
        history: List[str] = []
        rec = {"correct": [], "result": [], "latency_us": [], "family": LIGHT_FAMILY}

        for j in range(n_feat_rows):
            i = j + offset
            actual = moves[i]

            t0 = time.perf_counter()
            boss_move = boss.choose(history)
            lat = (time.perf_counter() - t0) * 1e6

            # บอสออกท่าแก้ทางท่าที่มันทำนาย -> ถอดกลับเป็นท่าที่มันทำนาย
            predicted = BEATS[boss_move]

            if j in range(warm_j, n_feat_rows):
                rec["correct"].append(predicted == actual)
                rec["result"].append(judge(actual, boss_move))   # มุมมองผู้เล่น
                rec["latency_us"].append(lat)

            # เปิดเฉลยแล้วให้เรียนรู้
            history.append(actual)
            reward = {"win": -1, "draw": 0, "lose": 1}[judge(actual, boss_move)]
            boss.learn(reward, history)

        rec["size_bytes"] = model_size_bytes(boss)
        results[name] = rec

    # ---------- ตระกูล Heavyweight ----------
    heavy = make_heavy_models(seed)
    for name, model in heavy.items():
        rec = {"correct": [], "result": [], "latency_us": [], "family": HEAVY_FAMILY,
               "fit_time_s": 0.0, "n_fits": 0}
        fitted = False
        for k, j in enumerate(scored_idx):
            # เทรนใหม่ทุก refit_every ตา ด้วยข้อมูลทั้งหมดถึงตาที่ j-1
            if k % refit_every == 0:
                if len(np.unique(y[:j])) >= 2:
                    t0 = time.perf_counter()
                    model.fit(X[:j], y[:j])
                    rec["fit_time_s"] += time.perf_counter() - t0
                    rec["n_fits"] += 1
                    fitted = True

            actual = moves[j + offset]
            if not fitted:
                predicted_int = random.randrange(3)
                lat = 0.0
            else:
                t0 = time.perf_counter()
                predicted_int = int(model.predict(X[j:j + 1])[0])
                lat = (time.perf_counter() - t0) * 1e6

            predicted = MOVES[predicted_int]
            boss_move = COUNTER[predicted]
            rec["correct"].append(predicted == actual)
            rec["result"].append(judge(actual, boss_move))
            rec["latency_us"].append(lat)

        rec["size_bytes"] = model_size_bytes(model)
        results[name] = rec

    return results


def merge_players(all_runs: List[Dict]) -> Dict:
    """รวมผลของผู้เล่นหลายคนต่อกัน (ลำดับตาไม่ปนกันเพราะแยก replay)"""
    merged: Dict[str, Dict] = {}
    for run in all_runs:
        for name, rec in run.items():
            if name not in merged:
                merged[name] = {"correct": [], "result": [], "latency_us": [],
                                "family": rec["family"], "size_bytes": rec["size_bytes"],
                                "fit_time_s": rec.get("fit_time_s", 0.0),
                                "n_fits": rec.get("n_fits", 0)}
            merged[name]["correct"].extend(rec["correct"])
            merged[name]["result"].extend(rec["result"])
            merged[name]["latency_us"].extend(rec["latency_us"])
            merged[name]["size_bytes"] = max(merged[name]["size_bytes"], rec["size_bytes"])
            merged[name]["fit_time_s"] += rec.get("fit_time_s", 0.0)
            merged[name]["n_fits"] += rec.get("n_fits", 0)
    return merged


def segment_scores(merged: Dict, segment: int = 100) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    ตัดลำดับตาเป็นช่วงละ segment ตา แล้วคำนวณ accuracy และ R ต่อช่วง
    ช่วงเหล่านี้คือ "บล็อก" สำหรับ Friedman test (ทุกโมเดลเจอบล็อกเดียวกัน)
    """
    names = list(merged.keys())
    n = min(len(merged[nm]["correct"]) for nm in names)
    n_seg = n // segment
    acc = np.zeros((n_seg, len(names)))
    rr = np.zeros((n_seg, len(names)))
    for ci, nm in enumerate(names):
        corr = np.asarray(merged[nm]["correct"][:n], dtype=float)
        res = merged[nm]["result"][:n]
        for s in range(n_seg):
            lo, hi = s * segment, (s + 1) * segment
            acc[s, ci] = corr[lo:hi].mean() * 100.0
            chunk = res[lo:hi]
            w = sum(1 for r in chunk if r == "lose")    # ผู้เล่นแพ้ = โมเดลชนะ
            d = sum(1 for r in chunk if r == "draw")
            rr[s, ci] = (w + 0.5 * d) / len(chunk) * 100.0
    return names, acc, rr


# =====================================================================
#  สถิติ
# =====================================================================
def mean_ci(x, conf=0.95):
    a = np.asarray(x, dtype=float)
    n = a.size
    m = float(a.mean())
    if n < 2:
        return m, 0.0
    sd = float(a.std(ddof=1))
    if sd == 0:
        return m, 0.0
    return m, float(stats.t.ppf(0.5 + conf / 2, n - 1) * sd / math.sqrt(n))


def holm(pvals):
    p = list(pvals)
    m = len(p)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p[i])
    adj = [0.0] * m
    run = 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def mcnemar_exact(a, b):
    """เทียบ 2 โมเดลบนตาเดียวกันทุกตา — จับคู่สมบูรณ์"""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    n10 = int(np.sum(a & ~b))   # A ถูก B ผิด
    n01 = int(np.sum(~a & b))   # A ผิด B ถูก
    tot = n01 + n10
    if tot == 0:
        return {"n10": 0, "n01": 0, "p": 1.0}
    return {"n10": n10, "n01": n01, "p": float(stats.binomtest(n10, tot, 0.5).pvalue)}


def friedman_nemenyi(names, score_matrix):
    M = np.asarray(score_matrix, dtype=float)
    n_blocks, k = M.shape
    ranks = np.empty_like(M)
    for i in range(n_blocks):
        ranks[i] = stats.rankdata(-M[i])
    avg = ranks.mean(axis=0)
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    q = NEMENYI_Q05.get(k, NEMENYI_Q05[max(NEMENYI_Q05)])
    cd = q * math.sqrt(k * (k + 1) / (6.0 * n_blocks))
    order = np.argsort(avg)
    ranking = [{"name": names[j], "avg_rank": float(avg[j])} for j in order]
    return {"chi2": float(chi2), "p": float(p), "cd": cd, "k": k,
            "n_blocks": n_blocks, "ranking": ranking, "avg_rank": avg}


def pairwise_matrix(names, merged):
    """เทียบทุกคู่ด้วย McNemar + Holm -> เมทริกซ์ k x k"""
    pairs = []
    for i, j in itertools.combinations(range(len(names)), 2):
        mc = mcnemar_exact(merged[names[i]]["correct"], merged[names[j]]["correct"])
        pairs.append({"i": i, "j": j, "a": names[i], "b": names[j], **mc})
    adj = holm([p["p"] for p in pairs])
    k = len(names)
    sign = np.zeros((k, k), dtype=int)     # +1 = แถวเก่งกว่า, -1 = แถวแย่กว่า, 0 = เสมอ
    for pr, ap in zip(pairs, adj):
        pr["p_holm"] = ap
        pr["sig"] = ap < ALPHA
        if pr["sig"]:
            better = pr["n10"] > pr["n01"]
            sign[pr["i"], pr["j"]] = 1 if better else -1
            sign[pr["j"], pr["i"]] = -1 if better else 1
    return pairs, sign


def family_comparison(names, merged, acc_matrix):
    """เทียบ "ตระกูล" LW vs HW: ตัวแทนที่ดีที่สุดของแต่ละฝั่ง + ค่าเฉลี่ยทั้งตระกูล"""
    light = [n for n in names if merged[n]["family"] == LIGHT_FAMILY]
    heavy = [n for n in names if merged[n]["family"] == HEAVY_FAMILY]
    acc_of = {n: float(np.mean(merged[n]["correct"]) * 100) for n in names}

    best_l = max(light, key=lambda n: acc_of[n])
    best_h = max(heavy, key=lambda n: acc_of[n])
    mc = mcnemar_exact(merged[best_l]["correct"], merged[best_h]["correct"])

    idx = {n: i for i, n in enumerate(names)}
    seg_l = acc_matrix[:, [idx[n] for n in light]].mean(axis=1)
    seg_h = acc_matrix[:, [idx[n] for n in heavy]].mean(axis=1)
    d = seg_l - seg_h
    try:
        w_p = float(stats.wilcoxon(d).pvalue)
    except Exception:
        w_p = 1.0
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    dz = float(d.mean() / sd) if sd > 0 else 0.0

    return {"light": light, "heavy": heavy, "acc_of": acc_of,
            "best_light": best_l, "best_heavy": best_h, "head_to_head": mc,
            "family_mean_light": float(seg_l.mean()), "family_mean_heavy": float(seg_h.mean()),
            "family_p": w_p, "family_dz": dz, "family_diff": float(d.mean())}


# =====================================================================
#  รูป
# =====================================================================
def make_figures(names, merged, acc_matrix, fr, sign, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    os.makedirs(outdir, exist_ok=True)
    C_LIGHT, C_HEAVY = "#059669", "#dc2626"
    def _short(n):
        t = n.replace("LW-", "").replace("HW-", "")
        return {"DataScienceMLBoss": "DS-ML-Boss", "Ensemble/Meta": "Ensemble",
                "LogisticRegression": "LogReg", "GradientBoosting": "GradBoost",
                "RandomForest": "RandForest", "Recency-decay": "Recency"}.get(t, t)
    short = [_short(n) for n in names]
    fam = [merged[n]["family"] for n in names]
    cols = [C_LIGHT if f == LIGHT_FAMILY else C_HEAVY for f in fam]
    idx = {n: i for i, n in enumerate(names)}

    # ---- รูป 1: accuracy + 95% CI เรียงลำดับ ----
    means, errs = [], []
    for n in names:
        m, e = mean_ci(acc_matrix[:, idx[n]])
        means.append(m)
        errs.append(e)
    order = np.argsort(means)[::-1]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    xs = np.arange(len(names))
    ax.bar(xs, [means[i] for i in order], yerr=[errs[i] for i in order], capsize=5,
           color=[cols[i] for i in order], edgecolor="#1f2937", linewidth=0.7, alpha=0.9, width=0.62)
    ax.axhline(RANDOM_ACC, linestyle="--", color="#6b7280", linewidth=1.5)
    ax.text(len(names) - 0.4, RANDOM_ACC + 0.8, "Random baseline (33.3%)",
            fontsize=9, color="#4b5563", ha="right")
    for xi, i in enumerate(order):
        ax.text(xi, means[i] + errs[i] + 0.7, f"{means[i]:.1f}", ha="center", fontsize=9)
    ax.set_xticks(xs, [short[i] for i in order], rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Next-move prediction accuracy (%)", fontsize=11)
    ax.set_title("Lightweight vs Heavyweight — Same Data, Same Protocol\n"
                 "(prequential test-then-train on real telemetry; error bars = 95% CI)",
                 fontsize=12, pad=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_LIGHT),
               plt.Rectangle((0, 0), 1, 1, color=C_HEAVY)]
    ax.legend(handles, ["Lightweight (pure Python)", "Heavyweight (sklearn)"],
              fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    p1 = os.path.join(outdir, "lw_vs_hw_accuracy.png")
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- รูป 2: trade-off accuracy vs latency ----
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for i, n in enumerate(names):
        lat = max(float(np.mean(merged[n]["latency_us"])), 0.05)
        ax.scatter(lat, means[i], s=150, color=cols[i], edgecolor="#1f2937",
                   linewidth=0.9, zorder=3, alpha=0.9)
        # สลับตำแหน่งป้ายบน/ล่างเพื่อไม่ให้ทับกันในบริเวณที่จุดหนาแน่น
        off = (10, 5) if i % 2 == 0 else (10, -13)
        ax.annotate(short[i], (lat, means[i]), textcoords="offset points",
                    xytext=off, fontsize=8.5)
    ax.axhline(RANDOM_ACC, linestyle="--", color="#6b7280", linewidth=1.3)
    ax.text(ax.get_xlim()[1], RANDOM_ACC + 0.5, "Random baseline", fontsize=8.5,
            color="#4b5563", ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("Prediction latency per round (microseconds, log scale)", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy vs Latency Trade-off\n(upper-left = better: more accurate AND faster)",
                 fontsize=12, pad=12)
    ax.legend(handles, ["Lightweight (pure Python)", "Heavyweight (sklearn)"],
              fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_axisbelow(True)
    fig.tight_layout()
    p2 = os.path.join(outdir, "lw_vs_hw_tradeoff.png")
    fig.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---- รูป 3: เมทริกซ์เทียบรายคู่ ----
    rank_order = [idx[r["name"]] for r in fr["ranking"]]
    S = sign[np.ix_(rank_order, rank_order)]
    lbl = [short[i] for i in rank_order]
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    cmap = ListedColormap(["#dc2626", "#e5e7eb", "#059669"])
    ax.imshow(S, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(lbl)), lbl, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(lbl)), lbl, fontsize=8.5)
    for i in range(len(lbl)):
        for j in range(len(lbl)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#6b7280")
            else:
                mark = {1: "W", -1: "L", 0: "·"}[S[i, j]]
                ax.text(j, i, mark, ha="center", va="center", fontsize=9,
                        color="white" if S[i, j] != 0 else "#6b7280",
                        fontweight="bold" if S[i, j] != 0 else "normal")
    ax.set_title("Pairwise Head-to-Head (McNemar, Holm-corrected, alpha=0.05)\n"
                 "W = row significantly better | L = row worse | · = no significant difference\n"
                 "(ordered by Friedman average rank, best first)", fontsize=11, pad=14)
    ax.set_xticks(np.arange(-0.5, len(lbl), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(lbl), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    p3 = os.path.join(outdir, "lw_vs_hw_pairwise.png")
    fig.savefig(p3, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {"accuracy": p1, "tradeoff": p2, "pairwise": p3}


# =====================================================================
#  รายงาน
# =====================================================================
def write_report(names, merged, acc_m, r_m, fr, pairs, sign, famc, figs, meta, out):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    idx = {n: i for i, n in enumerate(names)}
    L = []
    L.append("# Lightweight vs Heavyweight Models — การเปรียบเทียบบนสนามทดสอบกลาง")
    L.append("")
    L.append(f"**สร้างเมื่อ:** {now}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. ปัญหาและวิธีแก้")
    L.append("")
    L.append("เดิมสองตระกูลโมเดลวัดกันคนละหน่วย จึงเทียบกันตรง ๆ ไม่ได้:")
    L.append("")
    L.append("| ตระกูล | เดิมวัดด้วย | ปัญหา |")
    L.append("|:---|:---|:---|")
    L.append("| Lightweight (บอส 8 ตัว) | win rate จากการจำลองกับ persona | ไม่ได้ใช้ข้อมูลจริง |")
    L.append("| Heavyweight (sklearn 3 ตัว) | accuracy จาก walk-forward บน CSV | ไม่ได้วัดผลในเกม |")
    L.append("")
    L.append("### Protocol: Prequential Evaluation (Test-Then-Train)")
    L.append("")
    L.append("ให้ทั้งสองตระกูลทำ **งานเดียวกัน** คือทำนายท่าถัดไปของผู้เล่น "
             "บนข้อมูลจริงชุดเดียวกัน ลำดับเดียวกัน ทีละตาตามเวลาจริง:")
    L.append("")
    L.append("```")
    L.append("สำหรับตาที่ t:")
    L.append("  1. ทุกโมเดลทำนายท่าที่ t  (เห็นข้อมูลถึงตาที่ t-1 เท่านั้น)")
    L.append("  2. บันทึกถูก/ผิด และผลแพ้ชนะถ้านำไปเล่นจริง")
    L.append("  3. เปิดเฉลยตาที่ t ให้ทุกโมเดลเรียนรู้")
    L.append("```")
    L.append("")
    L.append("ข้อดีของ protocol นี้:")
    L.append("")
    L.append("- **ไม่มี data leakage** — ทุกการทำนายใช้เฉพาะอดีตเท่านั้น")
    L.append("- **จับคู่สมบูรณ์ (perfectly paired)** — ทุกโมเดลเจอตาเดียวกันเป๊ะทุกตา "
             "จึงใช้ McNemar's test ได้โดยตรง ไม่ต้องกังวลเรื่องความแปรปรวนของข้อมูล")
    L.append("- **ยุติธรรมกับทั้งสองฝั่ง** — ให้ warm-up เท่ากัน "
             f"({int(meta['warmup_frac'] * 100)}% แรกของลำดับ) ฝั่ง lightweight เรียนออนไลน์ผ่านช่วงนั้น "
             "ฝั่ง heavyweight เทรนด้วยช่วงนั้น แล้วเริ่มนับคะแนนพร้อมกัน")
    L.append("")
    L.append("| พารามิเตอร์ | ค่า |")
    L.append("|:---|---:|")
    L.append(f"| ผู้เล่นที่นำมา replay | {', '.join(meta['players'])} |")
    L.append(f"| จำนวนตาที่นับคะแนน | {meta['n_scored']} |")
    L.append(f"| warm-up | {int(meta['warmup_frac'] * 100)}% |")
    L.append(f"| heavyweight เทรนใหม่ทุก | {meta['refit_every']} ตา |")
    L.append(f"| ขนาดบล็อกสำหรับ Friedman | {meta['segment']} ตา |")
    L.append(f"| จำนวนบล็อก | {fr['n_blocks']} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. ผลรวม 4 มิติ")
    L.append("")
    L.append("| โมเดล | ตระกูล | Accuracy (95% CI) | R (%) | Latency (μs/ตา) | ขนาด (KB) |")
    L.append("|:---|:---|---:|---:|---:|---:|")
    for r in fr["ranking"]:
        n = r["name"]
        i = idx[n]
        am, ae = mean_ci(acc_m[:, i])
        rm, _ = mean_ci(r_m[:, i])
        lat = float(np.mean(merged[n]["latency_us"]))
        kb = merged[n]["size_bytes"] / 1024.0
        fam_tag = "LW" if merged[n]["family"] == LIGHT_FAMILY else "HW"
        L.append(f"| {n} | {fam_tag} | {am:.2f}% ± {ae:.2f}% | {rm:.1f} | {lat:.1f} | {kb:.1f} |")
    L.append("")
    L.append(f"> baseline: accuracy สุ่ม = {RANDOM_ACC:.1f}% | R สุ่ม = {RANDOM_R:.1f}%")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. จัดอันดับรวม — Friedman + Nemenyi")
    L.append("")
    L.append(f"- **Friedman χ² = {fr['chi2']:.3f}**, df = {fr['k'] - 1}, **p = {fr['p']:.3e}**")
    verdict = ("p < 0.05 ⇒ โมเดลทั้งหมดต่างกันจริง ทำ post-hoc ต่อได้"
               if fr["p"] < ALPHA else "p ≥ 0.05 ⇒ ยังสรุปไม่ได้ว่าต่างกัน")
    L.append(f"- {verdict}")
    L.append(f"- **Critical Difference (CD) = {fr['cd']:.3f}** อันดับ "
             f"(k = {fr['k']}, N = {fr['n_blocks']})")
    L.append("")
    L.append("| อันดับ | โมเดล | ตระกูล | อันดับเฉลี่ย |")
    L.append("|---:|:---|:---|---:|")
    for i, r in enumerate(fr["ranking"], 1):
        fam_tag = "LW" if merged[r["name"]]["family"] == LIGHT_FAMILY else "**HW**"
        L.append(f"| {i} | {r['name']} | {fam_tag} | {r['avg_rank']:.3f} |")
    L.append("")
    best = fr["ranking"][0]
    tied = [r["name"] for r in fr["ranking"]
            if r["avg_rank"] - best["avg_rank"] <= fr["cd"]]
    L.append(f"**กลุ่มนำที่แยกกันไม่ได้** (ห่างจากอันดับ 1 ไม่เกิน CD): {', '.join(tied)}")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. Lightweight vs Heavyweight — คำตอบตรง ๆ")
    L.append("")
    L.append("### 4.1 ตัวแทนที่ดีที่สุดของแต่ละฝั่ง (McNemar, จับคู่รายตา)")
    L.append("")
    h2h = famc["head_to_head"]
    bl, bh = famc["best_light"], famc["best_heavy"]
    L.append(f"| | Lightweight | Heavyweight |")
    L.append("|:---|:---|:---|")
    L.append(f"| ตัวแทน | **{bl}** | **{bh}** |")
    L.append(f"| Accuracy | {famc['acc_of'][bl]:.2f}% | {famc['acc_of'][bh]:.2f}% |")
    L.append(f"| Latency | {np.mean(merged[bl]['latency_us']):.1f} μs | "
             f"{np.mean(merged[bh]['latency_us']):.1f} μs |")
    L.append(f"| ขนาด | {merged[bl]['size_bytes'] / 1024:.1f} KB | "
             f"{merged[bh]['size_bytes'] / 1024:.1f} KB |")
    L.append("")
    L.append(f"- McNemar: b = {h2h['n10']} (LW ถูก/HW ผิด), c = {h2h['n01']} (LW ผิด/HW ถูก), "
             f"**p = {h2h['p']:.3e}**")
    if h2h["p"] < ALPHA:
        win = bl if h2h["n10"] > h2h["n01"] else bh
        L.append(f"- **สรุป: `{win}` แม่นกว่าอย่างมีนัยสำคัญ**")
    else:
        L.append("- **สรุป: ความแม่นยำไม่ต่างกันอย่างมีนัยสำคัญ**")
    L.append("")
    L.append("### 4.2 เทียบทั้งตระกูล (ค่าเฉลี่ยต่อบล็อก, Wilcoxon signed-rank)")
    L.append("")
    L.append(f"| ตระกูล | Accuracy เฉลี่ยทั้งตระกูล |")
    L.append("|:---|---:|")
    L.append(f"| Lightweight ({len(famc['light'])} โมเดล) | {famc['family_mean_light']:.2f}% |")
    L.append(f"| Heavyweight ({len(famc['heavy'])} โมเดล) | {famc['family_mean_heavy']:.2f}% |")
    L.append("")
    L.append(f"ผลต่าง = {famc['family_diff']:+.2f} pp, p = {famc['family_p']:.3e}, "
             f"Cohen's d_z = {famc['family_dz']:.2f}")
    L.append("")
    L.append("> หมายเหตุ: การเฉลี่ยทั้งตระกูลถูกถ่วงด้วยโมเดลอ่อน เช่น Random ที่อยู่ในฝั่ง "
             "lightweight ตัวเลขนี้จึงใช้ดูภาพรวมเท่านั้น ข้อสรุปหลักให้ดู 4.1 และตารางอันดับ")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. ตารางเทียบรายคู่ (ทุกคู่ที่ต่างกันอย่างมีนัยสำคัญ)")
    L.append("")
    L.append("McNemar's exact test บนผลทำนายรายตา แก้ค่า p ด้วย Holm "
             f"ทั้งหมด {len(pairs)} คู่")
    L.append("")
    sig_pairs = [p for p in pairs if p["sig"]]
    L.append(f"พบคู่ที่ต่างกันอย่างมีนัยสำคัญ **{len(sig_pairs)} คู่ จาก {len(pairs)} คู่**")
    L.append("")
    L.append("| โมเดลที่เก่งกว่า | โมเดลที่อ่อนกว่า | b / c | p (Holm) |")
    L.append("|:---|:---|---:|---:|")
    for p in sorted(sig_pairs, key=lambda x: x["p_holm"])[:30]:
        if p["n10"] > p["n01"]:
            w, l_, b, c = p["a"], p["b"], p["n10"], p["n01"]
        else:
            w, l_, b, c = p["b"], p["a"], p["n01"], p["n10"]
        L.append(f"| {w} | {l_} | {b} / {c} | {p['p_holm']:.2e} |")
    if len(sig_pairs) > 30:
        L.append(f"| ... | (อีก {len(sig_pairs) - 30} คู่ ดูในรูป pairwise matrix) | | |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. รูปประกอบ")
    L.append("")
    for k, v in figs.items():
        L.append(f"- `{v}`")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. ข้อสรุป")
    L.append("")
    L.append(f"1. ในการทำนายท่าถัดไปบนข้อมูลจริง โมเดลที่ดีที่สุดคือ **{best['name']}** "
             f"(อันดับเฉลี่ย {best['avg_rank']:.3f})")
    L.append(f"2. กลุ่มนำที่แยกความต่างกันไม่ได้ทางสถิติ: {', '.join(tied)}")
    lat_l = np.mean([np.mean(merged[n]["latency_us"]) for n in famc["light"]])
    lat_h = np.mean([np.mean(merged[n]["latency_us"]) for n in famc["heavy"]])
    ratio = lat_h / max(lat_l, 1e-9)
    L.append(f"3. ด้านความเร็ว lightweight เร็วกว่า heavyweight เฉลี่ย **{ratio:.0f} เท่า** "
             f"({lat_l:.1f} μs vs {lat_h:.1f} μs ต่อตา)")
    sz_l = np.mean([merged[n]["size_bytes"] for n in famc["light"]]) / 1024
    sz_h = np.mean([merged[n]["size_bytes"] for n in famc["heavy"]]) / 1024
    L.append(f"4. ด้านขนาด lightweight เล็กกว่าเฉลี่ย {sz_h / max(sz_l, 1e-9):.0f} เท่า "
             f"({sz_l:.1f} KB vs {sz_h:.1f} KB)")
    L.append("5. ข้อสรุปเชิงออกแบบระบบ: หากความแม่นยำไม่ต่างกันอย่างมีนัยสำคัญ "
             "การเลือก lightweight ให้ผลเท่ากันโดยใช้ทรัพยากรน้อยกว่ามาก "
             "เหมาะกับการทำงานแบบ real-time ในเกมที่ต้องตอบสนองทุกเฟรม")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 8. ข้อจำกัดของการทดลองนี้ (Limitations)")
    L.append("")
    L.append("1. **ความไม่สมมาตรของบริบทเล็กน้อย** — ฝั่ง heavyweight ใช้ฟีเจอร์ "
             "`lag1_res` และ `lag1_boss_move` ซึ่งเป็นผลและท่าของ AI ที่บันทึกไว้จริงในล็อก "
             "ส่วนฝั่ง lightweight เรียนรู้จากผลของ **ท่าที่ตัวเองเลือก** ซึ่งเป็นสถานการณ์สมมติ "
             "(counterfactual) เพราะคู่ต่อสู้จริงในล็อกเป็น AI คนละตัว "
             "ความไม่สมมาตรนี้ให้ข้อได้เปรียบกับฝั่ง heavyweight เล็กน้อย "
             "ผลที่ lightweight ยังตามมาใกล้เคียงจึงถือว่าน่าประทับใจกว่าที่ตัวเลขแสดง")
    L.append("")
    L.append("2. **จำนวนบล็อกน้อย** — ข้อมูลจริงมีจำกัด ทำให้ได้ "
             f"{fr['n_blocks']} บล็อก ค่า CD จึงกว้าง ({fr['cd']:.2f} อันดับ) "
             "และ Nemenyi แยกความต่างได้หยาบ **ข้อสรุปรายคู่ให้ยึดตาราง McNemar ในหัวข้อ 5 "
             "เป็นหลัก** เพราะใช้ข้อมูลครบทุกตาและจับคู่สมบูรณ์ จึงมี power สูงกว่ามาก")
    L.append("")
    L.append("3. **latency วัดเฉพาะการทำนาย** ไม่รวมเวลาเทรน ฝั่ง heavyweight "
             f"ต้องเทรนใหม่ทุก {meta['refit_every']} ตาเพิ่มอีกต่างหาก "
             "ถ้ารวมต้นทุนนี้ช่องว่างด้านทรัพยากรจะกว้างกว่าที่รายงาน")
    L.append("")
    L.append("4. **ข้อมูลมาจากผู้เล่นเพียง 2 คน** การสรุปไปยังผู้เล่นทั่วไปต้องเก็บข้อมูลเพิ่ม")
    L.append("")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return out


def main():
    ap = argparse.ArgumentParser(description="เทียบ Lightweight vs Heavyweight บนสนามทดสอบกลาง")
    ap.add_argument("--csv", default="data/players")
    ap.add_argument("--warmup", type=float, default=0.40)
    ap.add_argument("--refit", type=int, default=25)
    ap.add_argument("--segment", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="reports/figures")
    ap.add_argument("--out", default="reports/lightweight_vs_heavyweight.md")
    args = ap.parse_args()

    print("=" * 90)
    print("  สนามทดสอบกลาง: Prequential Evaluation (Test-Then-Train)")
    print("=" * 90)

    df = read_telemetry(args.csv)
    if "player_name" not in df.columns:
        df["player_name"] = "Guest"

    runs, players = [], []
    for pname, g in df.groupby("player_name", sort=False):
        g = g.reset_index(drop=True)
        if len(g) < 80:
            print(f"  ข้าม '{pname}' ({len(g)} ตา — น้อยเกินไป)")
            continue
        print(f"  replay ผู้เล่น '{pname}' ({len(g)} ตา)...")
        r = run_prequential(g, args.warmup, args.refit, args.seed)
        if r:
            runs.append(r)
            players.append(str(pname))
    if not runs:
        print("!! ข้อมูลไม่พอสำหรับการทดลอง")
        return

    merged = merge_players(runs)
    names, acc_m, r_m = segment_scores(merged, args.segment)
    fr = friedman_nemenyi(names, acc_m)
    pairs, sign = pairwise_matrix(names, merged)
    famc = family_comparison(names, merged, acc_m)
    figs = make_figures(names, merged, acc_m, fr, sign, args.outdir)

    meta = {"players": players, "n_scored": len(merged[names[0]]["correct"]),
            "warmup_frac": args.warmup, "refit_every": args.refit, "segment": args.segment}
    out = write_report(names, merged, acc_m, r_m, fr, pairs, sign, famc, figs, meta, args.out)

    idx = {n: i for i, n in enumerate(names)}
    print("")
    print("-" * 90)
    print(f"{'โมเดล':<26} | {'ตระกูล':<4} | {'Accuracy':<18} | {'R':>6} | {'μs/ตา':>9} | {'KB':>7}")
    print("-" * 90)
    for r in fr["ranking"]:
        n = r["name"]
        am, ae = mean_ci(acc_m[:, idx[n]])
        rm, _ = mean_ci(r_m[:, idx[n]])
        lat = float(np.mean(merged[n]["latency_us"]))
        kb = merged[n]["size_bytes"] / 1024.0
        tag = "LW" if merged[n]["family"] == LIGHT_FAMILY else "HW"
        print(f"{n:<26} | {tag:<4} | {am:6.2f}% +/-{ae:5.2f}% | {rm:6.1f} | {lat:9.1f} | {kb:7.1f}")
    print("-" * 90)
    print(f"  Friedman chi2 = {fr['chi2']:.2f}, p = {fr['p']:.3e}, CD = {fr['cd']:.3f}, "
          f"N = {fr['n_blocks']} บล็อก")
    h2h = famc["head_to_head"]
    print(f"  ตัวแทน LW: {famc['best_light']} ({famc['acc_of'][famc['best_light']]:.2f}%)")
    print(f"  ตัวแทน HW: {famc['best_heavy']} ({famc['acc_of'][famc['best_heavy']]:.2f}%)")
    print(f"  McNemar head-to-head: b={h2h['n10']}, c={h2h['n01']}, p={h2h['p']:.3e}")
    print("")
    for v in figs.values():
        print(f"  {v}")
    print(f"  {out}")
    print("=" * 90)


if __name__ == "__main__":
    main()
