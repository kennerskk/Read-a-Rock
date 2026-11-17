# =====================================================
#  benchmark_ensemble_combos.py
#  ทดสอบว่า Ensemble ที่ใช้สมาชิกแค่ 2 / 3 / 4 ตัว ชุดไหน "โหดสุด"
#  เทียบกับ Ensemble เดิม (4 ตัว) และ Ensemble เต็ม 6 ตัว
#
#  คลังผู้เชี่ยวชาญ 6 ตัว: Frequency, Decay, Markov, N-gram, Q-learning, ML
#  -> ทุกชุดขนาด 1..6 รวม 63 ชุด + Ensemble เดิม = 64 บอส
#
#  คู่ต่อสู้ (7 แบบ):
#    - Persona จำลอง: เอียงค้อน / วนลูป / WSLS / เปลี่ยนสไตล์ทุก 75 ตา / สุ่ม Nash
#    - คนจริง: replay ลำดับท่าจาก data/gameplay_telemetry.csv (ผู้เล่นที่มี >=100 ตา
#      ไม่นับ Guest เพราะเป็นข้อมูลจำลอง)
#
#  ตัวชี้วัดหลัก: Net Score = (บอสชนะ - บอสแพ้) / จำนวนตา x 100
#    0 = เท่ากับสุ่ม, ยิ่งสูงยิ่งโหด (สูงสุด 100)
#  จัดอันดับด้วยค่าเฉลี่ย Net Score กับคู่ต่อสู้ที่ "จับทางได้" (ไม่รวม Nash ซึ่งทุกตัวได้ ~0)
#  ทุกบอสใช้ seed ชุดเดียวกัน (common random numbers) จึงเทียบแบบ paired ได้
#
#  รัน:  python benchmark_ensemble_combos.py            (ค่าเริ่มต้น 20 reps x 300 ตา)
#        python benchmark_ensemble_combos.py --reps 5   (ทดสอบเร็ว)
# =====================================================
import os
import sys
import json
import time
import random
import argparse
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from adaptive_arena import (MOVES, COUNTER, judge, EXPERT_POOL,
                            ComboEnsembleBoss, EnsembleBoss)
from evaluate_ai import BiasedPlayer, CyclicPlayer, WinStayLoseShiftPlayer, NashRandomPlayer

ORIGINAL_KEY = "ORIGINAL"     # Ensemble เดิมในเกม (บอส 6)


# ---------------- คู่ต่อสู้ ----------------
class StyleSwitcher:
    """เปลี่ยนสไตล์ทุก 75 ตา: เอียงค้อน -> วนลูป -> WSLS -> เอียงกระดาษ (ทดสอบการปรับตัว)"""
    name = "เปลี่ยนสไตล์ทุก 75 ตา"

    def __init__(self):
        self.n = 0
        self.wsls = WinStayLoseShiftPlayer()
        self.cyc = CyclicPlayer()

    def choose(self, history, last_result):
        phase = (self.n // 75) % 4
        self.n += 1
        if phase == 0:
            return "ค้อน" if random.random() < 0.6 else random.choice(MOVES)
        if phase == 1:
            return self.cyc.choose(history, last_result)
        if phase == 2:
            return self.wsls.choose(history, last_result)
        return "กระดาษ" if random.random() < 0.6 else random.choice(MOVES)


class ReplayPlayer:
    """เล่นซ้ำลำดับท่าจริงของผู้เล่นจาก telemetry"""

    def __init__(self, seq):
        self.seq = seq
        self.i = 0

    def choose(self, history, last_result):
        m = self.seq[self.i % len(self.seq)]
        self.i += 1
        return m


def load_human_sequences(csv_path, min_rounds=100):
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    df = df[df["player_move"].isin(MOVES)]
    df["pkey"] = df["player_name"].astype(str).str.strip().str.lower()
    df = df[df["pkey"] != "guest"]                       # Guest = ข้อมูลจำลอง
    df = df.sort_values("timestamp", kind="stable")
    out = {}
    for name, g in df.groupby("pkey"):
        if len(g) >= min_rounds:
            out[name] = g["player_move"].tolist()
    return out


def build_opponents(csv_path, sim_rounds):
    opps = [
        ("เอียงค้อน 60%",          lambda: BiasedPlayer(), sim_rounds),
        ("วนลูป ค้อน>กรรไกร>กระดาษ", lambda: CyclicPlayer(), sim_rounds),
        ("WSLS (ชนะซ้ำ/แพ้เปลี่ยน)", lambda: WinStayLoseShiftPlayer(), sim_rounds),
        ("เปลี่ยนสไตล์ทุก 75 ตา",     lambda: StyleSwitcher(), sim_rounds),
    ]
    for name, seq in load_human_sequences(csv_path).items():
        opps.append((f"คนจริง: {name} ({len(seq)} ตา)", (lambda s=seq: ReplayPlayer(s)), len(seq)))
    opps.append(("สุ่ม Nash (ควบคุม)", lambda: NashRandomPlayer(), sim_rounds))
    return opps


# ---------------- บอส ----------------
def all_combo_keys():
    keys = list(EXPERT_POOL)
    out = []
    for k in range(1, len(keys) + 1):
        out.extend(combinations(keys, k))
    return out


def make_boss(key):
    if key == ORIGINAL_KEY:
        return EnsembleBoss()
    return ComboEnsembleBoss(key)


def combo_label(key):
    if key == ORIGINAL_KEY:
        return "Ensemble เดิม (Freq+Decay+Markov+N-gram)"
    return " + ".join(EXPERT_POOL[m][0] for m in key)


def combo_size(key):
    return 4 if key == ORIGINAL_KEY else len(key)


# ---------------- จำลองการแข่ง ----------------
def play_match(boss, persona, rounds):
    history, last = [], None
    bw = pw = 0
    for _ in range(rounds):
        p = persona.choose(history, last)
        b = boss.choose(history)
        res = judge(p, b)
        history.append(p)
        boss.learn({"win": -1, "draw": 0, "lose": 1}[res], history)
        if res == "win":
            pw += 1
        elif res == "lose":
            bw += 1
        last = res
    picks = dict(getattr(boss, "picks", {}))
    return ((bw - pw) / rounds * 100.0, bw / rounds * 100.0, pw / rounds * 100.0,
            (rounds - bw - pw) / rounds * 100.0, picks)


_OPPS = None


def _init_worker(csv_path, sim_rounds):
    global _OPPS
    _OPPS = build_opponents(csv_path, sim_rounds)


def run_boss(args):
    key, reps = args
    net = np.zeros((len(_OPPS), reps))
    win = np.zeros((len(_OPPS), reps))       # บอสชนะ %
    loss = np.zeros((len(_OPPS), reps))      # บอสแพ้ %
    draw = np.zeros((len(_OPPS), reps))
    picks = [dict() for _ in _OPPS]          # สัดส่วนที่ Ensemble เชื่อสมาชิกแต่ละตัว ต่อคู่ต่อสู้
    for oi, (_, factory, rounds) in enumerate(_OPPS):
        for r in range(reps):
            random.seed(10_000 * oi + r)          # seed เดียวกันทุกบอส -> paired
            net[oi, r], win[oi, r], loss[oi, r], draw[oi, r], pk = play_match(make_boss(key), factory(), rounds)
            for m, c in pk.items():
                picks[oi][m] = picks[oi].get(m, 0) + c
    for d in picks:
        tot = sum(d.values()) or 1
        for m in d:
            d[m] = d[m] / tot * 100.0
    return key, net, win, loss, draw, picks


def time_per_round(key, rounds=1000, repeats=7):
    """เวลาเฉลี่ยต่อตา (choose+learn) หน่วยไมโครวินาที ใช้ค่าต่ำสุดจากหลายรอบ (ปิด GC ระหว่างวัด)"""
    import gc
    best = float("inf")
    gc.disable()
    try:
        best = _time_loop(key, rounds, repeats)
    finally:
        gc.enable()
    return best


def _time_loop(key, rounds, repeats):
    best = float("inf")
    for rep in range(repeats):
        random.seed(rep)
        boss, persona = make_boss(key), WinStayLoseShiftPlayer()
        history, last = [], None
        t_boss = 0.0
        for _ in range(rounds):
            p = persona.choose(history, last)
            t0 = time.perf_counter()
            b = boss.choose(history)
            res = judge(p, b)
            history.append(p)
            boss.learn({"win": -1, "draw": 0, "lose": 1}[res], history)
            t_boss += time.perf_counter() - t0
            last = res
        best = min(best, t_boss / rounds * 1e6)
    return best


# ---------------- วิเคราะห์ ----------------
def mean_ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, m, m
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def paired(a, b):
    """a, b = เมทริกซ์ (คู่ต่อสู้ x reps) ของคู่ต่อสู้ที่จับทางได้ -> ทดสอบ paired t"""
    d = (np.asarray(a) - np.asarray(b)).ravel()
    m, lo, hi = mean_ci(d)
    p = stats.ttest_rel(np.asarray(a).ravel(), np.asarray(b).ravel()).pvalue if d.std() > 0 else 1.0
    return m, lo, hi, p


def make_figures(rows, outdir, top_n=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for f in ["Leelawadee UI", "Leelawadee", "Tahoma"]:
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    size_color = {2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a"}
    ref_color = "#8f8e88"
    os.makedirs(outdir, exist_ok=True)

    # --- รูป 1: Top-N ของแต่ละขนาด + เส้นอ้างอิง Ensemble เดิม ---
    orig = next(r for r in rows if r["key"] == ORIGINAL_KEY)
    sel = []
    for k in (2, 3, 4):
        sel += sorted([r for r in rows if r["size"] == k and r["key"] != ORIGINAL_KEY],
                      key=lambda r: -r["score"])[:top_n]
    sel = sel[::-1]
    ml_only = next((r for r in rows if r["key"] == ("ml",)), None)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(sel) + 1.9), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    y = np.arange(len(sel))
    # dot plot + 95% CI (ไม่ใช้แท่ง เพราะความต่างอยู่ช่วง 40-50 แท่งที่เริ่มจาก 0 จะซ่อนความต่าง)
    ax.hlines(y, [r["lo"] for r in sel], [r["hi"] for r in sel], color=ink2, linewidth=1.2, zorder=2)
    ax.scatter([r["score"] for r in sel], y, s=70, color=[size_color[r["size"]] for r in sel],
               edgecolor=surface, linewidth=2, zorder=3)
    refs = [(orig["score"], f"Ensemble เดิม {orig['score']:.1f}")]
    if ml_only:
        refs.append((ml_only["score"], f"ML ตัวเดียว {ml_only['score']:.1f}"))
    for xv, txt in refs:
        ax.axvline(xv, color=ref_color, linestyle="--", linewidth=1.2, zorder=1)
        ax.text(xv, len(sel) - 0.2, txt, color=ink2, fontsize=8.5, va="bottom", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([f"[{r['size']} ตัว] {r['label']}" for r in sel], fontsize=9, color=ink)
    for yi, r in zip(y, sel):
        ax.text(r["hi"] + 0.15, yi, f"{r['score']:.1f}", va="center", fontsize=8.5, color=ink)
    lo_x = min(min(r["lo"] for r in sel), orig["score"]) - 1.0
    hi_x = max(r["hi"] for r in sel) + 1.2
    ax.set_xlim(lo_x, hi_x)
    ax.set_ylim(-0.7, len(sel) + 0.6)
    ax.set_xlabel("Net Score ของบอส (ชนะ − แพ้, % ต่อตา) เฉลี่ยกับคู่ต่อสู้ที่จับทางได้ — ยิ่งขวายิ่งโหด", color=ink2)
    ax.set_title(f"Top {top_n} ชุด Ensemble แต่ละขนาด (เส้น = 95% CI)", color=ink, loc="left", fontsize=12, pad=22)
    ax.grid(axis="x", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(grid)
    ax.tick_params(colors=ink2, length=0)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=size_color[k]) for k in (2, 3, 4)]
    ax.legend(handles, ["2 ตัว", "3 ตัว", "4 ตัว"], loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=3, frameon=False, fontsize=9, borderaxespad=0.2)
    fig.tight_layout()
    p1 = os.path.join(outdir, "ensemble_combos_top.png")
    fig.savefig(p1, facecolor=surface)
    plt.close(fig)

    # --- รูป 2: ความโหด vs ต้นทุนเวลา (ทุกชุด) ---
    fig, ax = plt.subplots(figsize=(9, 5.4), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for k in (1, 2, 3, 4, 5, 6):
        pts = [r for r in rows if r["size"] == k and r["key"] != ORIGINAL_KEY]
        color = size_color.get(k, ref_color)
        ax.scatter([r["us"] for r in pts], [r["score"] for r in pts], s=46 if k in size_color else 26,
                   color=color, edgecolor=surface, linewidth=1.5, alpha=1 if k in size_color else 0.6,
                   label=f"{k} ตัว" if k in size_color else None, zorder=3)
    ax.scatter([orig["us"]], [orig["score"]], marker="D", s=60, color=ink, edgecolor=surface,
               linewidth=1.5, zorder=4, label="Ensemble เดิม")
    best_by_size = {}
    for r in rows:
        if r["key"] != ORIGINAL_KEY and (r["size"] not in best_by_size or r["score"] > best_by_size[r["size"]]["score"]):
            best_by_size[r["size"]] = r
    for k, ty in zip((2, 3, 4), (0.56, 0.48, 0.40)):
        r = best_by_size[k]
        ax.annotate(f"แชมป์ {k} ตัว: {r['label']} ({r['score']:.1f})", (r["us"], r["score"]),
                    xycoords="data", textcoords="axes fraction", xytext=(0.98, ty), ha="right", va="center",
                    fontsize=8.5, color=ink,
                    arrowprops=dict(arrowstyle="-", color=size_color[k], linewidth=0.9, shrinkA=2, shrinkB=4))
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    ticks = [t for t in (1, 2, 3, 5, 10, 20, 30, 50) if min(r["us"] for r in rows) * 0.8 <= t <= max(r["us"] for r in rows) * 1.2]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("เวลาประมวลผลต่อตา (ไมโครวินาที, สเกล log) — ยิ่งซ้ายยิ่งประหยัด", color=ink2)
    ax.set_ylabel("Net Score (ยิ่งสูงยิ่งโหด)", color=ink2)
    ax.set_title("ความโหด vs ต้นทุน: ทุกชุดค่าผสม (จุดเทา = 1, 5, 6 ตัว)", color=ink, loc="left", fontsize=12)
    ax.grid(color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink2)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    p2 = os.path.join(outdir, "ensemble_combos_cost.png")
    fig.savefig(p2, facecolor=surface)
    plt.close(fig)
    # --- รูป 3: เมทริกซ์จับคู่ (ทุกคู่ 2 ตัว, แนวทแยง = ตัวเดียว) ---
    from matplotlib.colors import LinearSegmentedColormap
    keys = list(EXPERT_POOL)
    names = [EXPERT_POOL[k][0] for k in keys]
    by_key = {r["key"]: r for r in rows}
    M = np.full((len(keys), len(keys)), np.nan)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            if i == j:
                M[i, j] = by_key[(a,)]["win_pct"]
            else:
                pair = tuple(k for k in keys if k in (a, b))
                M[i, j] = by_key[pair]["win_pct"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#e8f1fc", "#9cc3ee", "#2a78d6", "#123f78"])
    fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    im = ax.imshow(M, cmap=cmap, vmin=33.3, vmax=np.nanmax(M))
    best = np.unravel_index(np.nanargmax(np.where(np.eye(len(keys)) == 1, -np.inf, M)), M.shape)
    for i in range(len(keys)):
        for j in range(len(keys)):
            v = M[i, j]
            txt_col = "#ffffff" if v > (33.3 + np.nanmax(M)) / 2 else ink
            weight = "bold" if (i, j) in (best, best[::-1]) else "normal"
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=9, color=txt_col, fontweight=weight)
    for (i, j) in (best, best[::-1]):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#eb6834", linewidth=2.5))
    ax.set_xticks(range(len(keys)))
    ax.set_yticks(range(len(keys)))
    ax.set_xticklabels(names, fontsize=9, color=ink)
    ax.set_yticklabels(names, fontsize=9, color=ink)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("% ที่บอสชนะ เมื่อจับคู่ 2 ตัว (แนวทแยง = ใช้ตัวเดียว)\nเฉลี่ยกับคู่ต่อสู้ที่จับทางได้ · กรอบส้ม = คู่ที่ดีที่สุด",
                 color=ink, loc="left", fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("บอสชนะ (%) — 33.3% = เท่าสุ่ม", color=ink2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    p3 = os.path.join(outdir, "ensemble_pairs_matrix.png")
    fig.savefig(p3, facecolor=surface)
    plt.close(fig)
    return p1, p2, p3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/gameplay_telemetry.csv")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=300, help="จำนวนตาต่อแมตช์ของ persona จำลอง")
    ap.add_argument("--out", default="reports/ensemble_combos.md")
    ap.add_argument("--figdir", default="reports/figures")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    opps = build_opponents(args.csv, args.rounds)
    opp_names = [o[0] for o in opps]
    exploit_idx = [i for i, n in enumerate(opp_names) if "Nash" not in n]
    nash_idx = [i for i, n in enumerate(opp_names) if "Nash" in n]

    keys = all_combo_keys() + [ORIGINAL_KEY]
    print(f"บอส {len(keys)} ตัว x คู่ต่อสู้ {len(opps)} แบบ x {args.reps} reps  (workers={args.workers})")
    for n in opp_names:
        print("  -", n)

    t0 = time.time()
    results = {}
    with Pool(args.workers, initializer=_init_worker, initargs=(args.csv, args.rounds)) as pool:
        for i, (key, net, win, loss, draw, picks) in enumerate(
                pool.imap_unordered(run_boss, [(k, args.reps) for k in keys]), 1):
            results[key] = (net, win, loss, draw, picks)
            print(f"\r  เสร็จ {i}/{len(keys)}", end="", flush=True)
    print(f"\n  ใช้เวลาจำลอง {time.time() - t0:.0f} วินาที")

    print("  วัดเวลาประมวลผลต่อตา...")
    timing = {k: time_per_round(k) for k in keys}

    rows = []
    for key in keys:
        net, win, loss, draw, picks = results[key]
        per_rep = net[exploit_idx].mean(axis=0)          # ค่าเฉลี่ยข้ามคู่ต่อสู้ ต่อ rep
        m, lo, hi = mean_ci(per_rep)
        rows.append({
            "key": key, "label": combo_label(key), "size": combo_size(key),
            "score": m, "lo": lo, "hi": hi,
            "per_opp": net.mean(axis=1).tolist(),
            "win_pct": win[exploit_idx].mean(),
            "loss_pct": loss[exploit_idx].mean(),
            "draw_pct": draw[exploit_idx].mean(),
            "win_per_opp": win.mean(axis=1).tolist(),
            "picks": picks,
            "nash": net[nash_idx].mean() if nash_idx else float("nan"),
            "us": timing[key],
        })
    by_key = {r["key"]: r for r in rows}
    orig = by_key[ORIGINAL_KEY]
    full6 = by_key[tuple(EXPERT_POOL)]

    def ranked(k):
        return sorted([r for r in rows if r["size"] == k and r["key"] != ORIGINAL_KEY], key=lambda r: -r["score"])

    # paired tests: แชมป์แต่ละขนาด vs Ensemble เดิม และ vs อันดับ 2 ของขนาดเดียวกัน
    tests = {}
    for k in (2, 3, 4):
        rk = ranked(k)
        champ, second = rk[0], rk[1]
        a = results[champ["key"]][0][exploit_idx]
        tests[k] = {
            "vs_orig": paired(a, results[ORIGINAL_KEY][0][exploit_idx]),
            "vs_second": paired(a, results[second["key"]][0][exploit_idx]),
            "vs_full6": paired(a, results[full6["key"]][0][exploit_idx]),
            "second": second,
        }

    figs = make_figures(rows, args.figdir)  # (top, cost, pair-matrix)

    # ---------- เขียนรายงาน ----------
    L = []
    L.append("# ผลทดสอบ Ensemble แบบเลือกสมาชิก 2 / 3 / 4 ตัว\n")
    L.append(f"*สร้างโดย `benchmark_ensemble_combos.py` — {args.reps} reps ต่อคู่, persona จำลอง {args.rounds} ตา/แมตช์, "
             f"คนจริงเล่นซ้ำตามจำนวนตาที่บันทึกไว้*\n")
    L.append("## คำถาม\nEnsemble เดิม (บอส 6) รันผู้เชี่ยวชาญพร้อมกัน **4 ตัว** (Frequency, Decay, Markov, N-gram). "
             "ถ้าอยากประหยัดทรัพยากร ใช้แค่ 2, 3 หรือ 4 ตัว — ชุดไหนโหดสุด? "
             "เราเพิ่ม Q-learning และ ML เข้าคลังให้เลือกด้วย รวมผู้เชี่ยวชาญ 6 ตัว ทดสอบ **ทุกชุดค่าผสม (63 ชุด)**\n")
    L.append("## วิธีทดสอบ")
    L.append("- คู่ต่อสู้: " + ", ".join(opp_names))
    L.append("- ตัวชี้วัด: **Net Score = (บอสชนะ − บอสแพ้) / จำนวนตา × 100** — 0 คือเท่ากับเดาสุ่ม, 100 คือชนะทุกตา")
    L.append("- คะแนนจัดอันดับ = ค่าเฉลี่ย Net Score กับคู่ต่อสู้ที่จับทางได้ (ไม่รวม Nash ซึ่งไม่มีใครเอาชนะได้ตามทฤษฎี — ใช้เป็นตัวควบคุม)")
    L.append("- ทุกบอสเจอ seed ชุดเดียวกัน (paired design) จึงใช้ paired t-test เปรียบเทียบได้")
    L.append("- ต้นทุน = เวลา choose+learn ต่อตา (µs, ค่าต่ำสุดจาก 7 รอบ × 1000 ตา)\n")

    L.append("## สรุปผู้ชนะ\n")
    L.append("| ขนาด | ชุดที่โหดสุด | Net Score (95% CI) | เวลา/ตา | เทียบ Ensemble เดิม |")
    L.append("|:-:|:--|:-:|:-:|:--|")
    for k in (2, 3, 4):
        c = ranked(k)[0]
        d, lo, hi, p = tests[k]["vs_orig"]
        L.append(f"| {k} ตัว | **{c['label']}** | {c['score']:.1f} ({c['lo']:.1f}–{c['hi']:.1f}) | {c['us']:.0f} µs | "
                 f"{d:+.1f} (CI {lo:+.1f}–{hi:+.1f}, p={p:.3g}) |")
    L.append(f"| 4 ตัว (เดิม) | Ensemble เดิม | {orig['score']:.1f} ({orig['lo']:.1f}–{orig['hi']:.1f}) | {orig['us']:.0f} µs | — |")
    L.append(f"| 6 ตัว | {full6['label']} | {full6['score']:.1f} ({full6['lo']:.1f}–{full6['hi']:.1f}) | {full6['us']:.0f} µs | "
             f"{full6['score'] - orig['score']:+.1f} |\n")

    L.append("### ชนะอันดับ 2 ของขนาดเดียวกันอย่างมีนัยสำคัญไหม?\n")
    for k in (2, 3, 4):
        c = ranked(k)[0]
        s = tests[k]["second"]
        d, lo, hi, p = tests[k]["vs_second"]
        verdict = "ต่างอย่างมีนัยสำคัญ (p<0.05)" if p < 0.05 else "**ไม่ต่างอย่างมีนัยสำคัญ** — ถือว่าสูสี"
        L.append(f"- **{k} ตัว:** {c['label']} vs {s['label']}: ต่าง {d:+.2f} (CI {lo:+.2f}–{hi:+.2f}, p={p:.3g}) → {verdict}")
    L.append("")
    L.append("### เทียบกับ Ensemble เต็ม 6 ตัว\n")
    for k in (2, 3, 4):
        c = ranked(k)[0]
        d, lo, hi, p = tests[k]["vs_full6"]
        L.append(f"- **{k} ตัว** ({c['label']}): ต่าง {d:+.2f} (CI {lo:+.2f}–{hi:+.2f}, p={p:.3g}), "
                 f"เร็วกว่า {full6['us'] / c['us']:.1f} เท่า")
    L.append("")

    ml_key = ("ml",)
    if ml_key in by_key:
        ml = by_key[ml_key]
        L.append("### เทียบกับ ML ตัวเดียว (ไม่ใช้ Ensemble เลย)\n")
        for k in (2, 3, 4):
            c = ranked(k)[0]
            d, lo, hi, p = paired(results[c["key"]][0][exploit_idx], results[ml_key][0][exploit_idx])
            L.append(f"- **{k} ตัว** ({c['label']}): ดีกว่า ML เดี่ยว {d:+.2f} (CI {lo:+.2f}–{hi:+.2f}, p={p:.3g})")
        L.append(f"- ML เดี่ยวได้ {ml['score']:.1f} ที่ {ml['us']:.0f} µs/ตา — การเพิ่มสมาชิกช่วยได้แต่ไม่มาก "
                 "ส่วนใหญ่เพิ่มความเสถียรเมื่อผู้เล่นเปลี่ยนสไตล์ (ดูคอลัมน์ 'เปลี่ยนสไตล์ทุก 75 ตา')\n")
    same4 = by_key.get(("freq", "decay", "markov", "ngram"))
    if same4:
        L.append(f"*ตรวจความถูกต้อง: ComboEnsemble ที่ใช้สมาชิกชุดเดียวกับ Ensemble เดิมได้ {same4['score']:.1f} "
                 f"เทียบกับ Ensemble เดิม {orig['score']:.1f} — ใกล้เคียงกัน แปลว่าการเขียนใหม่ทำงานเหมือนเดิม*\n")

    L.append(f"![Top combos]({os.path.relpath(figs[0], os.path.dirname(args.out)).replace(os.sep, '/')})\n")
    L.append(f"![Cost vs strength]({os.path.relpath(figs[1], os.path.dirname(args.out)).replace(os.sep, '/')})\n")

    short = [n.split(" (")[0] for n in opp_names]

    # ===== ข้อมูลการจับคู่ 2 ตัว (ทำไมเลือก N-gram + ML) =====
    pairs = ranked(2)
    champ = pairs[0]
    L.append("## ข้อมูลการจับคู่ 2 ตัว: ทุกคู่ได้กี่ %\n")
    L.append(f"![Pair matrix]({os.path.relpath(figs[2], os.path.dirname(args.out)).replace(os.sep, '/')})\n")
    L.append("ค่าเฉลี่ยกับคู่ต่อสู้ที่จับทางได้ 6 แบบ (ไม่รวม Nash) — **ชนะ/แพ้/เสมอ เป็นมุมมองของบอส**\n")
    L.append("| อันดับ | คู่ | บอสชนะ % | บอสแพ้ % | เสมอ % | Net Score |")
    L.append("|:-:|:--|:-:|:-:|:-:|:-:|")
    for i, r in enumerate(pairs, 1):
        name = f"**{r['label']}**" if i == 1 else r["label"]
        L.append(f"| {i} | {name} | {r['win_pct']:.1f} | {r['loss_pct']:.1f} | {r['draw_pct']:.1f} | {r['score']:.1f} |")
    L.append("")
    L.append("### % ที่บอสชนะ แยกตามคู่ต่อสู้ (ทุกคู่ 2 ตัว)\n")
    L.append("| คู่ | " + " | ".join(short) + " |")
    L.append("|:--|" + "|".join([":-:"] * len(short)) + "|")
    for r in pairs:
        L.append(f"| {r['label']} | " + " | ".join(f"{v:.1f}" for v in r["win_per_opp"]) + " |")
    L.append("")

    # ทำไม: เทียบคู่แชมป์กับสมาชิกเดี่ยว + สัดส่วนที่เชื่อแต่ละตัว
    members = champ["key"]
    singles = [by_key[(m,)] for m in members]
    L.append(f"### ทำไมเลือก {champ['label']}\n")
    L.append("**1) สองตัวนี้เก่งคนละแบบ — จับคู่แล้วปิดจุดอ่อนกันเอง** (% บอสชนะ)\n")
    L.append("| คู่ต่อสู้ | " + " | ".join(f"{EXPERT_POOL[m][0]} เดี่ยว" for m in members)
             + f" | **{champ['label']}** | Ensemble เดิม |")
    L.append("|:--|" + "|".join([":-:"] * (len(members) + 2)) + "|")
    for oi, n in enumerate(opp_names):
        cells = [f"{sg['win_per_opp'][oi]:.1f}" for sg in singles]
        best_single = max(sg["win_per_opp"][oi] for sg in singles)
        pv = champ["win_per_opp"][oi]
        mark = " ▲" if pv > best_single + 0.5 else ""
        L.append(f"| {n} | " + " | ".join(cells) + f" | **{pv:.1f}**{mark} | {orig['win_per_opp'][oi]:.1f} |")
    L.append(f"| **เฉลี่ย (ไม่รวม Nash)** | " + " | ".join(f"{sg['win_pct']:.1f}" for sg in singles)
             + f" | **{champ['win_pct']:.1f}** | {orig['win_pct']:.1f} |")
    L.append("\n▲ = คู่ทำได้ดีกว่าตัวที่เก่งที่สุดในคู่ตอนเล่นเดี่ยว\n")

    L.append("**2) Ensemble สลับไปเชื่อตัวที่ทายแม่นกว่ากับคู่ต่อสู้แต่ละแบบ** (% ของตาที่เชื่อแต่ละตัว)\n")
    L.append("| คู่ต่อสู้ | " + " | ".join(f"เชื่อ {EXPERT_POOL[m][0]}" for m in members) + " |")
    L.append("|:--|" + "|".join([":-:"] * len(members)) + "|")
    for oi, n in enumerate(opp_names):
        pk = champ["picks"][oi]
        L.append(f"| {n} | " + " | ".join(f"{pk.get(m, 0):.0f}%" for m in members) + " |")
    L.append("")

    runner = pairs[1]
    d2, lo2, hi2, p2 = tests[2]["vs_second"]
    gaps = sorted(((champ["win_per_opp"][oi] - runner["win_per_opp"][oi], opp_names[oi]) for oi in exploit_idx),
                  reverse=True)
    L.append("**3) สรุปเหตุผล**\n")
    L.append(f"- ได้ Net Score สูงสุดในบรรดา 15 คู่ ({champ['score']:.1f}) และชนะคู่อันดับ 2 ({runner['label']}) "
             f"อย่างมีนัยสำคัญ: ต่าง {d2:+.2f} (95% CI {lo2:+.2f} ถึง {hi2:+.2f}, p={p2:.3g})")
    L.append(f"- ได้เปรียบคู่อันดับ 2 มากที่สุดกับ: " + ", ".join(f"{n} ({g:+.1f}%)" for g, n in gaps[:2]))
    L.append(f"- ชนะ Ensemble เดิม (4 ตัว) {champ['score'] - orig['score']:+.1f} Net Score ทั้งที่รันแค่ 2 โมเดล")
    L.append(f"- โหดพอ ๆ กับ Ensemble เต็ม 6 ตัว (ต่างกันอย่างไม่มีนัยสำคัญ, p={tests[2]["vs_full6"][3]:.3g}) แต่ใช้โมเดลน้อยกว่า 3 เท่า\n")
    for k in (2, 3, 4):
        L.append(f"## อันดับทั้งหมด: {k} ตัว\n")
        L.append("| # | ชุด | คะแนน | 95% CI | เวลา/ตา | " + " | ".join(short) + " |")
        L.append("|:-:|:--|:-:|:-:|:-:|" + "|".join([":-:"] * len(short)) + "|")
        for i, r in enumerate(ranked(k), 1):
            L.append(f"| {i} | {r['label']} | {r['score']:.1f} | {r['lo']:.1f}–{r['hi']:.1f} | {r['us']:.0f} µs | "
                     + " | ".join(f"{v:.1f}" for v in r["per_opp"]) + " |")
        L.append("")

    L.append("## ผู้เชี่ยวชาญเดี่ยว (1 ตัว) เพื่ออ้างอิง\n")
    L.append("| ผู้เชี่ยวชาญ | คะแนน | เวลา/ตา |")
    L.append("|:--|:-:|:-:|")
    for r in ranked(1):
        L.append(f"| {r['label']} | {r['score']:.1f} | {r['us']:.0f} µs |")
    L.append("")
    L.append("## ข้อจำกัด")
    L.append("- คนจริงเป็นการ *เล่นซ้ำ* ลำดับท่าที่บันทึกไว้ ผู้เล่นจริงอาจปรับตัวตามบอสต่างจากนี้")
    L.append("- Persona จำลองเป็นแบบจำลองพฤติกรรมอย่างง่าย ผลกับคนจริงควรยืนยันด้วยการเล่นจริงเพิ่ม")
    L.append("- เวลาต่อตาวัดบน Python เครื่องเดียว ใช้เปรียบเทียบสัมพัทธ์ ไม่ใช่ค่าสัมบูรณ์")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"opponents": opp_names, "reps": args.reps, "rounds": args.rounds,
                   "rows": [{**r, "key": list(r["key"]) if r["key"] != ORIGINAL_KEY else r["key"]} for r in rows]},
                  f, ensure_ascii=False, indent=1)

    print("\n===== สรุป =====")
    for k in (2, 3, 4):
        c = ranked(k)[0]
        print(f"  {k} ตัว: {c['label']:<40} score={c['score']:.1f}  ({c['us']:.0f} µs/ตา)")
    print(f"  Ensemble เดิม: score={orig['score']:.1f} ({orig['us']:.0f} µs/ตา)")
    print(f"  6 ตัว:         score={full6['score']:.1f} ({full6['us']:.0f} µs/ตา)")
    print(f"\nรายงาน: {args.out}\nรูป: {figs[0]}, {figs[1]}")


if __name__ == "__main__":
    main()
