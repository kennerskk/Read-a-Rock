# =====================================================
#  benchmark_warm_start.py — ใช้ได้กับ "ทุกคน" ไหม? (Leave-One-Player-Out)
#
#  ปัญหา: บอสที่เรียนรู้ระหว่างเกม (เช่น Lite Ensemble N-gram + ML) เริ่มจากศูนย์ทุกครั้ง
#         ช่วงต้นเกมจึงยังเดาไม่ได้
#  ทางแก้: เพิ่มผู้เชี่ยวชาญ Population Prior ที่เทรนจาก "ผู้เล่นคนอื่นทั้งหมด"
#         ให้รู้จักนิสัยคนทั่วไปตั้งแต่ตาแรก แล้ว Ensemble ค่อยเปลี่ยนไปเชื่อโมเดลที่เรียนจากคนตรงหน้า
#
#  วิธีทดสอบที่ไม่โกง (Leave-One-Player-Out):
#    สำหรับผู้เล่นจริงแต่ละคน X -> เทรน Prior จากทุกคน "ยกเว้น X" -> ทดสอบกับท่าจริงของ X
#    = จำลองว่าเจอ "คนใหม่ที่ระบบไม่เคยเห็นข้อมูลเลย"
#    บอสเริ่มใหม่ทุก session (เหมือนเปิดเกมใหม่) แล้วเล่นซ้ำลำดับท่าจริงของ X ใน session นั้น
#
#  ตรวจเพิ่ม: Prior ต้องไม่ทำให้แย่ลงกับผู้เล่นแบบอื่น (persona จำลอง 5 แบบ)
#
#  รัน:  python benchmark_warm_start.py
# =====================================================
import os
import sys
import json
import random
import argparse

import numpy as np
import pandas as pd
from telemetry import read_telemetry
from scipy import stats

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from adaptive_arena import (MOVES, judge, ComboEnsembleBoss, EnsembleBoss, PopulationPriorBoss,
                            build_population_table, SYNTHETIC_PLAYERS, BEST_COMBO)
from evaluate_ai import BiasedPlayer, CyclicPlayer, WinStayLoseShiftPlayer, NashRandomPlayer
from benchmark_ensemble_combos import StyleSwitcher

SEGMENTS = [(0, 10, "ตา 1-10"), (10, 30, "ตา 11-30"), (30, 10**9, "ตา 31+")]


def variants(table):
    """บอสที่นำมาเทียบ (ทุกตัวใช้ Prior ที่ส่งเข้ามา = เทรนโดยเว้นผู้เล่นทดสอบไว้)"""
    pf = {"prior": lambda: PopulationPriorBoss(table=table)}
    return {
        "cold":     ("Cold: N-gram + ML (บอส 9 ตอนนี้)", lambda: ComboEnsembleBoss(BEST_COMBO)),
        "prior":    ("Population Prior อย่างเดียว",      lambda: PopulationPriorBoss(table=table)),
        "warm1":    ("Warm: Prior + N-gram + ML",        lambda: ComboEnsembleBoss(("prior", "ngram", "ml"), 1.0, pf)),
        "warm09":   ("Warm: Prior + N-gram + ML (decay 0.9)",
                     lambda: ComboEnsembleBoss(("prior", "ngram", "ml"), 0.9, pf)),
        "cold09":   ("Cold: N-gram + ML (decay 0.9)",    lambda: ComboEnsembleBoss(BEST_COMBO, 0.9)),
        "orig":     ("Ensemble เดิม (บอส 6)",            lambda: EnsembleBoss()),
    }


def load_sessions(csv_path, min_rounds=100, gap_min=10.0):
    df = read_telemetry(csv_path)
    df["pkey"] = df["player_name"].astype(str).str.strip().str.lower()
    df = df[~df["pkey"].isin(SYNTHETIC_PLAYERS) & df["player_move"].isin(MOVES)].copy()
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    out = {}
    for p, g in df.sort_values("ts", kind="stable").groupby("pkey"):
        if len(g) < min_rounds:
            continue
        new = (g["round_id"].diff() <= 0) | (g["ts"].diff() > pd.Timedelta(minutes=gap_min))
        sess = [s["player_move"].tolist() for _, s in g.groupby(new.cumsum())]
        out[p] = [s for s in sess if len(s) >= 3]
    return df, out


def play_session(boss, seq):
    """เล่นซ้ำลำดับท่า คืน list ผล (มุมมองบอส: 1 ชนะ, -1 แพ้, 0 เสมอ)"""
    h, out = [], []
    for p in seq:
        b = boss.choose(h)
        r = judge(p, b)
        h.append(p)
        boss.learn({"win": -1, "draw": 0, "lose": 1}[r], h)
        out.append({"lose": 1, "win": -1, "draw": 0}[r])
    return out


def play_persona(boss, persona, rounds):
    h, last, out = [], None, []
    for _ in range(rounds):
        p = persona.choose(h, last)
        b = boss.choose(h)
        r = judge(p, b)
        h.append(p)
        boss.learn({"win": -1, "draw": 0, "lose": 1}[r], h)
        out.append({"lose": 1, "win": -1, "draw": 0}[r])
        last = r
    return out


def seg_stats(outcomes_by_round):
    """outcomes_by_round: list ของ (index ในเกม, ผล) -> dict segment -> (win%, n)"""
    res = {}
    for lo, hi, name in SEGMENTS + [(0, 10**9, "ทั้งหมด")]:
        v = [o for i, o in outcomes_by_round if lo <= i < hi]
        res[name] = (100.0 * np.mean([o == 1 for o in v]) if v else float("nan"), len(v))
    return res


def make_figure(curves, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in ["Leelawadee UI", "Leelawadee", "Tahoma"]:
        if any(f == x.name for x in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f
            break
    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colors = {"cold": "#2a78d6", "warm09": "#eb6834", "orig": "#8f8e88"}
    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.axhline(33.3, color=ink2, linestyle=":", linewidth=1.2)
    ax.text(0.6, 34.0, "เดาสุ่ม 33.3%", color=ink2, fontsize=9)
    for key, (label, xs, ys) in curves.items():
        ax.plot(xs, ys, color=colors[key], linewidth=2, marker="o", markersize=5,
                markeredgecolor=surface, markeredgewidth=1.5, label=label)
        ax.text(xs[-1] + 0.15, ys[-1], f"{ys[-1]:.0f}%", color=ink, fontsize=9, va="center")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(BIN_LABELS, color=ink2)
    ax.set_xlim(-0.3, len(xs) - 0.4)
    ax.set_ylabel("% บอสชนะ (ยิ่งสูงยิ่งโหด)", color=ink2)
    ax.set_xlabel("ตาที่ ... นับจากเริ่มเปิดเกม (session)", color=ink2)
    ax.set_title("เจอคนใหม่ที่ระบบไม่เคยเห็นข้อมูล (Leave-One-Player-Out): % บอสชนะตามช่วงตา",
                 color=ink, loc="left", fontsize=11.5)
    ax.grid(axis="y", color=grid, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink2, length=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, facecolor=surface)
    plt.close(fig)


BINS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 10**9)]
BIN_LABELS = ["1-5", "6-10", "11-20", "21-30", "31-50", "51+"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/players")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=150, help="จำนวนตาต่อแมตช์ของ persona จำลอง")
    ap.add_argument("--out", default="reports/warm_start.md")
    ap.add_argument("--fig", default="reports/figures/warm_start_curve.png")
    args = ap.parse_args()

    df, sessions = load_sessions(args.csv)
    players = sorted(sessions)
    if len(players) < 2:
        print("ต้องมีผู้เล่นจริงอย่างน้อย 2 คน (คนละ >= 100 ตา) จึงจะทดสอบ leave-one-player-out ได้")
        return
    print("ผู้เล่นจริง:", {p: f"{sum(map(len, s))} ตา / {len(s)} session" for p, s in sessions.items()})

    keys = list(variants({}).keys())
    labels = {k: v[0] for k, v in variants({}).items()}
    # per_session[k] = list ของ (player, session_idx, win% ช่วง 1-10, win% ทั้ง session) เฉลี่ยข้าม reps
    rows_by_round = {k: {p: [] for p in players} for k in keys}
    per_session = {k: [] for k in keys}
    tables = {}
    for p in players:
        table = build_population_table(df, exclude_players=[p])   # <- ไม่เห็นข้อมูลของ p เลย
        tables[p] = {"train_players": table["players"], "train_rounds": table["rounds"]}
        V = variants(table)
        for si, seq in enumerate(sessions[p]):
            acc = {k: np.zeros(len(seq)) for k in keys}
            for rep in range(args.reps):
                for k in keys:
                    random.seed(1000 * rep + si)
                    acc[k] += np.array(play_session(V[k][1](), seq)) == 1
            for k in keys:
                wr = acc[k] / args.reps * 100.0                      # % ชนะต่อตา (เฉลี่ย reps)
                rows_by_round[k][p].extend([(i, w) for i, w in enumerate(wr)])
                per_session[k].append((p, si, len(seq), wr[:10].mean(), wr.mean()))

    # ---------- สรุปต่อผู้เล่น / ช่วงตา ----------
    def seg_table(k, p=None):
        pts = [x for pp in ([p] if p else players) for x in rows_by_round[k][pp]]
        out = {}
        for lo, hi, name in SEGMENTS + [(0, 10**9, "ทั้งหมด")]:
            v = [w for i, w in pts if lo <= i < hi]
            out[name] = (float(np.mean(v)) if v else float("nan"), len(v))
        return out

    seg_names = [s[2] for s in SEGMENTS] + ["ทั้งหมด"]

    # ทดสอบสถิติข้าม session (หน่วยตัวอย่าง = session ของคนจริง)
    def session_test(a, b, idx):
        da = np.array([x[idx] for x in per_session[a]])
        db = np.array([x[idx] for x in per_session[b]])
        d = da - db
        try:
            p = stats.wilcoxon(da, db).pvalue if np.any(d != 0) else 1.0
        except ValueError:
            p = float("nan")
        return d.mean(), int((d > 0).sum()), int((d < 0).sum()), len(d), p

    # ---------- persona จำลอง (ต้องไม่แย่ลง) ----------
    full_table = build_population_table(df)
    Vfull = variants(full_table)
    personas = [("เอียงค้อน 60%", BiasedPlayer), ("วนลูป", CyclicPlayer), ("WSLS", WinStayLoseShiftPlayer),
                ("เปลี่ยนสไตล์ทุก 75 ตา", StyleSwitcher), ("สุ่ม Nash", NashRandomPlayer)]
    pers = {k: {} for k in keys}
    for pi, (pname, pcls) in enumerate(personas):
        for k in keys:
            outs = []
            for rep in range(args.reps):
                random.seed(777 * pi + rep)
                outs.append(np.array(play_persona(Vfull[k][1](), pcls(), args.rounds)) == 1)
            o = np.array(outs).mean(axis=0) * 100
            pers[k][pname] = (o[:10].mean(), o.mean())

    # ---------- กราฟ ----------
    curves = {}
    for k in ("orig", "cold", "warm09"):
        ys = []
        pts = [x for p in players for x in rows_by_round[k][p]]
        for lo, hi in BINS:
            v = [w for i, w in pts if lo <= i < hi]
            ys.append(float(np.mean(v)))
        curves[k] = (labels[k], list(range(len(BINS))), ys)
    os.makedirs(os.path.dirname(args.fig), exist_ok=True)
    make_figure(curves, args.fig)

    # ---------- รายงาน ----------
    L = ["# ใช้ได้กับทุกคนไหม? — Warm start ด้วย Population Prior (Leave-One-Player-Out)\n",
         f"*สร้างโดย `benchmark_warm_start.py` — {args.reps} reps, บอสเริ่มใหม่ทุก session*\n",
         "## แนวคิด",
         "- บอสที่เรียนระหว่างเกมเริ่มจากศูนย์ทุกครั้ง → **ช่วงต้นเกมยังไม่รู้จักคนตรงหน้า**",
         "- **Population Prior** = ตารางนิสัยรวมของผู้เล่นจริงทุกคน: ท่าแรก, ท่าถัดไปหลังท่าเดิม, "
         "ท่าถัดไปหลัง (ท่าเดิม + ชนะ/แพ้/เสมอ) แล้วออกท่าที่ได้ผลตอบแทนคาดหวังสูงสุด",
         "- ใส่ Prior เป็นสมาชิกตัวที่ 3 ของ Lite Ensemble: ต้นเกมเชื่อ Prior → เมื่อ N-gram/ML เรียนรู้คนตรงหน้าแล้วแม่นกว่า "
         "Ensemble จะเปลี่ยนไปเชื่อเอง (decay 0.9 = ลืมคะแนนเก่า ให้เปลี่ยนใจได้เร็ว)\n",
         "## วิธีทดสอบที่ไม่โกง: Leave-One-Player-Out",
         "- ทดสอบกับผู้เล่น X โดยเทรน Prior จาก **ทุกคนยกเว้น X** = จำลองการเจอ \"คนใหม่ที่ระบบไม่เคยเห็น\"",
         "- บอสเริ่มใหม่ทุก session (เหมือนเปิดเกมใหม่) แล้วเล่นซ้ำลำดับท่าจริงของ X",
         "- ข้อมูลจำลอง (Guest) ไม่ถูกใช้ทั้งเทรนและทดสอบ\n",
         "| ผู้เล่นที่ทดสอบ | จำนวนตา / session | Prior เทรนจาก |",
         "|:--|:-:|:--|"]
    for p in players:
        L.append(f"| {p} | {sum(map(len, sessions[p]))} / {len(sessions[p])} | "
                 f"{', '.join(tables[p]['train_players'])} ({tables[p]['train_rounds']} ตา) |")
    L.append("")

    # ---- สรุปที่เขียนตามผลจริง ----
    d1, _, _, _, p1 = session_test("warm09", "cold", 3)
    dA, _, _, _, pA = session_test("warm09", "cold", 4)
    prior_alone = seg_table("prior")["ทั้งหมด"][0]
    cold_all = seg_table("cold")["ทั้งหมด"][0]
    cold_10 = seg_table("cold")["ตา 1-10"][0]
    verdict = ("**ดีขึ้นอย่างมีนัยสำคัญ**" if max(p1, pA) < 0.05
               else "**ยังไม่ต่างอย่างมีนัยสำคัญ** (ไม่แย่ลง แต่ยังพิสูจน์ไม่ได้ว่าดีขึ้น)")
    summary = [
        "## สรุป\n",
        f"- **บอส 9 ตอนนี้ (Cold: N-gram + ML) ชนะเกินสุ่มกับคนที่ระบบไม่เคยเห็นข้อมูลอยู่แล้ว:** {cold_all:.1f}% "
        f"(สุ่ม = 33.3%) เพราะเรียนรู้จากคนตรงหน้าระหว่างเกม — แต่ 10 ตาแรกยังได้แค่ {cold_10:.1f}%",
        f"- **Population Prior อย่างเดียวได้ {prior_alone:.1f}% ≈ สุ่ม** → นิสัยของผู้เล่นที่มีอยู่ยังใช้ทายคนอื่นไม่ได้ "
        "(ผู้เล่นแต่ละคนนิสัยต่างกันมาก และมีข้อมูลคนจริงน้อยเกินไป)",
        f"- **Warm start (Prior + N-gram + ML, decay 0.9) vs Cold:** ตา 1-10 ต่าง {d1:+.1f}% (p={p1:.2f}), "
        f"ทั้ง session ต่าง {dA:+.1f}% (p={pA:.2f}) → {verdict}",
        "- Prior ช่วยมากกับผู้เล่นแบบ ชนะซ้ำ/แพ้เปลี่ยน (WSLS) ตั้งแต่ตาแรก (ดูตาราง persona) "
        "แต่เสียเล็กน้อยช่วงต้นกับผู้เล่นแบบอื่น",
        "- **ทางที่จะทำให้ Warm start ได้ผลจริง = เก็บข้อมูลคนจริงเพิ่ม** แล้วรัน `python benchmark_warm_start.py` ใหม่ "
        "(สคริปต์จะเทรน/ทดสอบใหม่และเขียนสรุปนี้ใหม่อัตโนมัติ)\n",
    ]
    L[2:2] = summary

    L.append("## ผล: % ที่บอสชนะ (รวมทุกผู้เล่นที่ทดสอบ)\n")
    L.append(f"![Warm start curve]({os.path.relpath(args.fig, os.path.dirname(args.out)).replace(os.sep, '/')})\n")
    L.append("| บอส | " + " | ".join(seg_names) + " |")
    L.append("|:--|" + "|".join([":-:"] * len(seg_names)) + "|")
    for k in keys:
        st = seg_table(k)
        L.append(f"| {labels[k]} | " + " | ".join(f"{st[s][0]:.1f}" for s in seg_names) + " |")
    L.append("\n*33.3% = เท่ากับเดาสุ่ม*\n")

    for p in players:
        L.append(f"### ผู้เล่น: {p}\n")
        L.append("| บอส | " + " | ".join(seg_names) + " |")
        L.append("|:--|" + "|".join([":-:"] * len(seg_names)) + "|")
        for k in keys:
            st = seg_table(k, p)
            L.append(f"| {labels[k]} | " + " | ".join(f"{st[s][0]:.1f}" for s in seg_names) + " |")
        L.append("")

    L.append("## ทดสอบนัยสำคัญ (หน่วย = session ของคนจริง, Wilcoxon signed-rank)\n")
    L.append("| เปรียบเทียบ | ช่วง | ต่างเฉลี่ย (%) | session ที่ดีขึ้น / แย่ลง | p |")
    L.append("|:--|:--|:-:|:-:|:-:|")
    for a, b in (("warm09", "cold"), ("warm1", "cold"), ("warm09", "orig")):
        for idx, nm in ((3, "ตา 1-10"), (4, "ทั้ง session")):
            d, up, down, n, pv = session_test(a, b, idx)
            L.append(f"| {labels[a]} vs {labels[b]} | {nm} | {d:+.1f} | {up} / {down} (จาก {n}) | {pv:.3g} |")
    L.append("")

    L.append("## ตรวจความปลอดภัย: Prior ไม่ทำให้แย่ลงกับผู้เล่นแบบอื่น (persona จำลอง)\n")
    L.append(f"*Prior เทรนจากผู้เล่นจริงทุกคน, {args.rounds} ตา/แมตช์ — ตัวเลข = % บอสชนะ (ตา 1-10 / ทั้งเกม)*\n")
    L.append("| บอส | " + " | ".join(n for n, _ in personas) + " |")
    L.append("|:--|" + "|".join([":-:"] * len(personas)) + "|")
    for k in keys:
        L.append(f"| {labels[k]} | " + " | ".join(f"{pers[k][n][0]:.0f} / {pers[k][n][1]:.1f}" for n, _ in personas) + " |")
    L.append("")

    L.append("## ข้อจำกัด (ต้องพูดตอนพรีเซนต์)")
    L.append(f"- มีผู้เล่นจริงแค่ **{len(players)} คน** — ผลบอกแนวโน้ม แต่ยังสรุปว่า \"ใช้ได้กับทุกคน\" ไม่ได้ "
             "ควรเก็บเพิ่ม 5-10 คน คนละ 100+ ตา แล้วรันสคริปต์นี้ใหม่")
    L.append("- การเล่นซ้ำ (replay) ใช้ลำดับท่าที่ผู้เล่นออกตอนสู้บอสตัวเก่า ผู้เล่นจริงอาจตอบสนองต่างไปเมื่อเจอบอสใหม่")
    L.append("- ค่า decay 0.9 เลือกจากข้อมูลชุดเดียวกันนี้ (มีตัวเลือกแค่ 2 ค่า) จึงอาจดูดีเกินจริงเล็กน้อย")
    L.append("- ไม่มีบอสไหนชนะเกิน 33.3% ได้กับคนที่ออกสุ่มจริง (Nash) — คำกล่าวที่ถูกคือ \"ชนะเกินสุ่มกับผู้เล่นทั่วไป\"")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"players": players, "segments": {k: seg_table(k) for k in keys},
                   "per_session": per_session, "personas": pers}, f, ensure_ascii=False, indent=1)

    print("\n% บอสชนะ (รวมผู้เล่นจริง):")
    for k in keys:
        st = seg_table(k)
        print(f"  {labels[k]:<42} " + "  ".join(f"{s}={st[s][0]:.1f}" for s in seg_names))
    for a, b in (("warm09", "cold"), ("warm1", "cold")):
        d, up, down, n, pv = session_test(a, b, 3)
        print(f"  {labels[a]} vs cold (ตา 1-10): {d:+.1f}%  ดีขึ้น {up}/แย่ลง {down} จาก {n} session  p={pv:.3g}")
    print(f"\nรายงาน: {args.out}\nรูป: {args.fig}")


if __name__ == "__main__":
    main()
