# =====================================================
#  test_efficiency.py — คำนวณค่า Efficiency Score (%)
#  ตามสูตรที่ผู้ใช้ระบุ:
#  1. R = (W + 0.5 * D) / N * 100
#  2. Efficiency (%) = (R_model - R_random) / (100 - R_random) * 100
# =====================================================
import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from adaptive_arena import BOSSES
from evaluate_ai import PERSONAS, evaluate_matchup

rounds = 1000

# 1. จำลองการแข่งและเก็บสถิติ W, D, L
raw_stats = {}
for b_cls in BOSSES:
    b_name = b_cls.name.split(".")[1].split("(")[0].strip()
    raw_stats[b_name] = {}
    for p_cls in PERSONAS:
        p_name = p_cls.name.split(".")[1].strip()
        win_rate, loss_rate, draw_rate = evaluate_matchup(b_cls, p_cls, rounds=rounds)
        W = int(round(win_rate * rounds / 100))
        L = int(round(loss_rate * rounds / 100))
        D = int(round(draw_rate * rounds / 100))
        # คำนวณ R ตามสูตรที่ผู้ใช้ให้: R = (W + 0.5 * D) / N * 100
        R = ((W + 0.5 * D) / rounds) * 100
        raw_stats[b_name][p_name] = {
            "W": W, "D": D, "L": L, "R": R, "pure_win": win_rate
        }

p_names = [p_cls.name.split(".")[1].strip() for p_cls in PERSONAS]

print("=" * 100)
print("ตารางที่ 1: อัตราการชนะดิบ (Raw Score: R = (W + 0.5*D)/N * 100)")
print("=" * 100)
print(f"{'โมเดล AI บอส':<22} | " + " | ".join([f"{pn:<18}" for pn in p_names]))
print("-" * 100)
for b_name, p_data in raw_stats.items():
    vals = [f"R = {p_data[pn]['R']:5.1f}%" for pn in p_names]
    print(f"{b_name:<22} | " + " | ".join(vals))

print("\n" + "=" * 100)
print("ตารางที่ 2: ค่าประสิทธิภาพ Efficiency (%) เทียบกับโมเดล 'สุ่มล้วน' (Random เป็น Baseline = 0%)")
print("สูตร: Efficiency (%) = (R_model - R_random) / (100 - R_random) * 100")
print("=" * 100)
print(f"{'โมเดล AI บอส':<22} | " + " | ".join([f"{pn:<18}" for pn in p_names]))
print("-" * 100)
for b_name, p_data in raw_stats.items():
    eff_vals = []
    for pn in p_names:
        r_model = p_data[pn]["R"]
        r_random = raw_stats["Random"][pn]["R"]
        # ป้องกันส่วนเป็น 0
        if 100.0 - r_random == 0:
            eff = 0.0
        else:
            eff = ((r_model - r_random) / (100.0 - r_random)) * 100.0
        sign = "+" if eff > 0 else ""
        eff_vals.append(f"{sign}{eff:5.1f}%")
    print(f"{b_name:<22} | " + " | ".join([f"{v:<18}" for v in eff_vals]))

print("\n" + "=" * 100)
print("ตารางที่ 3: กรณีเทียบกับเกณฑ์ R_random = 33.33% (ตามความน่าจะเป็นชนะบริสุทธิ์)")
print("สูตร: Efficiency (%) = (PureWin_model - 33.33) / (100 - 33.33) * 100")
print("=" * 100)
print(f"{'โมเดล AI บอส':<22} | " + " | ".join([f"{pn:<18}" for pn in p_names]))
print("-" * 100)
for b_name, p_data in raw_stats.items():
    eff_vals = []
    for pn in p_names:
        w_model = p_data[pn]["pure_win"]
        r_random = 33.33
        eff = ((w_model - r_random) / (100.0 - r_random)) * 100.0
        sign = "+" if eff > 0 else ""
        eff_vals.append(f"{sign}{eff:5.1f}%")
    print(f"{b_name:<22} | " + " | ".join([f"{v:<18}" for v in eff_vals]))
