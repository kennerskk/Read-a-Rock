import sys
import random
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from adaptive_arena import BOSSES, judge, MOVES, BEATS, COUNTER


# =====================================================
# 1. สร้าง Persona จำลองพฤติกรรมมนุษย์รูปแบบต่างๆ
# =====================================================
class BiasedPlayer:
    """มนุษย์ที่ชอบออกท่าใดท่าหนึ่งเป็นพิเศษ (เช่น ออกค้อน 60%)"""
    name = "1. มนุษย์เอียง (ออกค้อน 60%)"
    def choose(self, history, last_result):
        r = random.random()
        if r < 0.60:
            return "ค้อน"
        elif r < 0.80:
            return "กรรไกร"
        else:
            return "กระดาษ"


class CyclicPlayer:
    """มนุษย์ที่เล่นวนตามลูป (ค้อน -> กรรไกร -> กระดาษ)"""
    name = "2. มนุษย์เล่นวนลูป (ค้อน->กรรไกร->กระดาษ)"
    def __init__(self):
        self.seq = ["ค้อน", "กรรไกร", "กระดาษ"]
        self.idx = 0
    def choose(self, history, last_result):
        move = self.seq[self.idx % len(self.seq)]
        self.idx += 1
        return move


class WinStayLoseShiftPlayer:
    """
    มนุษย์จิตวิทยา Win-Stay, Lose-Shift (WSLS - งานวิจัย Nature 2014)
    - ถ้าตาที่แล้วชนะ -> มักจะออกท่าเดิมซ้ำ (Stay)
    - ถ้าตาที่แล้วแพ้  -> มักจะเปลี่ยนไปออกท่าที่ชนะท่าที่เพิ่งแพ้ (Shift)
    """
    name = "3. มนุษย์จิตวิทยา (ชนะออกซ้ำ/แพ้เปลี่ยนท่า)"
    def __init__(self):
        self.last_move = "ค้อน"
    def choose(self, history, last_result):
        if not history or last_result is None:
            self.last_move = random.choice(MOVES)
            return self.last_move

        if last_result == "win":  # ผู้เล่นชนะ ออกท่าเดิม
            move = self.last_move
        elif last_result == "lose":  # ผู้เล่นแพ้ เปลี่ยนไปออกท่าที่แก้ทาง
            move = COUNTER[self.last_move]
        else:
            move = random.choice(MOVES)

        self.last_move = move
        return move


class NashRandomPlayer:
    """มนุษย์ที่สุ่มสมบูรณ์แบบ (Nash Equilibrium 1/3 เท่ากันทุกท่า)"""
    name = "4. มนุษย์สุ่ม 100% (Nash Equilibrium)"
    def choose(self, history, last_result):
        return random.choice(MOVES)


PERSONAS = [BiasedPlayer, CyclicPlayer, WinStayLoseShiftPlayer, NashRandomPlayer]


# =====================================================
# 2. ฟังก์ชันจำลองการแข่งขัน (Simulation Engine)
# =====================================================
def evaluate_matchup(boss_cls, persona_cls, rounds=1000):
    boss = boss_cls()
    persona = persona_cls()

    opp_history = []
    boss_wins = 0
    player_wins = 0
    draws = 0
    last_result = None

    for _ in range(rounds):
        player_move = persona.choose(opp_history, last_result)
        boss_move = boss.choose(opp_history)

        # ตัดสินผล (มุมมองผู้เล่น)
        result = judge(player_move, boss_move)
        opp_history.append(player_move)

        # ให้บอสเรียนรู้ (reward: บอสชนะ +1, เสมอ 0, บอสแพ้ -1)
        reward = {"win": -1, "draw": 0, "lose": 1}[result]
        boss.learn(reward, opp_history)

        if result == "win":
            player_wins += 1
            last_result = "win"
        elif result == "lose":
            boss_wins += 1
            last_result = "lose"
        else:
            draws += 1
            last_result = "draw"

    win_rate = (boss_wins / rounds) * 100
    loss_rate = (player_wins / rounds) * 100
    draw_rate = (draws / rounds) * 100
    return win_rate, loss_rate, draw_rate


def run_all_evaluations():
    rounds = 1000
    print("=" * 85)
    print(f"  ผลการประเมินประสิทธิภาพ AI บอส (จำลองแข่ง {rounds} ตาต่อคู่ต่อสู้)")
    print("=" * 85)

    headers = [p_cls().name[:18] for p_cls in PERSONAS]
    header_line = f"{'ชื่อโมเดลบอส AI':<22} | " + " | ".join([f"{h:<18}" for h in headers])
    print(header_line)
    print("-" * 85)

    results_table = {}

    for b_cls in BOSSES:
        boss_name = b_cls.name.split(".")[1].split("(")[0].strip()
        row_str = f"{boss_name:<22} | "
        results_table[boss_name] = {}

        for p_cls in PERSONAS:
            win_rate, loss_rate, draw_rate = evaluate_matchup(b_cls, p_cls, rounds=rounds)
            results_table[boss_name][p_cls] = win_rate
            row_str += f"AI ชนะ {win_rate:5.1f}%   | "

        print(row_str)

    print("=" * 85)
    print("\n💡 การวิเคราะห์ผลเชิงวิชาการ:")
    print("1. เมื่อเจอกับ 'มนุษย์สุ่ม (Nash)' -> บอสทุกตัวจะชนะใกล้เคียง ~33.3% (ตามทฤษฎี Game Theory ไม่สามารถโกงความน่าจะเป็นบริสุทธิ์ได้)")
    print("2. เมื่อเจอกับ 'มนุษย์ที่มีพฤติกรรม/แพทเทิร์น' (เอียง, วนลูป, แพ้เปลี่ยนท่า) -> บอสที่มีโมเดลเรียนรู้ (Ensemble, Markov, Frequency) จะชนะสูงถึง 55% - 100%!")


import sys
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


if __name__ == "__main__":
    run_all_evaluations()
