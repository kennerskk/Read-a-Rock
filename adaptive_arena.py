# =====================================================
#  adaptive_arena.py — เกมเป่ายิ้งฉุบ (ค้อน/กรรไกร/กระดาษ)
#  เล่นเองกับ "บอส AI" ที่เรียนรู้จับทางผู้เล่น
#  เลือกได้ว่าจะเจอบอสสมองแบบไหน (8 algorithm + Custom Loaded Boss)
#  รองรับการ Import / Export สถานะสมองของบอส
#
#  วิธีเล่น:  python adaptive_arena.py
#  ท่า: ค้อนชนะกรรไกร | กรรไกรชนะกระดาษ | กระดาษชนะค้อน
# =====================================================
import os
import sys
import random
from collections import Counter, defaultdict
from typing import Dict, Any, List, Optional

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

MOVES = ["ค้อน", "กรรไกร", "กระดาษ"]
BEATS = {"ค้อน": "กรรไกร", "กรรไกร": "กระดาษ", "กระดาษ": "ค้อน"}   # key ชนะ value
# ท่าที่ใช้ "แก้ทาง" ท่า P ของผู้เล่น (กระดาษชนะค้อน, ค้อนชนะกรรไกร, กรรไกรชนะกระดาษ)
COUNTER = {"ค้อน": "กระดาษ", "กรรไกร": "ค้อน", "กระดาษ": "กรรไกร"}


# =====================================================
#  ตัวตัดสินผล  (มุมมองของผู้เล่น)
#  คืน "win" / "lose" / "draw"
# =====================================================
def judge(player_move: str, boss_move: str) -> str:
    if player_move == boss_move:
        return "draw"
    if BEATS[player_move] == boss_move:
        return "win"      # ท่าผู้เล่นชนะท่าบอส
    return "lose"


# =====================================================
#  บอสทุกตัวมีโครงสร้างเหมือนกัน:
#    .choose(opp_history) -> ท่าบอส
#    .learn(reward, opp_history) -> เรียนรู้
#    .get_state() -> dict (สำหรับ export)
#    .load_state(state) -> None (สำหรับ import)
#  opp_history = list ท่าที่ผู้เล่นเคยออก (เก่า -> ใหม่)
#  reward (มุมมองบอส): บอสชนะ +1 / เสมอ 0 / บอสแพ้ -1
# =====================================================
class Boss:
    name = "Boss"
    desc = ""

    def choose(self, opp_history: List[str]) -> str:
        return random.choice(MOVES)

    def learn(self, reward: int, opp_history: List[str]):
        pass

    def get_state(self) -> Dict[str, Any]:
        """คืนค่าสถานะสมองที่เรียนรู้เพื่อ Export เป็นโมเดล"""
        return {}

    def load_state(self, state: Dict[str, Any]):
        """โหลดสถานะสมองกลับคืนมา"""
        pass

    def export_model(self, filepath: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> str:
        """บันทึกโมเดลลงไฟล์ JSON ผ่าน model_io"""
        from model_io import save_boss_model
        return save_boss_model(self, filepath, metadata)

    @classmethod
    def from_file(cls, filepath: str):
        """โหลดโมเดลจากไฟล์ JSON ผ่าน model_io"""
        from model_io import load_boss_model
        return load_boss_model(filepath)


# ---------- Tier 1: ง่าย ----------
class RandomBoss(Boss):
    name = "1. Random (สุ่มล้วน)"
    desc = "ออกท่ามั่ว 1/3 เท่ากัน — ไม่เรียนรู้ ใช้เป็นเกณฑ์ Baseline"

    def choose(self, opp_history: List[str]) -> str:
        return random.choice(MOVES)


class FrequencyBoss(Boss):
    name = "2. Frequency (จับท่าที่ชอบ)"
    desc = "นับท่าที่ผู้เล่นออกบ่อยสุดตลอดเกม แล้วออกท่าแก้ทาง"

    def __init__(self):
        self.counts = Counter()

    def choose(self, opp_history: List[str]) -> str:
        if self.counts:
            predicted = self.counts.most_common(1)[0][0]
            return COUNTER[predicted]
        if not opp_history:
            return random.choice(MOVES)
        predicted = Counter(opp_history).most_common(1)[0][0]
        return COUNTER[predicted]

    def learn(self, reward: int, opp_history: List[str]):
        if opp_history:
            self.counts[opp_history[-1]] += 1

    def get_state(self) -> Dict[str, Any]:
        return {"counts": dict(self.counts)}

    def load_state(self, state: Dict[str, Any]):
        self.counts = Counter(state.get("counts", {}))


class DecayBoss(Boss):
    name = "3. Recency-decay (จับท่าที่ชอบช่วงหลัง)"
    desc = "ให้น้ำหนักท่าล่าสุดมากกว่า ปรับตามเมื่อผู้เล่นเปลี่ยนสไตล์"

    def __init__(self, decay: float = 0.85):
        self.decay = decay
        self.weights = {m: 0.0 for m in MOVES}

    def choose(self, opp_history: List[str]) -> str:
        # หากมีค่าสะสม weights ใช้น้ำหนักสะสม
        if any(v > 0 for v in self.weights.values()):
            predicted = max(self.weights, key=self.weights.get)
            return COUNTER[predicted]
        if not opp_history:
            return random.choice(MOVES)
        weights = {m: 0.0 for m in MOVES}
        w = 1.0
        for move in reversed(opp_history):
            weights[move] += w
            w *= self.decay
        predicted = max(weights, key=weights.get)
        return COUNTER[predicted]

    def learn(self, reward: int, opp_history: List[str]):
        if opp_history:
            last = opp_history[-1]
            for m in MOVES:
                self.weights[m] *= self.decay
            self.weights[last] += 1.0

    def get_state(self) -> Dict[str, Any]:
        return {"weights": dict(self.weights), "decay": self.decay}

    def load_state(self, state: Dict[str, Any]):
        self.decay = state.get("decay", 0.85)
        self.weights = {m: float(state.get("weights", {}).get(m, 0.0)) for m in MOVES}


# ---------- Tier 2: กลาง ----------
class MarkovBoss(Boss):
    name = "4. Markov (หลังท่า X มักตามด้วยอะไร)"
    desc = "จับลำดับสั้น: ดูว่าหลังท่าล่าสุด ผู้เล่นมักตามด้วยท่าไหน"

    def __init__(self):
        self.table = defaultdict(Counter)

    def choose(self, opp_history: List[str]) -> str:
        if not opp_history:
            return random.choice(MOVES)
        last = opp_history[-1]
        if self.table[last]:
            predicted = self.table[last].most_common(1)[0][0]
            return COUNTER[predicted]
        # สำรองด้วยการคำนวณจากประวัติสด
        if len(opp_history) >= 2:
            t = defaultdict(Counter)
            for prev, nxt in zip(opp_history, opp_history[1:]):
                t[prev][nxt] += 1
            if t[last]:
                predicted = t[last].most_common(1)[0][0]
                return COUNTER[predicted]
        return random.choice(MOVES)

    def learn(self, reward: int, opp_history: List[str]):
        if len(opp_history) >= 2:
            prev = opp_history[-2]
            nxt = opp_history[-1]
            self.table[prev][nxt] += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "table": {prev: dict(cnt) for prev, cnt in self.table.items()}
        }

    def load_state(self, state: Dict[str, Any]):
        self.table = defaultdict(Counter)
        raw_table = state.get("table", {})
        for prev, cnt in raw_table.items():
            self.table[prev] = Counter(cnt)


class NgramBoss(Boss):
    name = "5. N-gram (จับลำดับ k ท่าล่าสุด)"
    desc = "มองลำดับ k ท่าติดกันเป็น 'สถานะ' จับแพทเทิร์นซับซ้อนกว่า Markov"

    def __init__(self, k: int = 2):
        self.k = k
        self.table = defaultdict(Counter)

    def choose(self, opp_history: List[str]) -> str:
        k = self.k
        if len(opp_history) < k:
            return random.choice(MOVES)
        key = tuple(opp_history[-k:])
        if self.table[key]:
            predicted = self.table[key].most_common(1)[0][0]
            return COUNTER[predicted]
        # สำรองด้วยประวัติสด
        if len(opp_history) > k:
            t = defaultdict(Counter)
            for i in range(len(opp_history) - k):
                kk = tuple(opp_history[i:i + k])
                t[kk][opp_history[i + k]] += 1
            if t[key]:
                predicted = t[key].most_common(1)[0][0]
                return COUNTER[predicted]
        return random.choice(MOVES)

    def learn(self, reward: int, opp_history: List[str]):
        k = self.k
        if len(opp_history) >= k + 1:
            key = tuple(opp_history[-k-1:-1])
            nxt = opp_history[-1]
            self.table[key][nxt] += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "table": {"|".join(k): dict(cnt) for k, cnt in self.table.items()}
        }

    def load_state(self, state: Dict[str, Any]):
        self.k = state.get("k", 2)
        self.table = defaultdict(Counter)
        raw_table = state.get("table", {})
        for k_str, cnt in raw_table.items():
            key = tuple(k_str.split("|"))
            self.table[key] = Counter(cnt)


# ---------- Tier 3: ขั้นสูง ----------
class EnsembleBoss(Boss):
    name = "6. Ensemble/Meta (เลือกโมเดลที่แม่นสุดตอนนี้)"
    desc = "รันหลายโมเดลพร้อมกัน แล้วเชื่อตัวที่ทายแม่นสุดช่วงนี้ (กันโดนหลอก)"

    def __init__(self):
        self.experts = {
            "freq":   FrequencyBoss(),
            "decay":  DecayBoss(),
            "markov": MarkovBoss(),
            "ngram":  NgramBoss(),
        }
        self.scores = {n: 0 for n in self.experts}
        self.last_moves = {}

    def choose(self, opp_history: List[str]) -> str:
        # ให้คะแนนย้อนหลัง: โมเดลไหนทายตาที่แล้วถูก (ท่ามันชนะท่าจริง) บวกแต้ม
        if opp_history and self.last_moves:
            actual = opp_history[-1]
            for n, guessed in self.last_moves.items():
                if COUNTER[actual] == guessed:
                    self.scores[n] += 1
        best = max(self.scores, key=self.scores.get)
        move = self.experts[best].choose(opp_history)
        # จำท่าที่แต่ละโมเดล "จะออก" ไว้ให้คะแนนรอบหน้า
        self.last_moves = {n: e.choose(opp_history) for n, e in self.experts.items()}
        return move

    def learn(self, reward: int, opp_history: List[str]):
        for e in self.experts.values():
            e.learn(reward, opp_history)

    def get_state(self) -> Dict[str, Any]:
        return {
            "scores": dict(self.scores),
            "experts": {n: e.get_state() for n, e in self.experts.items()}
        }

    def load_state(self, state: Dict[str, Any]):
        self.scores = dict(state.get("scores", self.scores))
        exp_states = state.get("experts", {})
        for n, exp in self.experts.items():
            if n in exp_states:
                exp.load_state(exp_states[n])


class QLearningBoss(Boss):
    name = "7. Q-learning (Reinforcement Learning)"
    desc = "เรียนจากรางวัลแพ้/ชนะจริง state=ท่าล่าสุดผู้เล่น action=ท่าบอส"

    def __init__(self, alpha: float = 0.2, gamma: float = 0.9, eps: float = 0.1):
        self.Q = defaultdict(lambda: {m: 0.0 for m in MOVES})
        self.alpha, self.gamma, self.eps = alpha, gamma, eps
        self.prev_state = None
        self.prev_action = None

    def choose(self, opp_history: List[str]) -> str:
        state = opp_history[-1] if opp_history else "START"
        if random.random() < self.eps:
            action = random.choice(MOVES)           # explore
        else:
            action = max(self.Q[state], key=self.Q[state].get)  # exploit
        self.prev_state, self.prev_action = state, action
        return action

    def learn(self, reward: int, opp_history: List[str]):
        if self.prev_state is None:
            return
        next_state = opp_history[-1] if opp_history else "START"
        best_next = max(self.Q[next_state].values())
        old = self.Q[self.prev_state][self.prev_action]
        self.Q[self.prev_state][self.prev_action] = old + self.alpha * (
            reward + self.gamma * best_next - old)

    def get_state(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "eps": self.eps,
            "Q": {s: dict(actions) for s, actions in self.Q.items()}
        }

    def load_state(self, state: Dict[str, Any]):
        self.alpha = state.get("alpha", self.alpha)
        self.gamma = state.get("gamma", self.gamma)
        self.eps = state.get("eps", self.eps)
        raw_q = state.get("Q", {})
        self.Q = defaultdict(lambda: {m: 0.0 for m in MOVES})
        for s, actions in raw_q.items():
            self.Q[s] = {m: float(actions.get(m, 0.0)) for m in MOVES}


# ---------- Tier 4: โมเดล Data Science ขั้นสูง ----------
class MLPredictorBoss(Boss):
    name = "8. Data Science ML Boss (โมเดลพยากรณ์เชิงสถิติและพฤติกรรม)"
    desc = "สกัดฟีเจอร์ WSLS + Markov + ลำดับ N-gram + ปรับน้ำหนักการทำนายแบบ Real-time"

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # น้ำหนักความน่าเชื่อถือของกลยุทธ์ต่างๆ (สกัดจาก Feature Store)
        self.strategy_weights = weights or {
            "wsls": 2.5,       # Win-Stay Lose-Shift (จิตวิทยางานวิจัย Nature)
            "markov_1": 2.0,   # ความน่าจะเป็น Markov ลำดับที่ 1
            "markov_2": 1.8,   # ความน่าจะเป็น Markov ลำดับที่ 2
            "recency": 1.2,    # แนวโน้มท่าช่วงล่าสุด
            "frequency": 0.8   # ความถี่สะสมรวม
        }
        self.history: List[str] = []
        self.results: List[str] = []
        self.transitions = defaultdict(Counter)
        self.wsls_transitions = defaultdict(Counter)
        self.counts = Counter()

    def choose(self, opp_history: List[str]) -> str:
        if not opp_history:
            return random.choice(MOVES)

        # คำนวณความน่าจะเป็นสะสมของแต่ละท่าที่ผู้เล่นจะออก (P_predicted)
        move_scores = {m: 0.0 for m in MOVES}

        # 1. Feature: Win-Stay Lose-Shift (WSLS)
        if self.results and self.history:
            last_res = self.results[-1]
            last_mv = self.history[-1]
            state_key = f"{last_res}_{last_mv}"
            if self.wsls_transitions[state_key]:
                pred = self.wsls_transitions[state_key].most_common(1)[0][0]
                move_scores[pred] += self.strategy_weights.get("wsls", 2.0)
            else:
                # Heuristic ตามงานวิจัย Nature WSLS
                if last_res == "win":   # ผู้เล่นชนะ มักออกท่าเดิม
                    move_scores[last_mv] += self.strategy_weights.get("wsls", 2.0) * 0.7
                elif last_res == "lose": # ผู้เล่นแพ้ มักเปลี่ยนไปท่าที่ชนะท่าที่เพิ่งแพ้
                    move_scores[COUNTER[last_mv]] += self.strategy_weights.get("wsls", 2.0) * 0.7

        # 2. Feature: 1st Order Markov
        last_mv = opp_history[-1]
        if self.transitions[last_mv]:
            pred = self.transitions[last_mv].most_common(1)[0][0]
            move_scores[pred] += self.strategy_weights.get("markov_1", 1.8)

        # 3. Feature: Recency
        decay_w = 1.0
        for m in reversed(opp_history[-5:]):
            move_scores[m] += decay_w * self.strategy_weights.get("recency", 1.0) * 0.3
            decay_w *= 0.8

        # 4. Feature: Frequency
        if self.counts:
            fav = self.counts.most_common(1)[0][0]
            move_scores[fav] += self.strategy_weights.get("frequency", 0.8) * 0.5

        # เลือกท่าที่มีคะแนนพยากรณ์สูงสุด แล้วออกท่าแก้ทาง
        predicted_player_move = max(move_scores, key=move_scores.get)
        return COUNTER[predicted_player_move]

    def learn(self, reward: int, opp_history: List[str]):
        if not opp_history:
            return
        last_player_move = opp_history[-1]
        self.counts[last_player_move] += 1

        if len(opp_history) >= 2:
            prev_move = opp_history[-2]
            self.transitions[prev_move][last_player_move] += 1

        # แปลง reward กลับเป็น result มุมมองผู้เล่น
        res_map = {-1: "win", 0: "draw", 1: "lose"}
        player_res = res_map.get(reward, "draw")
        self.results.append(player_res)
        self.history.append(last_player_move)

        if len(self.history) >= 2 and len(self.results) >= 2:
            prev_res = self.results[-2]
            prev_mv = self.history[-2]
            cur_mv = self.history[-1]
            state_key = f"{prev_res}_{prev_mv}"
            self.wsls_transitions[state_key][cur_mv] += 1

    def get_state(self) -> Dict[str, Any]:
        return {
            "strategy_weights": dict(self.strategy_weights),
            "counts": dict(self.counts),
            "transitions": {k: dict(v) for k, v in self.transitions.items()},
            "wsls_transitions": {k: dict(v) for k, v in self.wsls_transitions.items()}
        }

    def load_state(self, state: Dict[str, Any]):
        self.strategy_weights = dict(state.get("strategy_weights", self.strategy_weights))
        self.counts = Counter(state.get("counts", {}))
        self.transitions = defaultdict(Counter)
        for k, v in state.get("transitions", {}).items():
            self.transitions[k] = Counter(v)
        self.wsls_transitions = defaultdict(Counter)
        for k, v in state.get("wsls_transitions", {}).items():
            self.wsls_transitions[k] = Counter(v)


# ---------- Custom Loaded Boss Wrapper ----------
class CustomSavedBoss(Boss):
    name = "[Custom Boss] โมเดลที่โหลดจากไฟล์"
    desc = "โมเดล AI บอสที่เรียนรู้เฉพาะตัวและถูกนำเข้าจากไฟล์ภายนอก"

    def __init__(self, filepath: Optional[str] = None):
        self.inner_boss: Boss = EnsembleBoss()
        self.loaded_filepath = filepath
        if filepath and os.path.exists(filepath):
            self.load_from_filepath(filepath)

    def load_from_filepath(self, filepath: str):
        from model_io import load_boss_model
        loaded = load_boss_model(filepath)
        self.inner_boss = loaded
        self.name = f"⭐ {getattr(loaded, 'name', 'Custom Boss')}"
        self.desc = f"โหลดจาก {os.path.basename(filepath)}"
        self.loaded_filepath = filepath

    def choose(self, opp_history: List[str]) -> str:
        return self.inner_boss.choose(opp_history)

    def learn(self, reward: int, opp_history: List[str]):
        self.inner_boss.learn(reward, opp_history)

    def get_state(self) -> Dict[str, Any]:
        return {
            "inner_type": self.inner_boss.__class__.__name__,
            "inner_state": self.inner_boss.get_state() if hasattr(self.inner_boss, "get_state") else {},
            "loaded_filepath": self.loaded_filepath
        }

    def load_state(self, state: Dict[str, Any]):
        if "loaded_filepath" in state and state["loaded_filepath"] and os.path.exists(state["loaded_filepath"]):
            self.load_from_filepath(state["loaded_filepath"])
        elif "inner_state" in state:
            inner_type = state.get("inner_type", "EnsembleBoss")
            reg = {b.__name__: b for b in [RandomBoss, FrequencyBoss, DecayBoss, MarkovBoss, NgramBoss, EnsembleBoss, QLearningBoss, MLPredictorBoss]}
            cls = reg.get(inner_type, EnsembleBoss)
            self.inner_boss = cls()
            if hasattr(self.inner_boss, "load_state"):
                self.inner_boss.load_state(state["inner_state"])


BOSSES = [
    RandomBoss,
    FrequencyBoss,
    DecayBoss,
    MarkovBoss,
    NgramBoss,
    EnsembleBoss,
    QLearningBoss,
    MLPredictorBoss
]


# =====================================================
#  เมนูเลือกบอส
# =====================================================
def pick_boss() -> Optional[Boss]:
    from model_io import list_saved_models, load_boss_model

    print("\n===== เลือกสมองบอสที่อยากเจอ =====")
    for idx, cls in enumerate(BOSSES, start=1):
        print(f"  {idx}. {cls.name}")
        print(f"       {cls.desc}")

    saved_models = list_saved_models("models")
    if saved_models:
        print(f"  [L] โหลดโมเดลที่บันทึกไว้ใน models/ ({len(saved_models)} โมเดล)")

    while True:
        choice = input(f"\nพิมพ์เลข 1-{len(BOSSES)} (หรือ L โหลดโมเดล, q ออก): ").strip().lower()
        if choice == "q":
            return None
        if choice == "l":
            if not saved_models:
                print("  ! ยังไม่มีไฟล์โมเดลในโฟลเดอร์ models/")
                continue
            print("\n--- รายการโมเดลที่บันทึกไว้ ---")
            for i, m in enumerate(saved_models, start=1):
                rounds = m.get("stats", {}).get("total_rounds", "?")
                print(f"  [{i}] {m.get('filename')} (บอส: {m.get('boss_name')}, แข่งแล้ว: {rounds} ตา)")
            pick_m = input(f"เลือกโมเดล 1-{len(saved_models)} (หรือกด Enter ยกเลิก): ").strip()
            if pick_m.isdigit() and 1 <= int(pick_m) <= len(saved_models):
                selected = saved_models[int(pick_m) - 1]
                loaded_boss = load_boss_model(selected["filepath"])
                print(f"\n>>> โหลดโมเดลสำเร็จ: {selected['filename']}")
                return loaded_boss
            continue

        if choice.isdigit() and 1 <= int(choice) <= len(BOSSES):
            return BOSSES[int(choice) - 1]()
        print(f"  ! เลือกไม่ถูกต้อง ลองพิมพ์ 1-{len(BOSSES)} หรือ L")


# =====================================================
#  วนเล่นกับบอสที่เลือก
# =====================================================
def play(boss: Boss):
    from model_io import save_boss_model

    opp_history = []          # ท่าที่ผู้เล่นออก
    win = lose = draw = 0
    round_no = 0
    num = {"1": "ค้อน", "2": "กรรไกร", "3": "กระดาษ"}

    print(f"\n>>> เข้าสู้กับ {boss.name}")
    print(">>> ท่า: 1=ค้อน  2=กรรไกร  3=กระดาษ   (s=เซฟโมเดลทันที, q=เลิก)")
    print(">>> กติกา: ค้อนชนะกรรไกร | กรรไกรชนะกระดาษ | กระดาษชนะค้อน\n")

    while True:
        raw = input(f"[ตา {round_no + 1}] ออกท่า (1/2/3, s=เซฟ, q=เลิก): ").strip().lower()
        if raw == "q":
            break
        if raw == "s":
            meta = {
                "stats": {
                    "total_rounds": round_no,
                    "player_wins": win,
                    "ai_wins": lose,
                    "draws": draw,
                    "ai_winrate_pct": (lose / round_no * 100) if round_no > 0 else 0.0
                }
            }
            saved_path = save_boss_model(boss, metadata=meta)
            print(f"  💾 บันทึกโมเดลเรียบร้อย: {saved_path}")
            continue

        if raw not in num:
            print("  ! พิมพ์ 1, 2, 3 หรือ s, q")
            continue

        player_move = num[raw]
        boss_move = boss.choose(opp_history)     # บอสเลือกจากประวัติ "ก่อน" ตานี้
        result = judge(player_move, boss_move)

        # อัปเดตประวัติ + ให้บอสเรียนรู้
        opp_history.append(player_move)
        reward = {"win": -1, "draw": 0, "lose": 1}[result]  # มุมมองบอส
        boss.learn(reward, opp_history)

        round_no += 1
        if result == "win":
            win += 1;  tag = "คุณชนะ! ✅"
        elif result == "lose":
            lose += 1; tag = "คุณแพ้ ❌"
        else:
            draw += 1; tag = "เสมอ ➖"

        wr = win / round_no * 100
        ai_wr = lose / round_no * 100
        print(f"    คุณออก {player_move} | บอสออก {boss_move} -> {tag}"
              f"   (คุณชนะ {win}/{round_no} = {wr:.0f}% | บอสชนะ {ai_wr:.0f}%)")

    # สรุปผล
    print("\n===== สรุปผล =====")
    if round_no == 0:
        print("ยังไม่ได้เล่นเลย")
    else:
        ai_wr = lose / round_no * 100
        print(f"เล่นทั้งหมด {round_no} ตา | คุณชนะ {win} | AI บอสชนะ {lose} | เสมอ {draw}")
        print(f"อัตราชนะของ AI บอส = {ai_wr:.1f}%  (อัตราชนะของคุณ = {win / round_no * 100:.1f}%)")

        # เสนอตัวเลือกให้ส่งออกโมเดล (Export Model)
        save_opt = input("\n💾 ต้องการส่งออก (Export) โมเดล AI บอสตัวนี้ไว้ใช้หรือไม่? (y/n): ").strip().lower()
        if save_opt == "y":
            model_name_input = input("ตั้งชื่อโมเดล (เว้นว่างเพื่อใช้ชื่ออัตโนมัติ): ").strip()
            meta = {
                "custom_name": model_name_input if model_name_input else f"บอสแก้ทางผู้เล่น (Winrate {ai_wr:.0f}%)",
                "stats": {
                    "total_rounds": round_no,
                    "player_wins": win,
                    "ai_wins": lose,
                    "draws": draw,
                    "ai_winrate_pct": ai_wr
                }
            }
            saved_path = save_boss_model(boss, metadata=meta)
            print(f"✅ ส่งออกโมเดลสำเร็จแล้ว! บันทึกอยู่ที่: {saved_path}")
            print(f"คุณสามารถนำไฟล์นี้ไปโหลดในเกม webcam (main_game.py) ได้ทันที!")


def main():
    print("=========================================")
    print("    ADAPTIVE ARENA — บอสเรียนรู้ + โมเดล I/O")
    print("=========================================")
    while True:
        boss = pick_boss()
        if boss is None:
            print("บายย 👋")
            break
        play(boss)
        again = input("\nเล่นบอสตัวอื่นต่อมั้ย? (y/n): ").strip().lower()
        if again != "y":
            print("บายย 👋")
            break


if __name__ == "__main__":
    main()
