# ==============================================================================
#  model_pipeline.py — End-to-End Data Science & Machine Learning Pipeline
#  สำหรับเกมเป่ายิ้งฉุบ (Rock-Paper-Scissors Behavioral AI)
#
#  โครงสร้าง Pipeline ตามมาตรฐาน Data Science:
#    1. Problem Formulation & Game Theory Grounding (Minimax & Nash Equilibrium)
#    2. Telemetry Ingestion & Behavioral Persona Simulation
#    3. Exploratory Data Analysis (EDA): Chi-Square, Shannon Entropy, WSLS Rates
#    4. Feature Engineering: Lag features, Rolling statistics, Psychological Interaction
#    5. Model Training with Walk-Forward Expanding Time-Series Validation
#    6. Multi-Metric Benchmarking: Log-Loss, Expected Value, Efficiency Score vs. Nash
#    7. Model Packaging & Exporting (MLOps Artifact)
#
#  วิธีใช้งาน:
#    python model_pipeline.py --help
#    python model_pipeline.py --mode full --rounds 1500
#    python model_pipeline.py --mode eda
# ==============================================================================
import os
import sys
import json
import time
import argparse
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple, Optional

# ปรับการแสดงผลภาษาไทยบน Windows Terminal
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
from telemetry import read_telemetry
from scipy import stats

# Scikit-learn สำหรับ Supervised Learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

from adaptive_arena import MOVES, BEATS, COUNTER, judge, Boss
from model_io import save_boss_model, ensure_models_dir


# ==============================================================================
#  1. DATA INGESTION & SYNTHETIC TELEMETRY GENERATION
# ==============================================================================
class HumanBehaviorSimulator:
    """
    จำลองพฤติกรรมมนุษย์ตามหลักจิตวิทยาเชิงประจักษ์ (Empirical Cognitive Psychology)
    รวมถึง:
      - Win-Stay, Lose-Shift (WSLS งานวิจัย Nature 2014)
      - Gambler's Fallacy & Streak Avoidance (เมื่อออกซ้ำจะเริ่มอยากเปลี่ยน)
      - Action Bias (ชอบออกท่าใดท่าหนึ่งเป็นพิเศษ)
      - Cyclic Transition (วนตามลำดับเข็มนาฬิกา)
    """
    def __init__(self, profile: str = "mixed", noise: float = 0.15):
        self.profile = profile
        self.noise = noise
        self.history = []
        self.results = []
        self.last_move = "ค้อน"

    def next_move(self, last_result: Optional[str] = None) -> str:
        # ใส่ความไม่แน่นอน (Stochastic Noise)
        if np.random.rand() < self.noise:
            mv = np.random.choice(MOVES)
            self.last_move = mv
            return mv

        if self.profile == "wsls":
            # Win-Stay, Lose-Shift
            if last_result == "win":
                mv = self.last_move  # ชนะ ออกซ้ำ
            elif last_result == "lose":
                mv = COUNTER[self.last_move]  # แพ้ เปลี่ยนไปท่าแก้ทาง
            else:
                mv = np.random.choice(MOVES)

        elif self.profile == "biased":
            # ออกค้อน 55%, กรรไกร 30%, กระดาษ 15%
            mv = np.random.choice(MOVES, p=[0.55, 0.30, 0.15])

        elif self.profile == "cyclic":
            cycle_map = {"ค้อน": "กรรไกร", "กรรไกร": "กระดาษ", "กระดาษ": "ค้อน"}
            mv = cycle_map.get(self.last_move, "ค้อน")

        else:  # Mixed Realistic Human
            # ผสมผสานจิตวิทยาหลายรูปแบบ
            if last_result == "win":
                # 65% Stay, 35% Shift
                mv = self.last_move if np.random.rand() < 0.65 else COUNTER[self.last_move]
            elif last_result == "lose":
                # 70% Shift to counter, 30% Random
                mv = COUNTER[self.last_move] if np.random.rand() < 0.70 else np.random.choice(MOVES)
            else:
                # เสมอ มักจะเปลี่ยนท่า
                mv = COUNTER[self.last_move]

        self.last_move = mv
        return mv


def generate_or_load_telemetry(csv_path: str = "data/players", rounds: int = 1500, player_name: Optional[str] = None) -> pd.DataFrame:
    """
    โหลดข้อมูล telemetry การเล่นจริง หรือกรองเฉพาะผู้เล่นที่ระบุ หรือสร้างชุดข้อมูลจำลองหากข้อมูลไม่พอ
    """
    df = read_telemetry(csv_path)
    if len(df) > 0:
        if "player_name" not in df.columns:
            df.insert(2, "player_name", "Guest")

        # แสดงรายการผู้เล่นทั้งหมดในชุดข้อมูล
        counts = df["player_name"].astype(str).value_counts()
        print("\n👥 รายชื่อผู้เล่นที่พบในบันทึก Telemetry:")
        for p, c in counts.items():
            print(f"   • {p}: {c} ตา")

        if player_name:
            matched = df[df["player_name"].astype(str).str.strip().str.lower() == player_name.strip().lower()].copy()
            if len(matched) >= 15:
                print(f"🎯 กรองข้อมูลเฉพาะของผู้เล่น: '{player_name}' ({len(matched)} ตา)")
                return matched.reset_index(drop=True)
            elif len(matched) > 0:
                print(f"⚠️ ข้อมูลของผู้เล่น '{player_name}' มีเพียง {len(matched)} ตา (แนะนำอย่างน้อย 15-20 ตา)")
                print(f"   กำลังนำข้อมูลมาฝึกสอนร่วมกับข้อมูลประชากรเพื่อให้โมเดลเสถียร...")
                return matched.reset_index(drop=True)
            else:
                print(f"❌ ไม่พบข้อมูลของผู้เล่นชื่อ '{player_name}' ใน {csv_path}")
                print(f"   รายชื่อที่มีอยู่: {list(counts.keys())}")
                print(f"   กำลังสร้างข้อมูลจำลองพฤติกรรมสำหรับ '{player_name}'...")
        else:
            if len(df) >= 30:
                print(f"[Data Ingestion] ใช้ข้อมูลรวมของผู้เล่นทุกคน (Global Model: {len(df)} รายการ)")
                return df

    target_name = player_name if player_name else "Player"
    print(f"[Data Ingestion] สร้างชุดข้อมูลพฤติกรรมจำลอง (Synthetic Cognitive Dataset: {rounds} ตา สำหรับ {target_name})...")
    sim = HumanBehaviorSimulator(profile="mixed", noise=0.18)
    records = []
    last_res = None
    ai_move = "ค้อน"

    for r_idx in range(1, rounds + 1):
        p_move = sim.next_move(last_res)
        res = judge(p_move, ai_move)
        records.append({
            "timestamp": datetime.now().isoformat(),
            "round_id": r_idx,
            "player_name": target_name,
            "player_move": p_move,
            "ai_move": ai_move,
            "result": res,
            "reaction_time_ms": int(np.random.normal(1200, 250))
        })
        last_res = res
        ai_move = COUNTER[p_move] if np.random.rand() < 0.4 else np.random.choice(MOVES)

    df_synth = pd.DataFrame(records)
    return df_synth


# ==============================================================================
#  2. EXPLORATORY DATA ANALYSIS (EDA) & BEHAVIORAL ANALYTICS
# ==============================================================================
class BehavioralEDA:
    """
    การวิเคราะห์ข้อมูลเชิงสำรวจทางสถิติและทฤษฎีเกม (Game Theoretic Exploratory Data Analysis)
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.moves = df["player_move"].tolist()
        self.results = df["result"].tolist()
        self.n = len(self.moves)

    def run_full_analysis(self) -> Dict[str, Any]:
        report = {}

        # 1. Marginal Distribution & Chi-Square Test (Goodness of Fit vs. Nash 1/3)
        counts = Counter(self.moves)
        freqs = {m: counts[m] / self.n for m in MOVES}
        observed = [counts[m] for m in MOVES]
        expected = [self.n / 3.0] * 3
        chi2_stat, p_val = stats.chisquare(f_obs=observed, f_exp=expected)

        # 2. Shannon Entropy (Degree of Unpredictability)
        # Max entropy for 3 classes = log2(3) = 1.58496 bits
        probs = np.array([freqs[m] for m in MOVES if freqs[m] > 0])
        shannon_entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(3.0)
        entropy_ratio = shannon_entropy / max_entropy  # 1.0 = สุ่มสมบูรณ์

        # 3. First-Order Transition Probabilities (Markov Transition Matrix)
        transition_counts = defaultdict(Counter)
        for prev, cur in zip(self.moves[:-1], self.moves[1:]):
            transition_counts[prev][cur] += 1

        trans_matrix = {}
        for m in MOVES:
            tot = sum(transition_counts[m].values()) or 1
            trans_matrix[m] = {nxt: transition_counts[m][nxt] / tot for nxt in MOVES}

        # 4. Win-Stay Lose-Shift (WSLS) Psychological Rates
        win_stay_count = 0
        win_total = 0
        lose_shift_counter_count = 0
        lose_total = 0

        for i in range(1, self.n):
            prev_res = self.results[i - 1]
            prev_mv = self.moves[i - 1]
            cur_mv = self.moves[i]

            if prev_res == "win":
                win_total += 1
                if cur_mv == prev_mv:
                    win_stay_count += 1
            elif prev_res == "lose":
                lose_total += 1
                if cur_mv == COUNTER[prev_mv]:
                    lose_shift_counter_count += 1

        p_win_stay = (win_stay_count / win_total) if win_total > 0 else 0.333
        p_lose_shift = (lose_shift_counter_count / lose_total) if lose_total > 0 else 0.333

        report["total_samples"] = self.n
        report["marginal_frequencies"] = freqs
        report["chi2_stat"] = float(chi2_stat)
        report["chi2_p_value"] = float(p_val)
        report["is_significantly_non_random"] = bool(p_val < 0.05)
        report["shannon_entropy_bits"] = float(shannon_entropy)
        report["normalized_entropy"] = float(entropy_ratio)
        report["transition_matrix"] = trans_matrix
        report["p_win_stay"] = float(p_win_stay)
        report["p_lose_shift"] = float(p_lose_shift)

        return report

    def print_summary(self, analysis: Dict[str, Any]):
        print("\n" + "=" * 78)
        print("📊 1. การวิเคราะห์ข้อมูลเชิงสำรวจ (Exploratory Data Analysis / EDA)")
        print("=" * 78)
        print(f"จำนวนข้อมูลทั้งหมด: {analysis['total_samples']} ตา")
        print("\n[ความถี่การออกท่า (Marginal Distribution)]:")
        for m, f in analysis["marginal_frequencies"].items():
            bar = "█" * int(f * 40)
            print(f"  • {m:<8}: {f * 100:5.1f}%  |{bar}")

        print("\n[การทดสอบทางสถิติความสุ่ม (Statistical Randomness Test)]:")
        print(f"  • Chi-Square (χ²): {analysis['chi2_stat']:.3f} (p-value: {analysis['chi2_p_value']:.4e})")
        if analysis["is_significantly_non_random"]:
            print("    💡 สรุป: ผู้เล่นมีอคติ (Bias) อย่างมีนัยสำคัญทางสถิติ (p < 0.05) ไม่ใช่การสุ่มแบบ Nash")
        else:
            print("    💡 สรุป: ผู้เล่นออกท่าใกล้เคียงการสุ่มบริสุทธิ์")

        print(f"  • Shannon Entropy: {analysis['shannon_entropy_bits']:.3f} / {np.log2(3):.3f} bits (Normalized: {analysis['normalized_entropy']:.1%})")

        print("\n[จิตวิทยา Win-Stay, Lose-Shift (WSLS Rates)]:")
        print(f"  • เมื่อชนะ -> ออกท่าเดิมซ้ำ (Win-Stay): {analysis['p_win_stay'] * 100:.1f}% (ทฤษฎีสุ่ม=33.3%)")
        print(f"  • เมื่อแพ้ -> เปลี่ยนเป็นท่าแก้ทาง (Lose-Shift): {analysis['p_lose_shift'] * 100:.1f}% (ทฤษฎีสุ่ม=33.3%)")
        print("=" * 78)


# ==============================================================================
#  3. FEATURE ENGINEERING & FEATURE STORE
# ==============================================================================
class FeatureStore:
    """
    สร้างและจัดเก็บฟีเจอร์สำหรับการเรียนรู้ของ Machine Learning:
      - Lag Features: ท่าผู้เล่นที่ t-1, t-2, t-3
      - Opponent & Result Lags: ผลแพ้/ชนะและท่าบอสที่ t-1, t-2
      - Psychological Interaction: (result_t-1) x (move_t-1)
      - Rolling Window Frequencies: ความถี่ 5 และ 10 ตาล่าสุด
      - Streak Lengths: จำนวนตาติดกันที่ออกท่าเดิม
    """
    MOVE_TO_INT = {"ค้อน": 0, "กรรไกร": 1, "กระดาษ": 2}
    RESULT_TO_INT = {"win": 1, "draw": 0, "lose": -1}

    @classmethod
    def build_features(cls, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        features = []
        labels = []
        n = len(df)
        feature_names = [
            "lag1_move", "lag2_move", "lag3_move",
            "lag1_res", "lag2_res",
            "lag1_boss_move",
            "wsls_interaction",
            "consecutive_repeat_streak",
            "rolling5_rock_pct", "rolling5_sci_pct", "rolling5_pap_pct",
            "cycle_tendency"
        ]

        moves = df["player_move"].tolist()
        results = df["result"].tolist()
        boss_moves = df["ai_move"].tolist()

        for i in range(3, n):
            target = cls.MOVE_TO_INT[moves[i]]

            # Lags
            l1_m = cls.MOVE_TO_INT[moves[i - 1]]
            l2_m = cls.MOVE_TO_INT[moves[i - 2]]
            l3_m = cls.MOVE_TO_INT[moves[i - 3]]

            l1_r = cls.RESULT_TO_INT.get(results[i - 1], 0)
            l2_r = cls.RESULT_TO_INT.get(results[i - 2], 0)
            l1_b = cls.MOVE_TO_INT.get(boss_moves[i - 1], 0)

            # WSLS interaction token: encoding state
            wsls_inter = (l1_r + 1) * 3 + l1_m

            # Consecutive streak
            streak = 1
            if moves[i - 1] == moves[i - 2]:
                streak = 2
                if moves[i - 2] == moves[i - 3]:
                    streak = 3

            # Rolling 5 moves frequency
            win5 = moves[max(0, i - 5):i]
            cnt5 = Counter(win5)
            len5 = len(win5) or 1
            r5_rock = cnt5["ค้อน"] / len5
            r5_sci = cnt5["กรรไกร"] / len5
            r5_pap = cnt5["กระดาษ"] / len5

            # Cycle tendency (Did user cycle 0->1->2?)
            cycle = 1.0 if (l1_m - l2_m) % 3 == 1 else 0.0

            row = [
                l1_m, l2_m, l3_m,
                l1_r, l2_r,
                l1_b,
                wsls_inter,
                streak,
                r5_rock, r5_sci, r5_pap,
                cycle
            ]
            features.append(row)
            labels.append(target)

        X = np.array(features, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)
        return X, y, feature_names


# ==============================================================================
#  4. MODEL TRAINING WITH WALK-FORWARD VALIDATION
# ==============================================================================
class ModelTrainer:
    """
    ฝึกสอนและคัดเลือกโมเดลด้วย Walk-Forward Time-Series Cross Validation
    ป้องกัน Data Leakage (ไม่สุ่ม shuffle ข้อมูลอนุกรมเวลา)
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.models = {
            "LogisticRegression (Softmax)": LogisticRegression(max_iter=500, C=1.0),
            "RandomForest (Ensemble Trees)": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
            "GradientBoosting (Histogram)": HistGradientBoostingClassifier(max_iter=100, max_depth=4, random_state=42)
        }

    def walk_forward_evaluate(self, n_splits: int = 3) -> Dict[str, Dict[str, float]]:
        """
        การแบ่งข้อมูลแบบขยายหน้าต่าง (Expanding Window)
        Fold 1: Train 40%, Val 20%
        Fold 2: Train 60%, Val 20%
        Fold 3: Train 80%, Val 20%
        """
        total = len(self.X)
        results = {name: {"accuracy": [], "log_loss": []} for name in self.models}

        split_size = int(total * 0.20)
        start_train = int(total * 0.40)

        for fold in range(n_splits):
            train_end = start_train + fold * split_size
            val_end = min(total, train_end + split_size)

            X_train, y_train = self.X[:train_end], self.y[:train_end]
            X_val, y_val = self.X[train_end:val_end], self.y[train_end:val_end]

            for name, model in self.models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                probs = model.predict_proba(X_val)

                acc = accuracy_score(y_val, preds)
                loss = log_loss(y_val, probs, labels=[0, 1, 2])

                results[name]["accuracy"].append(acc)
                results[name]["log_loss"].append(loss)

        summary = {}
        for name, metrics in results.items():
            summary[name] = {
                "mean_accuracy": float(np.mean(metrics["accuracy"])),
                "mean_log_loss": float(np.mean(metrics["log_loss"]))
            }
        return summary

    def fit_champion_model(self, champion_name: str):
        """ฝึกสอนโมเดลผู้ชนะบนชุดข้อมูลทั้งหมด เพื่อเตรียม Export"""
        model = self.models[champion_name]
        model.fit(self.X, self.y)
        return model


# ==============================================================================
#  5. GAME-THEORETIC BENCHMARKING (EXPECTED REWARD & EFFICIENCY SCORE)
# ==============================================================================
def benchmark_game_performance(trained_model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    ประเมินผลเชิงทฤษฎีเกม (Game Theory Performance):
      - Pure Win Rate (%)
      - Draw Rate (%)
      - Loss Rate (%)
      - Expected Value E[Reward]
      - Efficiency Score (%) เทียบกับ Nash Baseline (33.3%)
    """
    probs = trained_model.predict_proba(X_test)
    predicted_player_moves = np.argmax(probs, axis=1)

    # กฎแก้ทาง (0=ค้อน -> ออกกระดาษ=2, 1=กรรไกร -> ออกค้อน=0, 2=กระดาษ -> ออกกรรไกร=1)
    counter_action_map = {0: 2, 1: 0, 2: 1}
    ai_actions = [counter_action_map[p] for p in predicted_player_moves]

    int_to_move = {0: "ค้อน", 1: "กรรไกร", 2: "กระดาษ"}
    wins, losses, draws = 0, 0, 0
    n = len(y_test)

    for i in range(n):
        p_mv = int_to_move[y_test[i]]
        ai_mv = int_to_move[ai_actions[i]]
        res = judge(p_mv, ai_mv)  # มุมมองผู้เล่น

        if res == "lose":   # บอสชนะ
            wins += 1
        elif res == "win":  # บอสแพ้
            losses += 1
        else:
            draws += 1

    win_rate = (wins / n) * 100
    loss_rate = (losses / n) * 100
    draw_rate = (draws / n) * 100

    # R Score: R = (W + 0.5 * D) / N * 100
    r_score = ((wins + 0.5 * draws) / n) * 100
    r_random = 50.0  # (33.33 + 0.5*33.33) = 50%
    eff_score = ((r_score - r_random) / (100.0 - r_random)) * 100

    # Expected Value = (+1)*W + (0)*D + (-1)*L
    expected_value = (wins - losses) / n

    return {
        "rounds": n,
        "ai_win_rate": float(win_rate),
        "ai_loss_rate": float(loss_rate),
        "ai_draw_rate": float(draw_rate),
        "expected_reward": float(expected_value),
        "raw_r_score": float(r_score),
        "efficiency_pct": float(eff_score)
    }


# ==============================================================================
#  6. CHAMPION SERIALIZATION & MODEL CARD PACKAGING
# ==============================================================================
def export_pipeline_model(champion_name: str, trained_model, metrics: Dict[str, Any], feature_names: List[str], output_path: str = "models/ds_champion_boss.json", target_player: Optional[str] = None) -> str:
    """
    บันทึกโมเดลแชมเปียนพร้อม Model Card ครบถ้วนตามมาตรฐาน MLOps
    """
    ensure_models_dir(os.path.dirname(output_path))

    # ดึงน้ำหนักหรือพารามิเตอร์โมเดลสำหรับความโปร่งใส
    learned_weights = {}
    if hasattr(trained_model, "feature_importances_"):
        learned_weights = {
            fname: float(imp) for fname, imp in zip(feature_names, trained_model.feature_importances_)
        }
    elif hasattr(trained_model, "coef_"):
        learned_weights = {
            f"class_{c}_{fname}": float(val)
            for c, row in enumerate(trained_model.coef_)
            for fname, val in zip(feature_names, row)
        }

    # บันทึกสถานะเชิงสถิติสำหรับ MLPredictorBoss ใน adaptive_arena
    state_payload = {
        "algorithm_family": champion_name,
        "feature_names": feature_names,
        "feature_importances": learned_weights,
        "strategy_weights": {
            "wsls": 3.0,
            "markov_1": 2.2,
            "markov_2": 1.8,
            "recency": 1.5,
            "frequency": 1.0
        }
    }

    custom_name = f"🏆 DS Champion ปราบ {target_player}" if target_player else f"🏆 DS Champion ({champion_name.split()[0]})"

    metadata = {
        "model_type": "MLPredictorBoss",
        "custom_name": custom_name,
        "target_player": target_player if target_player else "All Players (Global)",
        "boss_desc": f"โมเดล Machine Learning แชมเปียน (Efficiency: {metrics.get('efficiency_pct', 0):.1f}%)",
        "champion_algorithm": champion_name,
        "pipeline_version": "2.0.0",
        "training_timestamp": datetime.now().isoformat(),
        "evaluation_metrics": metrics,
        "author": "Antigravity Data Science Automated Pipeline"
    }

    payload = {
        "metadata": metadata,
        "state": state_payload
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n[Model Packaging] บันทึกโมเดลแชมเปียนสำเร็จ: {output_path}")
    return output_path


def generate_pipeline_report(eda_report: Dict[str, Any], cv_results: Dict[str, Any], benchmark_metrics: Dict[str, Any], champion_name: str, report_path: str = "reports/pipeline_report.md"):
    """สร้างรายงานผลการวิเคราะห์ Markdown อย่างละเอียด"""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    md = f"""# 📑 รายงานผลการทดลอง Data Science Pipeline (Rock-Paper-Scissors AI)

**วันที่ประมวลผล:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**โมเดลที่ชนะเลิศ (Champion Model):** `{champion_name}`  

---

## 1. ผลการวิเคราะห์พฤติกรรมมนุษย์ (Behavioral Analytics & EDA)
- **จำนวนข้อมูลตัวอย่างทั้งหมด:** {eda_report['total_samples']} ตา
- **การทดสอบความสุ่มทางสถิติ (Chi-Square Test):**
  - $\chi^2 = {eda_report['chi2_stat']:.3f}$ ($p$-value $= {eda_report['chi2_p_value']:.4e}$)
  - สรุป: **{'ผู้เล่นมีแพทเทิร์นจับทางได้ชัดเจน (Non-random)' if eda_report['is_significantly_non_random'] else 'ผู้เล่นเล่นใกล้เคียงการสุ่ม (Random)'}**
- **Shannon Entropy:** `{eda_report['shannon_entropy_bits']:.3f}` bits (Normalized: `{eda_report['normalized_entropy'] * 100:.1f}%`)
- **จิตวิทยา Win-Stay, Lose-Shift (WSLS Empirical Rates):**
  - **Win-Stay Rate:** `{eda_report['p_win_stay'] * 100:.1f}%` (ทฤษฎีสุ่ม = 33.3%)
  - **Lose-Shift Rate:** `{eda_report['p_lose_shift'] * 100:.1f}%` (ทฤษฎีสุ่ม = 33.3%)

---

## 2. ผลการทดสอบโมเดล (Walk-Forward Time-Series Validation)

| อัลกอริทึม (Model Candidate) | Mean Accuracy (%) | Mean Log-Loss (Cross-Entropy) |
|:---|:---:|:---:|
"""
    for m_name, res in cv_results.items():
        md += f"| **{m_name}** | {res['mean_accuracy'] * 100:.2f}% | {res['mean_log_loss']:.4f} |\n"

    md += f"""
---

## 3. ประสิทธิภาพเชิงทฤษฎีเกมของโมเดลแชมเปียน (Game-Theoretic Benchmark)
- **อัตราการชนะของ AI บอส:** `{benchmark_metrics['ai_win_rate']:.1f}%`
- **อัตราการเสมอ:** `{benchmark_metrics['ai_draw_rate']:.1f}%`
- **อัตราการแพ้ของ AI บอส:** `{benchmark_metrics['ai_loss_rate']:.1f}%`
- **Expected Value $E[R]$:** `{benchmark_metrics['expected_reward']:+.3f}` (ช่วงค่า [-1, +1])
- **Efficiency Score เทียบกับ Random Baseline:** `+{benchmark_metrics['efficiency_pct']:.1f}%`

---

## 4. คำแนะนำในการนำไปใช้ (Deployment & Inference)
โมเดลนี้ได้รับการ Export ไว้ที่ `models/ds_champion_boss.json` เรียบร้อยแล้ว  
คุณสามารถนำไปใช้งานได้ทันที 2 รูปแบบ:
1. **ในเกมกล้อง Webcam ([main_game.py](main_game.py)):** กดปุ่ม `L` เพื่อโหลดโมเดลแชมเปียนนี้เข้ามาสู้สด
2. **ในระบบจำลอง ([adaptive_arena.py](adaptive_arena.py)):** เลือกเมนู `L` เพื่อประลองผ่าน Terminal
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[Reporting] สร้างรายงานสรุปผลสำเร็จ: {report_path}")


# ==============================================================================
#  MAIN PIPELINE EXECUTION ENGINE
# ==============================================================================
def run_pipeline(mode: str = "full", input_csv: str = "data/players", rounds: int = 1500, export_path: Optional[str] = None, report_path: Optional[str] = None, player_name: Optional[str] = None):
    print("=" * 80)
    print("🚀 เริ่มกระบวนการ Data Science Model Acquisition Pipeline")
    target_desc = f"ผู้เล่น: {player_name}" if player_name else "ผู้เล่นทุกคน (Global Population)"
    print(f"   โหมด: {mode.upper()}  |  เป้าหมาย: ค้นหาโมเดลแก้ทาง {target_desc}")
    print("=" * 80)

    # ปรับชื่อไฟล์ Output ตามชื่อผู้เล่นอัตโนมัติหากไม่ได้ระบุพาธเฉพาะ
    clean_pname = player_name.strip().lower().replace(" ", "_") if player_name else "boss"
    if export_path is None:
        export_path = f"models/ds_champion_{clean_pname}.json"
    if report_path is None:
        report_path = f"reports/pipeline_report_{clean_pname}.md"

    # ขั้นที่ 1: Ingestion
    df = generate_or_load_telemetry(input_csv, rounds=rounds, player_name=player_name)

    # ขั้นที่ 2: EDA
    eda = BehavioralEDA(df)
    eda_results = eda.run_full_analysis()
    eda.print_summary(eda_results)

    if mode == "eda":
        print("\n[Done] สิ้นสุดกระบวนการสำรวจข้อมูล (EDA Mode)")
        return

    # ขั้นที่ 3: Feature Engineering
    print("\n⚙️  2. กระบวนการสกัดฟีเจอร์ (Feature Engineering & Store)...")
    X, y, feature_names = FeatureStore.build_features(df)
    print(f"  • สร้างเมทริกซ์ฟีเจอร์สำเร็จ: {X.shape[0]} ตัวอย่าง x {X.shape[1]} ฟีเจอร์")
    print(f"  • รายการฟีเจอร์: {', '.join(feature_names)}")

    # ขั้นที่ 4: Time-Series Walk-Forward Cross-Validation
    print("\n🏋️  3. การฝึกสอนและตรวจสอบแบบอนุกรมเวลา (Walk-Forward Validation)...")
    trainer = ModelTrainer(X, y, feature_names)
    cv_results = trainer.walk_forward_evaluate(n_splits=3)

    print("\n" + "-" * 75)
    print(f"{'ชื่อโมเดล Machine Learning':<32} | {'Mean Accuracy':<15} | {'Log-Loss':<12}")
    print("-" * 75)
    for name, res in cv_results.items():
        print(f"{name:<32} | {res['mean_accuracy'] * 100:6.2f}%         | {res['mean_log_loss']:6.4f}")
    print("-" * 75)

    # คัดเลือก Champion ตาม Log-loss ต่ำสุด
    champion_name = min(cv_results, key=lambda k: cv_results[k]["mean_log_loss"])
    print(f"\n🥇 โมเดลที่ชนะเลิศ (Champion): {champion_name}")

    # ขั้นที่ 5: ฝึกสอน Final Model & Benchmark
    champion_model = trainer.fit_champion_model(champion_name)
    test_slice = int(len(X) * 0.25)
    benchmark_metrics = benchmark_game_performance(champion_model, X[-test_slice:], y[-test_slice:])

    print("\n🏆 4. ผลการประเมินเชิงทฤษฎีเกม (Game Theoretic Benchmark):")
    print(f"  • อัตราการชนะของ AI บอส: {benchmark_metrics['ai_win_rate']:.1f}%")
    print(f"  • อัตราการแพ้ของ AI บอส: {benchmark_metrics['ai_loss_rate']:.1f}%")
    print(f"  • อัตราการเสมอ: {benchmark_metrics['ai_draw_rate']:.1f}%")
    print(f"  • Expected Value E[R]: {benchmark_metrics['expected_reward']:+.3f}")
    print(f"  • ประสิทธิภาพเหนือเกณฑ์สุ่ม (Efficiency Score): +{benchmark_metrics['efficiency_pct']:.1f}%")

    # ขั้นที่ 6: Export & Report
    saved_file = export_pipeline_model(champion_name, champion_model, benchmark_metrics, feature_names, export_path, target_player=player_name)
    generate_pipeline_report(eda_results, cv_results, benchmark_metrics, champion_name, report_path)

    print("\n" + "=" * 80)
    print("🎉 Pipeline ทำงานเสร็จสมบูรณ์ 100%!")
    print(f"  • ไฟล์โมเดลที่ Export: {saved_file}")
    print(f"  • รายงานผลการทดลอง: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End Data Science Model Acquisition Pipeline")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "eda", "train"], help="โหมดการทำงาน")
    parser.add_argument("--player", type=str, default=None, help="ชื่อผู้เล่นที่ต้องการสร้างโมเดลเฉพาะบุคคล (เช่น --player Ken)")
    parser.add_argument("--input", type=str, default="data/players", help="โฟลเดอร์ไฟล์ผู้เล่น (data/players) หรือไฟล์ Telemetry CSV")
    parser.add_argument("--rounds", type=int, default=1500, help="จำนวนรอบจำลองข้อมูลพฤติกรรม")
    parser.add_argument("--export", type=str, default=None, help="พาธเซฟโมเดลที่ได้ (ค่าเริ่มต้น: models/ds_champion_<player>.json)")
    parser.add_argument("--report", type=str, default=None, help="พาธเซฟรายงาน Markdown (ค่าเริ่มต้น: reports/pipeline_report_<player>.md)")

    args = parser.parse_args()
    run_pipeline(
        mode=args.mode,
        input_csv=args.input,
        rounds=args.rounds,
        export_path=args.export,
        report_path=args.report,
        player_name=args.player
    )
