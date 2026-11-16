# 🎮 Read-a-Rock: เกมเป่ายิ้งฉุบ AI Arena & Data Science Model Acquisition Pipeline

โปรเจกต์เกมเป่ายิ้งฉุบแบบโต้ตอบผ่านกล้องเว็บแคม (Webcam) ที่ขับเคลื่อนด้วยการทำงานร่วมกันของ **Vision AI Model** และ **Decision/Behavioral AI Model** พร้อมระบบ **End-to-End Data Science Pipeline** สำหรับฝึกสอน คัดเลือก และ **Import / Export** โมเดล AI ที่เรียนรู้จับทางผู้เล่นออกมาใช้งานได้จริง!

---

## 🧠 การทำงานร่วมกันของระบบ AI

```
    [ กล้อง Webcam ] ──────────────────────────────────────────────┐
           │ (ภาพสด BGR)                                           │ (บันทึก Telemetry ทุกตา)
           ▼                                                       ▼
┌──────────────────────────────────────────────┐        ┌───────────────────────────────────┐
│  โมเดลที่ 1: MediaPipe Hands (Vision AI)       │        │  Data Science Pipeline            │
│  - ตรวจจับจุดพิกัดข้อต่อมือ 21 จุด (Landmarks)   │        │  (model_pipeline.py)              │
│  - นับจำนวนนิ้ว:                                │        │  - EDA & Chi-Square & Entropy     │
│      • 0 นิ้ว (กำมือ)      -> "ค้อน" ✊         │        │  - Feature Store & WSLS Lags      │
│      • 2 นิ้ว (ชี้ + กลาง) -> "กรรไกร" ✌️      │        │  - Walk-Forward Time-Series CV    │
│      • 5 นิ้ว (แบมือ)      -> "กระดาษ" ✋       │        │  - Benchmarking & Packaging       │
└──────────────────────────────────────────────┘        └───────────────────────────────────┘
           │                                                               │
           │ (ท่าที่ตรวจจับได้)                                             │ (Exported Model)
           ▼                                                               ▼
┌──────────────────────────────────────────────┐        ┌───────────────────────────────────┐
│  โมเดลที่ 2: Adaptive Arena (Decision AI)    │ ◄──────┤  [S] Export / [L] Import Model    │
│  - เลือกบอส AI ได้ 8 รูปแบบสมอง                │        │  - บันทึกสมองบอสที่แก้ทางเราได้     │
│  - โหลด Custom Saved Model ที่เทรนไว้มาเล่นได้   │        │  - โหลดกลับมาสู้ใหม่ได้ตลอดเวลา    │
│  - อัปเดตการเรียนรู้ผ่านรางวัล (Reward Feedback)   │        └───────────────────────────────────┘
└──────────────────────────────────────────────┘
           │
           ▼
    [ ตัดสินผลแพ้/ชนะ + แสดง HUD บนจอภาพสด ]
```

---

## 🤖 ระดับสมองของบอส AI (8 Algorithms ใน adaptive_arena.py)

| เลข | ชื่อโมเดล AI บอส | ระดับ | กลยุทธ์การทำงาน |
|:---:|:---|:---:|:---|
| **1** | **RandomBoss** | ง่าย | สุ่มออกท่า 1/3 เท่ากัน ไม่เรียนรู้ (ใช้เป็นเกณฑ์วัด Baseline ทฤษฎี Nash) |
| **2** | **FrequencyBoss** | ง่าย | นับความถี่ท่าที่ผู้เล่นออกบ่อยที่สุดตลอดทั้งเกม แล้วออกท่าแก้ทาง |
| **3** | **DecayBoss** | ง่าย | ให้น้ำหนักท่าที่เพิ่งออกล่าสุดมากกว่า (Recency Decay) ปรับตัวเมื่อผู้เล่นเปลี่ยนสไตล์ |
| **4** | **MarkovBoss** | ปานกลาง | First-Order Markov Chain: วิเคราะห์ความน่าจะเป็นแบบมีเงื่อนไข เช่น หลังออกค้อน มักจะตามด้วยอะไร |
| **5** | **NgramBoss** | ปานกลาง | วิเคราะห์ลำดับ $k$-gram ท่าล่าสุด เพื่อจับแพทเทิร์นที่ซับซ้อนขึ้น |
| **6** | **EnsembleBoss** | ขั้นสูง *(ค่าเริ่มต้น)* | รันโมเดล 2, 3, 4, 5 พร้อมกัน แล้วเลือกเชื่อโมเดลที่ทายถูกมากที่สุดในช่วงนั้น |
| **7** | **QLearningBoss** | ขั้นสูง | Reinforcement Learning (Q-Learning): เรียนรู้ค่า $Q(s, a)$ จากผลรางวัลแพ้ (+1), เสมอ (0), ชนะ (-1) |
| **8** | **MLPredictorBoss** | Data Science | สกัดฟีเจอร์ WSLS + Markov + Recency + Streak ปรับค่าน้ำหนักโมเดลแบบ Real-time |
| **L** | **CustomSavedBoss** | พิเศษ | โหลดโมเดลที่เซฟไว้จากโฟลเดอร์ `models/` มาเป็นคู่ต่อสู้ |

---

## 💾 โฟลวการเล่นเพื่อบันทึกข้อมูลและโหลดกลับมาสู้ต่อ (Gameplay & Persistence Flow)

คุณสามารถเก็บสถิติเพื่อฝึก AI ให้จำสไตล์การเล่นของคุณ และนำกลับมาสู้ใหม่ได้ **2 รูปแบบ**:

### 🎯 รูปแบบที่ 1: บันทึกสมองบอสแบบ Real-time (Online Learning Persistence)
*เหมาะสำหรับการเล่นต่อเนื่อง และต้องการบันทึกสถานะสมองของบอสตัวที่กำลังสู้ทันที*
1. **เริ่มเล่น:** รันเกมผ่านกล้อง `python main_game.py` หรือโหมดไว `python fast_mode.py`
2. **เลือกบอสที่มีระบบเรียนรู้:** เช่น บอส 6 (`EnsembleBoss`), บอส 7 (`QLearningBoss`), หรือบอส 8 (`MLPredictorBoss`)
3. **เล่นสะสมรอบ:** สู้ไปเรื่อยๆ สัก 20–50 ตา บอสจะเริ่มปรับน้ำหนักและเมทริกซ์ความน่าจะเป็นเพื่อแก้ทางคุณ
4. **บันทึกโมเดล (Export):** กดปุ่ม **`S`** บนคีย์บอร์ด ระบบจะเซฟสมองบอสลงโฟลเดอร์ `models/` เป็นไฟล์ JSON (เช่น `models/EnsembleBoss_xxx.json`)
5. **โหลดกลับมาสู้ในครั้งหน้า (Import):** เมื่อเปิดเกมใหม่ขึ้นมา ไม่ว่าจะวันไหน ให้กดปุ่ม **`L`** บนคีย์บอร์ด ระบบจะโหลดโมเดลตัวนี้กลับมาเป็นคู่ต่อสู้ทันที!

### 🔬 รูปแบบที่ 2: Data Science ML Pipeline (เทรนแชมเปียนโมเดลเฉพาะบุคคล)
*เหมาะสำหรับการวิเคราะห์เชิงลึก สกัดฟีเจอร์พฤติกรรมมนุษย์ (WSLS, Streak, Lags) ด้วย Machine Learning*

#### 👤 การระบุชื่อผู้เล่น (Player Profile):
ก่อนเริ่มเล่น ระบบจะถามชื่อผู้เล่น หรือสามารถกำหนดชื่อผ่าน Argument ได้ทันที:
```bash
# ระบุชื่อผู้เล่นในโหมดกล้อง Webcam
python main_game.py --player Ken

# ระบุชื่อผู้เล่นในโหมดไว Fast Arena
python fast_mode.py --player Ken
```
*(ใน `fast_mode.py` สามารถกดปุ่ม **`U`** เพื่อสลับชื่อผู้เล่นระหว่างเล่นได้ทันทีเมื่อเปลี่ยนคนเล่น)*

#### 🚀 การสั่งเทรนโมเดลแยกตามชื่อผู้เล่น:
เมื่อเพื่อนหลายคนเล่นในเครื่องเดียวกัน ข้อมูลจะถูกบันทึกแยกคอลัมน์ `player_name` ใน `data/gameplay_telemetry.csv` ทำให้สามารถสั่งเทรนโมเดลแก้ทางเฉพาะบุคคลได้:
```bash
# เทรนโมเดลแชมเปียนเฉพาะทางสำหรับแก้ทาง "Ken"
python model_pipeline.py --mode train --player Ken

# หรือดูสถิติพฤติกรรมและการออกท่าเฉพาะของ "Ken" (EDA Mode)
python model_pipeline.py --mode eda --player Ken
```
- **ผลลัพธ์ที่ได้:** ระบบจะสกัดเฉพาะตาที่ Ken เล่นมาวิเคราะห์ และ Export โมเดลออกมาเป็น `models/ds_champion_ken.json`
- **รอบหน้าเข้าเกม:** กดปุ่ม **`L`** เพื่อโหลดบอส `🏆 DS Champion ปราบ Ken` มาสู้ได้ทันที!

---

## 🔬 Data Science Model Acquisition Pipeline (`model_pipeline.py`)

ไปป์ไลน์การได้มาซึ่งโมเดลตามหลัก Data Science และ Machine Learning ตั้งแต่ต้นจนจบ:

### ขั้นตอนการทำงานของ Pipeline (7 Stages):
1. **Domain Formulation (ทฤษฎีเกม & จิตวิทยา):** วิเคราะห์อคติของผู้เล่นมนุษย์ (Non-Nash) เช่น Win-Stay, Lose-Shift (WSLS งานวิจัย Nature 2014) และ Gambler's Fallacy
2. **Data Ingestion & Telemetry Logging:** ทุกการเล่นใน [main_game.py](main_game.py) จะบันทึกผลลง `data/gameplay_telemetry.csv` อัตโนมัติ (หรือใช้ Simulator จำลองพฤติกรรมมนุษย์)
3. **Exploratory Data Analysis (EDA):**
   - **Chi-Square Goodness-of-Fit:** ทดสอบนัยสำคัญทางสถิติว่าผู้เล่นเล่นสุ่มจริงหรือไม่ ($p < 0.05$)
   - **Shannon Entropy ($H(X)$):** วัดระดับความไม่สามารถคาดเดาได้ (เปรียบเทียบกับขีดจำกัด $\log_2(3) \approx 1.585$ bits)
   - **WSLS Empirical Transition Rate:** คำนวณอัตราการออกท่าเดิมเมื่อชนะ และอัตราการเปลี่ยนท่าเมื่อแพ้
4. **Feature Engineering & Feature Store:**
   - Sequential Lags ($t-1, t-2, t-3$)
   - ผลแพ้ชนะและท่าของบอสในตาก่อนหน้า
   - ปฏิสัมพันธ์เชิงจิตวิทยา (WSLS Interaction token)
   - Rolling Window Move Distribution (ความถี่ใน 5 ตาล่าสุด)
   - Consecutive Move Streak Length
5. **Model Training & Time-Series Cross Validation:**
   - ใช้ **Walk-Forward Expanding Window Validation** (ไม่มี Data Leakage จากการสุ่ม shuffle ข้อมูลเวลา)
   - ทดสอบและเปรียบเทียบโมเดล: `LogisticRegression`, `RandomForest`, `HistGradientBoosting`
6. **Game-Theoretic Benchmarking:**
   - ประเมินผลด้วย **Multi-class Log-Loss**, **Accuracy**, **Expected Value $E[R]$**, และ **Efficiency Score (%)** เทียบกับ Nash Baseline
7. **Model Packaging & Automated Reporting:**
   - ส่งออกโมเดลแชมเปียนพร้อม Model Card เมทาดาทาครบถ้วนไปที่ `models/ds_champion_boss.json`
   - สร้างรายงานสรุปผลการทดลองที่ `reports/pipeline_report.md`

### คำสั่งรัน Pipeline:

```bash
# รัน Pipeline ครบวงจรตั้งแต่ต้นจนได้โมเดลแชมเปียน
python model_pipeline.py --mode full --rounds 1500

# รันเฉพาะการวิเคราะห์พฤติกรรมผู้เล่น (EDA Mode)
python model_pipeline.py --mode eda

# รันเทรนจากไฟล์ Telemetry การเล่นจริง
python model_pipeline.py --mode train --input data/gameplay_telemetry.csv
```

---

## ⌨️ ปุ่มควบคุมและคีย์ลัดในเกม Webcam

| ปุ่ม | คำสั่ง | รายละเอียด |
|:---:|:---|:---|
| `SPACE` | **เริ่มรอบใหม่** | เริ่มนับถอยหลัง 3.. 2.. 1.. เป่ายิ้งฉุบ! |
| `A` | **เปิด/ปิดโหมดอัตโนมัติ (AUTO)** | เล่นต่อเนื่องเรื่อย ๆ โดยเว้นระยะพักหลังแต่ละรอบ |
| `T` | **สลับความเร็วรอบ (Speed/Pace)** | ปรับเวลานับถอยหลังต่อตา (3s ➔ 2s ➔ 1.5s ➔ 1s) เพื่อเล่นให้ไวยิ่งขึ้น |
| `1` - `8` | **เปลี่ยนสมองบอส AI** | สลับไปเจอบอสหมายเลข 1 ถึง 8 ได้ทันทีระหว่างเล่น |
| `S` | **บันทึกโมเดล (Export)** | เซฟสมองบอสตัวปัจจุบันที่กำลังแก้ทางเราลงไฟล์ JSON |
| `L` | **โหลดโมเดล (Import)** | โหลดโมเดลที่บันทึกไว้หรือแชมเปียนโมเดลเข้ามาสู้ |
| `B` | **เปิดเมนูบอส** | เปิดหน้าต่างแสดงรายละเอียดและความสามารถของบอสทั้ง 8 ตัว |
| `R` | **รีเซ็ตคะแนน** | ล้างคะแนน Win/Loss และรีเซ็ตความจำสมองบอส |
| `M` | **ปิด/เปิดเสียง** | สลับเสียงเอฟเฟกต์นับถอยหลังและเสียงชนะ/แพ้ |
| `H` | **ซ่อน/แสดงโครงสร้างมือ** | สลับการวาดเส้นกระดูกมือ MediaPipe บนหน้าจอ |
| `Q` หรือ `ESC` | **ออกจากเกม** | ปิดหน้าต่างและคืนทรัพยากรกล้อง |

---

## 📦 การติดตั้งและเริ่มต้นใช้งาน (Quick Start)

### 1. โคลน Repository
```bash
git clone https://github.com/kennerskk/Read-a-Rock.git
cd Read-a-Rock
```

### 2. สร้าง Virtual Environment (แนะนำ)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 วิธีการรันเกมและระบบต่างๆ

```bash
# 1. รันเกมเป่ายิ้งฉุบผ่านกล้องเว็บแคม (โหมดมาตรฐาน 3 วินาที หรือกด [T] ในเกมเพื่อเร่งความเร็ว)
python main_game.py

# กำหนดเวลานับถอยหลังตั้งแต่เริ่มรัน (เช่น 1.5 วินาทีต่อตา เพื่อความรวดเร็ว)
python main_game.py --countdown 1.5

# 2. ⚡ รันโหมดไวพิเศษ (Fast Arena) — ไม่ต้องเปิดกล้อง + ปรับเวลาได้อิสระ
python fast_mode.py

# รัน Fast Arena พร้อมกำหนดเวลาต่อตา (เช่น 0.5 วินาที หรือ 0 วินาที = Instant รู้ผลทันที)
python fast_mode.py --pace 0.5 --boss 6

# 3. รันโหมดจำลองประลองผ่าน Terminal
python adaptive_arena.py

# 4. รันไปป์ไลน์ Data Science เพื่อเทรนโมเดลใหม่จากข้อมูลการเล่น
python model_pipeline.py --mode full

# 5. ทดสอบประสิทธิภาพ Efficiency Score เทียบกับโมเดลสุ่ม
python test_efficiency.py
```

---

## 📂 โครงสร้างไฟล์ในโปรเจกต์

- [main_game.py](main_game.py) — สคริปต์หลักเล่นผ่านกล้องเว็บแคม (รองรับปุ่ม `[T]` ปรับความเร็ว 1.0s - 3.0s)
- [fast_mode.py](fast_mode.py) — ⚡ โหมดประลองด่วน (No-Cam) เลือกท่าผ่านคีย์บอร์ด ปรับเวลาต่อตาได้อิสระ บันทึก Telemetry ครบถ้วน
- [model_pipeline.py](model_pipeline.py) — End-to-End Data Science & Machine Learning Pipeline
- [model_io.py](model_io.py) — ระบบบันทึก (Save/Export) และโหลด (Load/Import) โมเดลพร้อม Model Card
- [adaptive_arena.py](adaptive_arena.py) — โมเดล AI สมองบอส 8 รูปแบบ และระบบจำลองการแข่ง
- [hand_detector.py](hand_detector.py) — โมเดล Vision ตรวจจับพิกัดมือ 21 จุดและแยกแยะนิ้วด้วย MediaPipe
- [ui_renderer.py](ui_renderer.py) — ระบบวาด HUD แดชบอร์ด การ์ดผู้เล่น/บอส และ Toast แจ้งเตือนภาษาไทย
- [sound_effects.py](sound_effects.py) — ระบบเสียงเอฟเฟกต์แบบ Non-blocking (พร้อม Cross-platform fallback)
- [test_efficiency.py](test_efficiency.py) — คำนวณค่า Efficiency Score (%) เทียบกับ Nash Baseline
- [evaluate_ai.py](evaluate_ai.py) — ทดสอบประสิทธิภาพ AI กับ Persona รูปแบบต่างๆ
- `data/` — โฟลเดอร์เก็บข้อมูล Telemetry สดจากการเล่น (`gameplay_telemetry.csv`)
- `models/` — โฟลเดอร์เก็บโมเดล AI ที่ถูก Export ออกมา (`.json`)
- `reports/` — โฟลเดอร์เก็บรายงานผลการวิเคราะห์ Data Science Pipeline (`.md`)

---

## 🛠️ แนวทางการพัฒนาต่อยอดสำหรับผู้ร่วมพัฒนา (Developer Guide)

### 1. เพิ่มสมองบอส AI ตัวใหม่ (`adaptive_arena.py`)
คุณสามารถสร้างกลยุทธ์บอสตัวใหม่ได้ง่ายๆ โดยการสืบทอด (Inherit) จากคลาส `BaseBoss`:
```python
class MyNewBoss(BaseBoss):
    def __init__(self):
        super().__init__(name="MyNewBoss", level="ปานกลาง", description="คำอธิบายกลยุทธ์")
        # ประกาศ State หรือตัวแปรจดจำที่นี่

    def predict_player_move(self):
        # คืนค่าท่าที่คาดว่าผู้เล่นจะออก: 'rock', 'paper', หรือ 'scissors'
        return "rock"

    def record_round(self, player_move, ai_move, outcome):
        # อัปเดตการเรียนรู้หลังจบตา
        pass
```
จากนั้นลงทะเบียนในดิกชันนารี `BOSS_REGISTRY` หรือเพิ่มปุ่มลัดใน `main_game.py` ได้ทันที

### 2. ปรับแต่ง Feature Store ใน Data Science Pipeline (`model_pipeline.py`)
- เข้าไปที่ฟังก์ชัน `extract_features(df)` เพื่อเพิ่ม Lag Features, Moving Averages, หรือ Interaction Terms ใหม่ๆ
- เพิ่ม Classifier ตัวอื่น เช่น `XGBoost`, `CatBoost`, หรือ `LightGBM` ในขั้นตอนโมเดลเทรนนิ่ง

### 3. ปรับแต่งหน้าตา HUD และกราฟิก (`ui_renderer.py`)
- ปรับเปลี่ยนโทนสี, ไอคอน, สไตล์การ์ด และเลย์เอาต์ตำแหน่งแดชบอร์ดในคลาส `UIRenderer`
- รองรับการเรนเดอร์ฟอนต์ภาษาไทยและ Multi-platform font fallback อัตโนมัติ
