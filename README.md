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
│  - เลือกบอส AI ได้ 4 แบบ (คัดตัวเก่งจริง)       │        │  - บันทึกสมองบอสที่แก้ทางเราได้     │
│  - โหลด Custom Saved Model ที่เทรนไว้มาเล่นได้   │        │  - โหลดกลับมาสู้ใหม่ได้ตลอดเวลา    │
│  - อัปเดตการเรียนรู้ผ่านรางวัล (Reward Feedback)   │        └───────────────────────────────────┘
└──────────────────────────────────────────────┘
           │
           ▼
    [ ตัดสินผลแพ้/ชนะ + แสดง HUD บนจอภาพสด ]
```

---

## 🤖 บอส AI ในเกม (เหลือเฉพาะตัวที่เก่งจริงจาก benchmark)

| ปุ่ม | บอส | Net Score* | กลยุทธ์การทำงาน |
|:---:|:---|:---:|:---|
| **1** | **Markov** | 37.7 | First-Order Markov Chain: หลังออกท่า X ผู้เล่นมักตามด้วยท่าไหน |
| **2** | **N-gram** | 40.7 | มองลำดับ $k$ ท่าล่าสุดเป็นสถานะ จับแพทเทิร์นที่ซับซ้อนกว่า Markov |
| **3** | **Data Science ML Boss** | 39.6 | สกัดฟีเจอร์ WSLS + Markov + Recency + Frequency ปรับน้ำหนักแบบ Real-time |
| **4** | **Lite Ensemble** *(ค่าเริ่มต้น)* | **42.6** | รัน N-gram + ML พร้อมกัน แล้วเชื่อตัวที่ทายแม่นกว่าช่วงนั้น — ชุดที่ชนะการทดสอบทุกชุดค่าผสม ([reports/ensemble_combos.md](reports/ensemble_combos.md)) กด 4 ซ้ำเพื่อสลับเป็นชุด 3 ตัว / 4 ตัว / **Warm start** (Population Prior + N-gram + ML — ยังไม่ดีกว่าอย่างมีนัยสำคัญ ดู [reports/warm_start.md](reports/warm_start.md)) |
| **L** | **โมเดลที่เซฟไว้** | — | โหลดโมเดลจากโฟลเดอร์ `models/` มาเป็นคู่ต่อสู้ |

\* Net Score = (บอสชนะ − บอสแพ้) ÷ จำนวนตา × 100 เฉลี่ยกับคู่ต่อสู้ที่จับทางได้ (0 = เท่าสุ่ม)

**บอสที่ตัดออกจากเมนูเกม:** Random (~0), Frequency (−0.5), Decay (−10.0), Q-learning (20.7), Ensemble เดิม 4 ตัว (36.7)
โค้ดยังอยู่ใน `adaptive_arena.py` (`BOSSES`) ให้สคริปต์วิเคราะห์ใช้เป็นตัวเทียบ และโหลดโมเดลเก่าที่เซฟไว้ได้ตามปกติ

---

## 💾 โฟลวการเล่นเพื่อบันทึกข้อมูลและโหลดกลับมาสู้ต่อ (Gameplay & Persistence Flow)

คุณสามารถเก็บสถิติเพื่อฝึก AI ให้จำสไตล์การเล่นของคุณ และนำกลับมาสู้ใหม่ได้ **2 รูปแบบ**:

### 🎯 รูปแบบที่ 1: บันทึกสมองบอสแบบ Real-time (Online Learning Persistence)
*เหมาะสำหรับการเล่นต่อเนื่อง และต้องการบันทึกสถานะสมองของบอสตัวที่กำลังสู้ทันที*
1. **เริ่มเล่น:** รันเกมผ่านกล้อง `python main_game.py` หรือโหมดไว `python fast_mode.py`
2. **เลือกบอส:** ทุกตัวในเกมเรียนรู้ระหว่างเล่น แนะนำบอส 4 (Lite Ensemble, ค่าเริ่มต้น) หรือบอส 3 (ML)
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
เมื่อเพื่อนหลายคนเล่นในเครื่องเดียวกัน ข้อมูลจะถูกบันทึก **แยกไฟล์ต่อคน** ที่ `data/players/<ชื่อผู้เล่น>.csv` ทำให้สามารถสั่งเทรนโมเดลแก้ทางเฉพาะบุคคลได้:
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
2. **Data Ingestion & Telemetry Logging:** ทุกการเล่นใน [main_game.py](main_game.py) จะบันทึกผลลง `data/players/<ชื่อผู้เล่น>.csv` อัตโนมัติ (หรือใช้ Simulator จำลองพฤติกรรมมนุษย์)
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
python model_pipeline.py --mode train --input data/players
```

---

## ⌨️ ปุ่มควบคุมและคีย์ลัดในเกม Webcam

| ปุ่ม | คำสั่ง | รายละเอียด |
|:---:|:---|:---|
| `SPACE` | **เล่น 1 ตา** | เกมจะ **รอจนเห็นท่า** (ค้อน/กรรไกร/กระดาษ) ก่อน แล้วค่อยนับ 3-2-1 |
| `A` | **โหมดออโต้** | เล่นต่อเนื่องโดยไม่ต้องลดมือ: เปลี่ยนท่า = ออกทันที, ค้างท่าเดิม ~0.8 วิ = ออกท่าเดิมซ้ำ |
| `T` | **ความเร็วนับถอยหลัง** | 3s ➔ 2s ➔ 1.5s ➔ 1s |
| `B` | **เลือกบอส** | เปิดรายชื่อบอสทั้ง 4 ตัว แล้วกดตัวเลข |
| `1` - `4` | **เปลี่ยนบอส** | 1 Markov · 2 N-gram · 3 ML · 4 Lite Ensemble |
| `4` (กดซ้ำ) | **สลับขนาด Lite Ensemble** | 2 ➔ 3 ➔ 4 โมเดล ➔ Warm start |
| `S` / `L` | **บันทึก / โหลดโมเดล** | เซฟสมองบอสลง `models/` หรือโหลดกลับมาสู้ (กด L ซ้ำเพื่อเปลี่ยนไฟล์) |
| `N` | **ผู้เล่นคนใหม่** | คนใหม่มาเล่นต่อโดยไม่ต้องปิดเกม: สรุปผลคนเก่าในเทอร์มินัล, ตั้งชื่อ `visitor_<เดือนวัน>_<ลำดับ>` อัตโนมัติ, ล้างความจำบอส, แยกไฟล์ข้อมูลใหม่ |
| `R` | **รีเซ็ต** | ล้างคะแนนและความจำบอส |
| `M` | **เสียง** | เปิด/ปิดเสียงเอฟเฟกต์ |
| `G` | **เส้นโครงมือ** | แสดง/ซ่อนจุด Landmark ของ MediaPipe |
| `H` | **คู่มือปุ่ม** | แสดงปุ่มทั้งหมดบนจอ |
| `Q` / `ESC` | **ออกจากเกม** | ปิดหน้าต่างและคืนกล้อง |

### ⏱️ Reaction time ที่บันทึกลง `reaction_time_ms`
- **เกมกล้อง:** เวลาตั้งแต่เริ่มตา จนกล้องเห็นท่าใหม่นิ่ง ≥ 0.15 วิ — ถ้าเปลี่ยนท่าระหว่างนับถอยหลัง จะนับถึงท่าสุดท้าย
- ตาที่ "ค้างท่าเดิมไว้เพื่อออกซ้ำ" จะเว้นว่าง reaction time ไว้ (ผู้เล่นไม่ได้ขยับ จึงวัดไม่ได้)
- บอสออกท่า (emoji บนการ์ดบอส) ตอนนับถึง "ออก!" และท่าผู้เล่นถูกล็อกในจังหวะเดียวกัน จึงเปลี่ยนตามบอสไม่ได้
- **Fast mode:** เวลาตั้งแต่ขึ้นคำถามจนกด Enter
- ข้อมูลเก่าก่อนการแก้ไขนี้ (`Guest` และแถวที่ค่าเป็น 500/1000/1500/3000 พอดี) **ไม่ใช่** reaction time จริง เป็นค่าความเร็วนับถอยหลังที่ตั้งไว้ / ค่าสุ่มจาก simulator


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
python fast_mode.py --pace 0.5 --boss 4

# 3. รันโหมดจำลองประลองผ่าน Terminal
python adaptive_arena.py

# 4. รันไปป์ไลน์ Data Science เพื่อเทรนโมเดลใหม่จากข้อมูลการเล่น
python model_pipeline.py --mode full

# 5. ทดสอบประสิทธิภาพ Efficiency Score เทียบกับโมเดลสุ่ม
python test_efficiency.py

# 6. ทดสอบ Ensemble แบบ 2 / 3 / 4 โมเดล ทุกชุดค่าผสม (ผล: reports/ensemble_combos.md)
python benchmark_ensemble_combos.py

# 7. ใช้ได้กับคนใหม่ไหม? ทดสอบ Warm start แบบ Leave-One-Player-Out (ผล: reports/warm_start.md)
python benchmark_warm_start.py

# เริ่มเกมกล้องในโหมดออโต้ทันที
python main_game.py --player Ken --auto
```

---

## 📂 โครงสร้างไฟล์ในโปรเจกต์

- [main_game.py](main_game.py) — สคริปต์หลักเล่นผ่านกล้องเว็บแคม (รองรับปุ่ม `[T]` ปรับความเร็ว 1.0s - 3.0s)
- [fast_mode.py](fast_mode.py) — ⚡ โหมดประลองด่วน (No-Cam) เลือกท่าผ่านคีย์บอร์ด ปรับเวลาต่อตาได้อิสระ บันทึก Telemetry ครบถ้วน
- [model_pipeline.py](model_pipeline.py) — End-to-End Data Science & Machine Learning Pipeline
- [model_io.py](model_io.py) — ระบบบันทึก (Save/Export) และโหลด (Load/Import) โมเดลพร้อม Model Card
- [adaptive_arena.py](adaptive_arena.py) — โมเดล AI สมองบอสทั้งหมด (`BOSSES` 8 ตัวสำหรับวิเคราะห์, `GAME_BOSSES` 4 ตัวในเกม) และระบบจำลองการแข่ง
- [telemetry.py](telemetry.py) — บันทึก/อ่านข้อมูลการเล่นแยกไฟล์ตามผู้เล่น
- [hand_detector.py](hand_detector.py) — โมเดล Vision ตรวจจับพิกัดมือ 21 จุดและแยกแยะนิ้วด้วย MediaPipe
- [ui_renderer.py](ui_renderer.py) — ระบบวาด HUD ภาษาไทย การ์ดผู้เล่น/บอส (บอสออกท่าเป็น emoji แบบมีแอนิเมชัน) และ Toast
- [sound_effects.py](sound_effects.py) — ระบบเสียงเอฟเฟกต์แบบ Non-blocking (พร้อม Cross-platform fallback)
- [benchmark_ensemble_combos.py](benchmark_ensemble_combos.py) — ทดสอบทุกชุด Ensemble 2/3/4 โมเดล หาชุดที่โหดสุดเทียบกับทรัพยากร
- [benchmark_warm_start.py](benchmark_warm_start.py) — ทดสอบกับ "คนใหม่ที่ไม่เคยเห็นข้อมูล" (leave-one-player-out) และ Population Prior
- [test_efficiency.py](test_efficiency.py) — คำนวณค่า Efficiency Score (%) เทียบกับ Nash Baseline
- [evaluate_ai.py](evaluate_ai.py) — ทดสอบประสิทธิภาพ AI กับ Persona รูปแบบต่างๆ
- `data/players/` — ข้อมูลการเล่น **แยกไฟล์ต่อคน** (`ken.csv`, `nine.csv`, `three.csv`, ...) ส่วน `guest.csv` คือข้อมูลจำลองจาก simulator ไม่ใช่คนจริง
  - สคริปต์วิเคราะห์ทุกตัวอ่านรวมทุกไฟล์อัตโนมัติ (`--csv data/players` เป็นค่าเริ่มต้น) หรือส่งไฟล์เดียว เช่น `--csv data/players/ken.csv`
  - ตอนใส่ชื่อ เกมจะโชว์ชื่อที่มีอยู่แล้ว และเตือนถ้าพิมพ์ไทยปนอังกฤษ (เช่น `ธThree` จากลืมเปลี่ยนภาษาคีย์บอร์ด)
  - กด Enter ไม่ใส่ชื่อ = ได้ชื่อ `visitor_...` ไม่ซ้ำใคร (แต่ละคนแยกไฟล์) · ชื่อ `guest` ห้ามใช้ (สงวนไว้ให้ข้อมูลจำลอง)
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
