# 📑 รายงานผลการทดลอง Data Science Pipeline (Rock-Paper-Scissors AI)

**วันที่ประมวลผล:** 2026-11-16 14:54:10  
**โมเดลที่ชนะเลิศ (Champion Model):** `RandomForest (Ensemble Trees)`  

---

## 1. ผลการวิเคราะห์พฤติกรรมมนุษย์ (Behavioral Analytics & EDA)
- **จำนวนข้อมูลตัวอย่างทั้งหมด:** 197 ตา
- **การทดสอบความสุ่มทางสถิติ (Chi-Square Test):**
  - $\chi^2 = 0.254$ ($p$-value $= 8.8082e-01$)
  - สรุป: **ผู้เล่นเล่นใกล้เคียงการสุ่ม (Random)**
- **Shannon Entropy:** `1.584` bits (Normalized: `99.9%`)
- **จิตวิทยา Win-Stay, Lose-Shift (WSLS Empirical Rates):**
  - **Win-Stay Rate:** `15.4%` (ทฤษฎีสุ่ม = 33.3%)
  - **Lose-Shift Rate:** `64.2%` (ทฤษฎีสุ่ม = 33.3%)

---

## 2. ผลการทดสอบโมเดล (Walk-Forward Time-Series Validation)

| อัลกอริทึม (Model Candidate) | Mean Accuracy (%) | Mean Log-Loss (Cross-Entropy) |
|:---|:---:|:---:|
| **LogisticRegression (Softmax)** | 58.77% | 0.9937 |
| **RandomForest (Ensemble Trees)** | 57.89% | 0.9527 |
| **GradientBoosting (Histogram)** | 60.53% | 1.1594 |

---

## 3. ประสิทธิภาพเชิงทฤษฎีเกมของโมเดลแชมเปียน (Game-Theoretic Benchmark)
- **อัตราการชนะของ AI บอส:** `83.3%`
- **อัตราการเสมอ:** `12.5%`
- **อัตราการแพ้ของ AI บอส:** `4.2%`
- **Expected Value $E[R]$:** `+0.792` (ช่วงค่า [-1, +1])
- **Efficiency Score เทียบกับ Random Baseline:** `+79.2%`

---

## 4. คำแนะนำในการนำไปใช้ (Deployment & Inference)
โมเดลนี้ได้รับการ Export ไว้ที่ `models/ds_champion_boss.json` เรียบร้อยแล้ว  
คุณสามารถนำไปใช้งานได้ทันที 2 รูปแบบ:
1. **ในเกมกล้อง Webcam ([main_game.py](main_game.py)):** กดปุ่ม `L` เพื่อโหลดโมเดลแชมเปียนนี้เข้ามาสู้สด
2. **ในระบบจำลอง ([adaptive_arena.py](adaptive_arena.py)):** เลือกเมนู `L` เพื่อประลองผ่าน Terminal
