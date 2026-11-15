# 📑 รายงานผลการทดลอง Data Science Pipeline (Rock-Paper-Scissors AI)

**วันที่ประมวลผล:** 2026-10-21 18:45:15  
**โมเดลที่ชนะเลิศ (Champion Model):** `RandomForest (Ensemble Trees)`  

---

## 1. ผลการวิเคราะห์พฤติกรรมมนุษย์ (Behavioral Analytics & EDA)
- **จำนวนข้อมูลตัวอย่างทั้งหมด:** 1200 ตา
- **การทดสอบความสุ่มทางสถิติ (Chi-Square Test):**
  - $\chi^2 = 2.580$ ($p$-value $= 2.7527e-01$)
  - สรุป: **ผู้เล่นเล่นใกล้เคียงการสุ่ม (Random)**
- **Shannon Entropy:** `1.583` bits (Normalized: `99.9%`)
- **จิตวิทยา Win-Stay, Lose-Shift (WSLS Empirical Rates):**
  - **Win-Stay Rate:** `59.5%` (ทฤษฎีสุ่ม = 33.3%)
  - **Lose-Shift Rate:** `70.3%` (ทฤษฎีสุ่ม = 33.3%)

---

## 2. ผลการทดสอบโมเดล (Walk-Forward Time-Series Validation)

| อัลกอริทึม (Model Candidate) | Mean Accuracy (%) | Mean Log-Loss (Cross-Entropy) |
|:---|:---:|:---:|
| **LogisticRegression (Softmax)** | 68.20% | 0.8030 |
| **RandomForest (Ensemble Trees)** | 75.31% | 0.7026 |
| **GradientBoosting (Histogram)** | 70.99% | 0.8680 |

---

## 3. ประสิทธิภาพเชิงทฤษฎีเกมของโมเดลแชมเปียน (Game-Theoretic Benchmark)
- **อัตราการชนะของ AI บอส:** `79.3%`
- **อัตราการเสมอ:** `11.7%`
- **อัตราการแพ้ของ AI บอส:** `9.0%`
- **Expected Value $E[R]$:** `+0.702` (ช่วงค่า [-1, +1])
- **Efficiency Score เทียบกับ Random Baseline:** `+70.2%`

---

## 4. คำแนะนำในการนำไปใช้ (Deployment & Inference)
โมเดลนี้ได้รับการ Export ไว้ที่ `models/ds_champion_boss.json` เรียบร้อยแล้ว  
คุณสามารถนำไปใช้งานได้ทันที 2 รูปแบบ:
1. **ในเกมกล้อง Webcam ([main_game.py](main_game.py)):** กดปุ่ม `L` เพื่อโหลดโมเดลแชมเปียนนี้เข้ามาสู้สด
2. **ในระบบจำลอง ([adaptive_arena.py](adaptive_arena.py)):** เลือกเมนู `L` เพื่อประลองผ่าน Terminal
