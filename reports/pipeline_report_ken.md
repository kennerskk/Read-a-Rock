# 📑 รายงานผลการทดลอง Data Science Pipeline (Rock-Paper-Scissors AI)

**วันที่ประมวลผล:** 2026-11-16 20:07:06  
**โมเดลที่ชนะเลิศ (Champion Model):** `RandomForest (Ensemble Trees)`  

---

## 1. ผลการวิเคราะห์พฤติกรรมมนุษย์ (Behavioral Analytics & EDA)
- **จำนวนข้อมูลตัวอย่างทั้งหมด:** 502 ตา
- **การทดสอบความสุ่มทางสถิติ (Chi-Square Test):**
  - $\chi^2 = 7.223$ ($p$-value $= 2.7010e-02$)
  - สรุป: **ผู้เล่นมีแพทเทิร์นจับทางได้ชัดเจน (Non-random)**
- **Shannon Entropy:** `1.575` bits (Normalized: `99.4%`)
- **จิตวิทยา Win-Stay, Lose-Shift (WSLS Empirical Rates):**
  - **Win-Stay Rate:** `23.5%` (ทฤษฎีสุ่ม = 33.3%)
  - **Lose-Shift Rate:** `55.1%` (ทฤษฎีสุ่ม = 33.3%)

---

## 2. ผลการทดสอบโมเดล (Walk-Forward Time-Series Validation)

| อัลกอริทึม (Model Candidate) | Mean Accuracy (%) | Mean Log-Loss (Cross-Entropy) |
|:---|:---:|:---:|
| **LogisticRegression (Softmax)** | 44.44% | 1.1609 |
| **RandomForest (Ensemble Trees)** | 41.75% | 1.0748 |
| **GradientBoosting (Histogram)** | 44.44% | 1.3772 |

---

## 3. ประสิทธิภาพเชิงทฤษฎีเกมของโมเดลแชมเปียน (Game-Theoretic Benchmark)
- **อัตราการชนะของ AI บอส:** `71.0%`
- **อัตราการเสมอ:** `12.9%`
- **อัตราการแพ้ของ AI บอส:** `16.1%`
- **Expected Value $E[R]$:** `+0.548` (ช่วงค่า [-1, +1])
- **Efficiency Score เทียบกับ Random Baseline:** `+54.8%`

---

## 4. คำแนะนำในการนำไปใช้ (Deployment & Inference)
โมเดลนี้ได้รับการ Export ไว้ที่ `models/ds_champion_boss.json` เรียบร้อยแล้ว  
คุณสามารถนำไปใช้งานได้ทันที 2 รูปแบบ:
1. **ในเกมกล้อง Webcam ([main_game.py](main_game.py)):** กดปุ่ม `L` เพื่อโหลดโมเดลแชมเปียนนี้เข้ามาสู้สด
2. **ในระบบจำลอง ([adaptive_arena.py](adaptive_arena.py)):** เลือกเมนู `L` เพื่อประลองผ่าน Terminal
