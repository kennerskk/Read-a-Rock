# Lightweight vs Heavyweight Models — การเปรียบเทียบบนสนามทดสอบกลาง

**สร้างเมื่อ:** 2026-11-17 08:13:45

---

## 1. ปัญหาและวิธีแก้

เดิมสองตระกูลโมเดลวัดกันคนละหน่วย จึงเทียบกันตรง ๆ ไม่ได้:

| ตระกูล | เดิมวัดด้วย | ปัญหา |
|:---|:---|:---|
| Lightweight (บอส 8 ตัว) | win rate จากการจำลองกับ persona | ไม่ได้ใช้ข้อมูลจริง |
| Heavyweight (sklearn 3 ตัว) | accuracy จาก walk-forward บน CSV | ไม่ได้วัดผลในเกม |

### Protocol: Prequential Evaluation (Test-Then-Train)

ให้ทั้งสองตระกูลทำ **งานเดียวกัน** คือทำนายท่าถัดไปของผู้เล่น บนข้อมูลจริงชุดเดียวกัน ลำดับเดียวกัน ทีละตาตามเวลาจริง:

```
สำหรับตาที่ t:
  1. ทุกโมเดลทำนายท่าที่ t  (เห็นข้อมูลถึงตาที่ t-1 เท่านั้น)
  2. บันทึกถูก/ผิด และผลแพ้ชนะถ้านำไปเล่นจริง
  3. เปิดเฉลยตาที่ t ให้ทุกโมเดลเรียนรู้
```

ข้อดีของ protocol นี้:

- **ไม่มี data leakage** — ทุกการทำนายใช้เฉพาะอดีตเท่านั้น
- **จับคู่สมบูรณ์ (perfectly paired)** — ทุกโมเดลเจอตาเดียวกันเป๊ะทุกตา จึงใช้ McNemar's test ได้โดยตรง ไม่ต้องกังวลเรื่องความแปรปรวนของข้อมูล
- **ยุติธรรมกับทั้งสองฝั่ง** — ให้ warm-up เท่ากัน (40% แรกของลำดับ) ฝั่ง lightweight เรียนออนไลน์ผ่านช่วงนั้น ฝั่ง heavyweight เทรนด้วยช่วงนั้น แล้วเริ่มนับคะแนนพร้อมกัน

| พารามิเตอร์ | ค่า |
|:---|---:|
| ผู้เล่นที่นำมา replay | Guest, ken |
| จำนวนตาที่นับคะแนน | 1088 |
| warm-up | 40% |
| heavyweight เทรนใหม่ทุก | 25 ตา |
| ขนาดบล็อกสำหรับ Friedman | 50 ตา |
| จำนวนบล็อก | 21 |

---

## 2. ผลรวม 4 มิติ

| โมเดล | ตระกูล | Accuracy (95% CI) | R (%) | Latency (μs/ตา) | ขนาด (KB) |
|:---|:---|---:|---:|---:|---:|
| HW-RandomForest | HW | 67.05% ± 7.42% | 75.5 | 3904.6 | 545.7 |
| HW-GradientBoosting | HW | 64.29% ± 6.88% | 72.9 | 2212.0 | 416.8 |
| LW-Markov | LW | 61.90% ± 6.53% | 69.5 | 1.3 | 0.2 |
| LW-DataScienceMLBoss | LW | 61.52% ± 6.73% | 69.7 | 8.0 | 5.8 |
| LW-N-gram | LW | 61.33% ± 7.32% | 69.7 | 1.3 | 0.4 |
| LW-Ensemble/Meta | LW | 61.33% ± 7.23% | 69.4 | 7.3 | 0.8 |
| HW-LogisticRegression | HW | 61.33% ± 6.72% | 69.9 | 81.5 | 1.0 |
| LW-Q-learning | LW | 56.38% ± 6.87% | 67.4 | 0.7 | 0.5 |
| LW-Frequency | LW | 35.52% ± 2.70% | 50.0 | 1.1 | 0.2 |
| LW-Random | LW | 32.95% ± 3.82% | 49.8 | 0.5 | 0.0 |
| LW-Recency-decay | LW | 25.90% ± 4.77% | 49.0 | 0.9 | 0.2 |

> baseline: accuracy สุ่ม = 33.3% | R สุ่ม = 50.0%

---

## 3. จัดอันดับรวม — Friedman + Nemenyi

- **Friedman χ² = 109.912**, df = 10, **p = 5.558e-19**
- p < 0.05 ⇒ โมเดลทั้งหมดต่างกันจริง ทำ post-hoc ต่อได้
- **Critical Difference (CD) = 3.295** อันดับ (k = 11, N = 21)

| อันดับ | โมเดล | ตระกูล | อันดับเฉลี่ย |
|---:|:---|:---|---:|
| 1 | HW-RandomForest | **HW** | 2.548 |
| 2 | HW-GradientBoosting | **HW** | 3.881 |
| 3 | LW-Markov | LW | 4.619 |
| 4 | LW-DataScienceMLBoss | LW | 4.714 |
| 5 | LW-N-gram | LW | 5.095 |
| 6 | LW-Ensemble/Meta | LW | 5.095 |
| 7 | HW-LogisticRegression | **HW** | 5.262 |
| 8 | LW-Q-learning | LW | 7.071 |
| 9 | LW-Frequency | LW | 8.690 |
| 10 | LW-Random | LW | 9.262 |
| 11 | LW-Recency-decay | LW | 9.762 |

**กลุ่มนำที่แยกกันไม่ได้** (ห่างจากอันดับ 1 ไม่เกิน CD): HW-RandomForest, HW-GradientBoosting, LW-Markov, LW-DataScienceMLBoss, LW-N-gram, LW-Ensemble/Meta, HW-LogisticRegression

---

## 4. Lightweight vs Heavyweight — คำตอบตรง ๆ

### 4.1 ตัวแทนที่ดีที่สุดของแต่ละฝั่ง (McNemar, จับคู่รายตา)

| | Lightweight | Heavyweight |
|:---|:---|:---|
| ตัวแทน | **LW-Markov** | **HW-RandomForest** |
| Accuracy | 60.94% | 65.62% |
| Latency | 1.3 μs | 3904.6 μs |
| ขนาด | 0.2 KB | 545.7 KB |

- McNemar: b = 92 (LW ถูก/HW ผิด), c = 143 (LW ผิด/HW ถูก), **p = 1.064e-03**
- **สรุป: `HW-RandomForest` แม่นกว่าอย่างมีนัยสำคัญ**

### 4.2 เทียบทั้งตระกูล (ค่าเฉลี่ยต่อบล็อก, Wilcoxon signed-rank)

| ตระกูล | Accuracy เฉลี่ยทั้งตระกูล |
|:---|---:|
| Lightweight (8 โมเดล) | 49.61% |
| Heavyweight (3 โมเดล) | 64.22% |

ผลต่าง = -14.62 pp, p = 1.063e-04, Cohen's d_z = -1.64

> หมายเหตุ: การเฉลี่ยทั้งตระกูลถูกถ่วงด้วยโมเดลอ่อน เช่น Random ที่อยู่ในฝั่ง lightweight ตัวเลขนี้จึงใช้ดูภาพรวมเท่านั้น ข้อสรุปหลักให้ดู 4.1 และตารางอันดับ

---

## 5. ตารางเทียบรายคู่ (ทุกคู่ที่ต่างกันอย่างมีนัยสำคัญ)

McNemar's exact test บนผลทำนายรายตา แก้ค่า p ด้วย Holm ทั้งหมด 55 คู่

พบคู่ที่ต่างกันอย่างมีนัยสำคัญ **37 คู่ จาก 55 คู่**

| โมเดลที่เก่งกว่า | โมเดลที่อ่อนกว่า | b / c | p (Holm) |
|:---|:---|---:|---:|
| HW-RandomForest | LW-Recency-decay | 577 / 155 | 3.18e-56 |
| HW-GradientBoosting | LW-Recency-decay | 556 / 163 | 2.71e-49 |
| HW-RandomForest | LW-Random | 486 / 133 | 1.98e-46 |
| HW-RandomForest | LW-Frequency | 440 / 120 | 3.59e-42 |
| LW-Markov | LW-Recency-decay | 572 / 201 | 2.53e-40 |
| HW-LogisticRegression | LW-Recency-decay | 551 / 189 | 3.56e-40 |
| LW-DataScienceMLBoss | LW-Recency-decay | 569 / 202 | 1.51e-39 |
| HW-GradientBoosting | LW-Random | 473 / 149 | 1.59e-38 |
| LW-N-gram | LW-Recency-decay | 571 / 210 | 1.07e-37 |
| LW-Ensemble/Meta | LW-Recency-decay | 577 / 216 | 4.32e-37 |
| HW-GradientBoosting | LW-Frequency | 424 / 133 | 1.02e-34 |
| LW-Markov | LW-Random | 457 / 155 | 7.42e-34 |
| LW-DataScienceMLBoss | LW-Random | 451 / 153 | 2.08e-33 |
| HW-LogisticRegression | LW-Random | 447 / 154 | 1.98e-32 |
| LW-DataScienceMLBoss | LW-Frequency | 381 / 116 | 2.31e-32 |
| LW-Markov | LW-Frequency | 393 / 124 | 5.51e-32 |
| LW-Q-learning | LW-Recency-decay | 506 / 192 | 6.68e-32 |
| LW-Ensemble/Meta | LW-Random | 450 / 158 | 7.72e-32 |
| LW-N-gram | LW-Random | 451 / 159 | 9.75e-32 |
| HW-LogisticRegression | LW-Frequency | 381 / 121 | 8.46e-31 |
| LW-N-gram | LW-Frequency | 395 / 136 | 9.68e-29 |
| LW-Ensemble/Meta | LW-Frequency | 405 / 146 | 1.27e-27 |
| LW-Q-learning | LW-Random | 403 / 158 | 3.73e-24 |
| LW-Q-learning | LW-Frequency | 365 / 153 | 1.65e-19 |
| HW-RandomForest | LW-Q-learning | 215 / 107 | 5.46e-08 |
| LW-Frequency | LW-Recency-decay | 273 / 171 | 4.45e-05 |
| HW-GradientBoosting | LW-Q-learning | 203 / 124 | 4.26e-04 |
| HW-RandomForest | HW-LogisticRegression | 128 / 68 | 6.11e-04 |
| LW-Markov | LW-Q-learning | 120 / 63 | 8.18e-04 |
| HW-RandomForest | LW-N-gram | 141 / 80 | 1.28e-03 |
| ... | (อีก 7 คู่ ดูในรูป pairwise matrix) | | |

---

## 6. รูปประกอบ

- `reports/figures/lw_vs_hw_accuracy.png`
- `reports/figures/lw_vs_hw_tradeoff.png`
- `reports/figures/lw_vs_hw_pairwise.png`

---

## 7. ข้อสรุป

1. ในการทำนายท่าถัดไปบนข้อมูลจริง โมเดลที่ดีที่สุดคือ **HW-RandomForest** (อันดับเฉลี่ย 2.548)
2. กลุ่มนำที่แยกความต่างกันไม่ได้ทางสถิติ: HW-RandomForest, HW-GradientBoosting, LW-Markov, LW-DataScienceMLBoss, LW-N-gram, LW-Ensemble/Meta, HW-LogisticRegression
3. ด้านความเร็ว lightweight เร็วกว่า heavyweight เฉลี่ย **782 เท่า** (2.6 μs vs 2066.0 μs ต่อตา)
4. ด้านขนาด lightweight เล็กกว่าเฉลี่ย 315 เท่า (1.0 KB vs 321.2 KB)
5. ข้อสรุปเชิงออกแบบระบบ: หากความแม่นยำไม่ต่างกันอย่างมีนัยสำคัญ การเลือก lightweight ให้ผลเท่ากันโดยใช้ทรัพยากรน้อยกว่ามาก เหมาะกับการทำงานแบบ real-time ในเกมที่ต้องตอบสนองทุกเฟรม

---

## 8. ข้อจำกัดของการทดลองนี้ (Limitations)

1. **ความไม่สมมาตรของบริบทเล็กน้อย** — ฝั่ง heavyweight ใช้ฟีเจอร์ `lag1_res` และ `lag1_boss_move` ซึ่งเป็นผลและท่าของ AI ที่บันทึกไว้จริงในล็อก ส่วนฝั่ง lightweight เรียนรู้จากผลของ **ท่าที่ตัวเองเลือก** ซึ่งเป็นสถานการณ์สมมติ (counterfactual) เพราะคู่ต่อสู้จริงในล็อกเป็น AI คนละตัว ความไม่สมมาตรนี้ให้ข้อได้เปรียบกับฝั่ง heavyweight เล็กน้อย ผลที่ lightweight ยังตามมาใกล้เคียงจึงถือว่าน่าประทับใจกว่าที่ตัวเลขแสดง

2. **จำนวนบล็อกน้อย** — ข้อมูลจริงมีจำกัด ทำให้ได้ 21 บล็อก ค่า CD จึงกว้าง (3.29 อันดับ) และ Nemenyi แยกความต่างได้หยาบ **ข้อสรุปรายคู่ให้ยึดตาราง McNemar ในหัวข้อ 5 เป็นหลัก** เพราะใช้ข้อมูลครบทุกตาและจับคู่สมบูรณ์ จึงมี power สูงกว่ามาก

3. **latency วัดเฉพาะการทำนาย** ไม่รวมเวลาเทรน ฝั่ง heavyweight ต้องเทรนใหม่ทุก 25 ตาเพิ่มอีกต่างหาก ถ้ารวมต้นทุนนี้ช่องว่างด้านทรัพยากรจะกว้างกว่าที่รายงาน

4. **ข้อมูลมาจากผู้เล่นเพียง 2 คน** การสรุปไปยังผู้เล่นทั่วไปต้องเก็บข้อมูลเพิ่ม
