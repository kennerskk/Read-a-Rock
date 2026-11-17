# ตัวเลขและรูปสำหรับสไลด์นำเสนอ

**สร้างเมื่อ:** 2026-11-17 07:59:19

---

## สไลด์ข้อ 3 — Data Sets & Class Labels

### Class labels (3 คลาส)

| index | ภาษาไทย | English | จำนวนตัวอย่าง |
|---:|:---|:---|---:|
| 0 | ค้อน | Rock | 631 |
| 1 | กรรไกร | Scissors | 570 |
| 2 | กระดาษ | Paper | 614 |

### การแบ่งข้อมูล (Walk-Forward Time-Series Split)

| รายการ | ค่า |
|:---|---:|
| ข้อมูลทั้งหมด (ตา) | 1815 |
| จำนวนฟีเจอร์ | 12 |
| Train เริ่มต้น | 726 ตา (40%) |
| Validation ต่อ fold | 108 ตา |
| จำนวน fold | 10 |
| ตัวอย่าง out-of-sample รวม | 1080 |

> ใช้ walk-forward (expanding window) แทนการ random split เพราะข้อมูลเป็นอนุกรมเวลา การสุ่มสลับจะทำให้โมเดลเห็นอนาคต (data leakage)

---

## สไลด์ข้อ 4 — Confusion Matrix & ROC

**โมเดลที่ดีที่สุด (เลือกจาก macro AUC): `RandomForest` — AUC = 0.759**

### ตารางเทียบทุกโมเดล

| โมเดล | Accuracy (95% CI) | Macro AUC | สรุป |
|:---|---:|---:|:---|
| LogisticRegression | 55.46% ± 11.01% | 0.703 |  |
| RandomForest | 57.41% ± 13.28% | 0.759 | **ดีที่สุด** |
| GradientBoosting | 58.98% ± 10.55% | 0.745 |  |

### Confusion Matrix (ค่าดิบ)

| actual \ predicted | Rock | Scissors | Paper | total |
|:---|---:|---:|---:|---:|
| **Rock** | 213 | 124 | 55 | 392 |
| **Scissors** | 56 | 188 | 101 | 345 |
| **Paper** | 75 | 49 | 219 | 343 |

### Precision / Recall / F1 ต่อคลาส

| คลาส | Precision | Recall | F1 | Support |
|:---|---:|---:|---:|---:|
| Rock (ค้อน) | 0.619 | 0.543 | 0.579 | 392 |
| Scissors (กรรไกร) | 0.521 | 0.545 | 0.533 | 345 |
| Paper (กระดาษ) | 0.584 | 0.638 | 0.610 | 343 |

### AUC แยกตามคลาสของโมเดลที่ดีที่สุด

| คลาส | AUC |
|:---|---:|
| Rock | 0.766 |
| Scissors | 0.735 |
| Paper | 0.775 |

---

## ไฟล์รูปที่สร้าง

- `reports/figures/confusion_matrix_best.png` — Confusion matrix ของโมเดลที่ดีที่สุด (สไลด์ข้อ 4)
- `reports/figures/roc_comparison.png` — ROC เทียบทุกโมเดล (สไลด์ข้อ 4)
- `reports/figures/roc_per_class_best.png` — ROC แยกคลาส (สไลด์เสริม)
- `reports/figures/model_comparison_bar.png` — Accuracy + 95% CI (สไลด์ข้อ 4)

> ป้ายคลาสในรูปใช้ภาษาอังกฤษ (Rock / Scissors / Paper) เพราะฟอนต์เริ่มต้นของ matplotlib ไม่รองรับภาษาไทย จะแสดงเป็นสี่เหลี่ยมว่าง
