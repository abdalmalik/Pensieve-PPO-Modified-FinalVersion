import matplotlib.pyplot as plt
import numpy as np

# قيم Baseline (من الاختبار السابق)
# ملاحظة: هذه قيم تقريبية، يمكنك استخدام القيم الفعلية من test_results القديمة
baseline_qoe = [0.887, 2.161, 4.978, 6.357, 8.390, 9.429, 10.544, 11.438, 11.795, 12.556, -3.065, 0.75, 1.20, -0.45]

# قيم Optimized (142 قيمة)
optimized_qoe = [4.3,1.85,1.85,0.3,0.75,4.3,4.3,4.3,1.85,0.75,1.48376112992822,2.85,0.75,2.85,-9.53626395545971,1.85,0.3,-3.15215227789635,0.3,1.85,4.3,2.85,1.85,0.75,2.85,1.85,0.3,0.75,0.75,1.85,0.75,0.75,0.3,0.85,1.85,0.3,0.75,1.85,1.85,-1.25,1.85,1.85,4.3,0.75,0.3,0.3,-0.15,0.3,0.75,0.3,0.75,1.4,1.85,0.3,1.85,1.85,-0.35,-0.35,1.85,1.85,0.75,-0.15,0.75,0.75,0.3,0.3,1.85,1.85,4.3,-0.6,0.75,1.85,4.3,4.3,1.85,1.85,0.75,1.85,-0.35,0.75,1.85,2.85,1.85,0.3,0.75,-0.35,0.3,1.85,0.75,0.75,0.75,1.85,0.3,0.75,0.75,0.75,0.75,0.75,1.85,1.85,-0.15,0.3,0.75,1.85,0.75,0.75,0.75,0.3,0.3,-0.15,0.75,0.75,0.75,0.75,-6.46085154963581,0.75,1.85,0.75,0.75,0.75,0.75,0.75,1.85,0.75,0.75,0.3,0.55,0.75,-1.25,0.3,0.75,0.3,0.3,0.75,1.85,2.85,0.75,-0.35,1.85,1.85,0.75,0.75]

# إنشاء الرسم البياني
plt.figure(figsize=(10, 6))
data = [baseline_qoe, optimized_qoe]
bp = plt.boxplot(data, labels=['Baseline (17,700 epochs)', 'Optimized (99,900 epochs)'], patch_artist=True)

# تلوين المربعات
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightgreen')

plt.ylabel('Quality of Experience (QoE)', fontsize=12)
plt.title('Comparison: Baseline vs Optimized Model', fontsize=14)
plt.axhline(y=0.75, color='green', linestyle='--', linewidth=1.5, label='Target QoE = 0.75')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('final_results/comparison_boxplot.png', dpi=150)
print("Saved to final_results/comparison_boxplot.png")
