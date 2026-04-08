import matplotlib.pyplot as plt
import numpy as np

# قيم QoE من الاختبار
qoe_values = [4.3,1.85,1.85,0.3,0.75,4.3,4.3,4.3,1.85,0.75,1.48376112992822,2.85,0.75,2.85,-9.53626395545971,1.85,0.3,-3.15215227789635,0.3,1.85,4.3,2.85,1.85,0.75,2.85,1.85,0.3,0.75,0.75,1.85,0.75,0.75,0.3,0.85,1.85,0.3,0.75,1.85,1.85,-1.25,1.85,1.85,4.3,0.75,0.3,0.3,-0.15,0.3,0.75,0.3,0.75,1.4,1.85,0.3,1.85,1.85,-0.35,-0.35,1.85,1.85,0.75,-0.15,0.75,0.75,0.3,0.3,1.85,1.85,4.3,-0.6,0.75,1.85,4.3,4.3,1.85,1.85,0.75,1.85,-0.35,0.75,1.85,2.85,1.85,0.3,0.75,-0.35,0.3,1.85,0.75,0.75,0.75,1.85,0.3,0.75,0.75,0.75,0.75,0.75,1.85,1.85,-0.15,0.3,0.75,1.85,0.75,0.75,0.75,0.3,0.3,-0.15,0.75,0.75,0.75,0.75,-6.46085154963581,0.75,1.85,0.75,0.75,0.75,0.75,0.75,1.85,0.75,0.75,0.3,0.55,0.75,-1.25,0.3,0.75,0.3,0.3,0.75,1.85,2.85,0.75,-0.35,1.85,1.85,0.75,0.75]

# إنشاء الرسم البياني
plt.figure(figsize=(12, 8))
plt.hist(qoe_values, bins=25, edgecolor='black', alpha=0.7, color='steelblue')
plt.axvline(np.mean(qoe_values), color='red', linestyle='--', linewidth=2, label=f'Mean QoE = {np.mean(qoe_values):.3f}')
plt.axvline(0.75, color='green', linestyle=':', linewidth=2, label='Target QoE = 0.75')

plt.xlabel('Quality of Experience (QoE)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of QoE Across 142 Test Sessions (Optimized Model)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('final_results/qoe_distribution_optimized.png', dpi=150)
plt.savefig('final_results/qoe_distribution_optimized.pdf')
print("Saved to final_results/qoe_distribution_optimized.png")
