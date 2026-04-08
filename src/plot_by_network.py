import matplotlib.pyplot as plt
import numpy as np
import os

# قراءة البيانات وتصنيفها حسب نوع الشبكة
network_types = {
    'bus': [], 'car': [], 'ferry': [], 'metro': [], 'train': [], 'tram': []
}

test_files = os.listdir('test_results')
for f in test_files:
    for ntype in network_types.keys():
        if ntype in f:
            with open(f'test_results/{f}', 'r') as file:
                lines = [l.strip() for l in file if l.strip()]
                if lines:
                    qoe = float(lines[-1].split()[-1])
                    network_types[ntype].append(qoe)
                    break

# حساب المتوسطات
types = list(network_types.keys())
means = [np.mean(network_types[t]) for t in types]
stds = [np.std(network_types[t]) for t in types]

# إنشاء الرسم البياني
plt.figure(figsize=(12, 6))
bars = plt.bar(types, means, yerr=stds, capsize=8, color='steelblue', alpha=0.7, edgecolor='black')
plt.axhline(y=0.75, color='red', linestyle='--', linewidth=2, label='Target QoE = 0.75')
plt.ylabel('Average QoE', fontsize=12)
plt.xlabel('Network Type', fontsize=12)
plt.title('Performance Across Different Network Scenarios (Optimized Model)', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('final_results/performance_by_network.png', dpi=150)
print("Saved to final_results/performance_by_network.png")
