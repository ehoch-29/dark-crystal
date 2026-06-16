import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('sweep_results_20260429_120130.csv')

# Convert units
v = df['measured_v'] * 1000   # V -> mV
i = df['dev_I']         # A -> mA

# Segments: 0->+12V, +12->-12V, -12->0
seg1 = slice(0, 120)
seg2 = slice(119, 360)
seg3 = slice(359, 480)

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(v[seg1], i[seg1], color='tab:blue',   label='0 → +12 V', linewidth=1.5)
ax.plot(v[seg2], i[seg2], color='tab:orange', label='+12 → −12 V', linewidth=1.5)
ax.plot(v[seg3], i[seg3], color='tab:green',  label='−12 → 0 V', linewidth=1.5)

ax.set_xlabel('Measured voltage (mV)', fontsize=12)
ax.set_ylabel('Device current (μA)', fontsize=12)
ax.set_title('I–V sweep', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sweep_plot.png', dpi=150, bbox_inches='tight')
print("Saved!")
