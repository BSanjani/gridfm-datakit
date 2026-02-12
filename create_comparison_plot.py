import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

metrics = ["Vm MAE\n(% pu)", "Vm RMSE\n(pu)", "Vm R²", "Va MAE\n(deg)", "Va R²"]
baseline = [0.476, 1.177, 0.533, 2.749, 0.738]
droop = [0.359, 0.552, 0.895, 1.160, 0.946]

x = np.arange(len(metrics))
width = 0.35

ax1 = axes[0]
bars1 = ax1.bar(x - width/2, baseline, width, label="Baseline", color="#FF6B6B", alpha=0.8)
bars2 = ax1.bar(x + width/2, droop, width, label="Droop Control", color="#4ECDC4", alpha=0.8)
ax1.set_ylabel("Value", fontsize=12, fontweight="bold")
ax1.set_title("Performance Metrics Comparison", fontsize=14, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

improvements = [(baseline[i] - droop[i])/baseline[i]*100 if i != 2 and i != 4 else (droop[i] - baseline[i])/baseline[i]*100 for i in range(len(metrics))]

ax2 = axes[1]
colors = ["#51CF66" if imp > 0 else "#FF6B6B" for imp in improvements]
bars = ax2.bar(metrics, improvements, color=colors, alpha=0.8)
ax2.set_ylabel("Improvement (%)", fontsize=12, fontweight="bold")
ax2.set_title("Droop Control Improvement over Baseline", fontsize=14, fontweight="bold")
ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.8)
ax2.grid(axis="y", alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, improvements)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f"{val:.1f}%", ha="center", va="bottom" if val > 0 else "top", fontweight="bold")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
print("Saved: model_comparison.png")
