"""Plot error from conv.csv for each method."""
import pandas as pd
import matplotlib.pyplot as plt
import os

labels = {
    0: "BH β, BH α",
    2: "Polak–Ribière β, strong Wolfe LS α",
    3: "L-BFGS (m=5), strong Wolfe LS α",
}

base = "/data2/vnikitin"

method_color = {0: 'C0', 2: 'C1', 3: 'C2'}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18,
    "legend.fontsize": 15, "xtick.labelsize": 16, "ytick.labelsize": 16,
})

fig, ax_err = plt.subplots(figsize=(5.5, 10))
fig.suptitle("Brain data [1125, 4, 512, 512]", fontsize=15)

for m in [0, 2, 3]:
    path = os.path.join(base, f"comp_method{m}_bin2", "conv.csv")
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    df = pd.read_csv(path)
    ax_err.semilogy(df["iter"], df["err"], label=labels[m], color=method_color[m])

ax_err.set_xlabel("iteration")
ax_err.set_ylabel("error")
ax_err.legend(loc='upper right')
ax_err.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
out = os.path.join(base, "comp_methods.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
