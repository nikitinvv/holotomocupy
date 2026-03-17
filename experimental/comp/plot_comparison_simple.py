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
    "font.size": 24, "axes.titlesize": 26, "axes.labelsize": 24,
    "legend.fontsize": 21, "xtick.labelsize": 22, "ytick.labelsize": 22,
})

fig, ax_err = plt.subplots(figsize=(7, 10))
#fig.suptitle("Gradient descent with BH step till iter 3, other methods after", fontsize=15)

for m in [0, 2, 3]:
    path = os.path.join(base, f"comp_method{m}_bin2", "conv_st2.csv")
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    df = pd.read_csv(path)
    ax_err.semilogy(df["iter"], df["err"], label=labels[m], color=method_color[m], lw=2.5)

ax_err.set_xlabel("iteration")
ax_err.set_ylabel("error")
ax_err.legend(loc='upper center')
ax_err.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
out = os.path.join(base, "comp_methods_st2.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
