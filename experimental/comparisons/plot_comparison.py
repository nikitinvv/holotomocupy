"""Plot error, alpha, beta from conv.csv for each method."""
import pandas as pd
import matplotlib.pyplot as plt
import os

labels = {
    4: "BH β, exact LS α",
    0: "BH β, BH α",
    5: "BH β, strong Wolfe LS α",
    10: "BH β, Armijo LS α",
    1: "Polak–Ribière β, BH α",
    2: "Polak–Ribière β, strong Wolfe LS α",
    7: "Polak–Ribière β, Armijo LS α",
    6: "L-BFGS (m=5), BH α",
    3: "L-BFGS (m=5), strong Wolfe LS α",
    9: "L-BFGS (m=5), Armijo LS α",
    12: "L-BFGS (m=10), BH α",
    8: "L-BFGS (m=10), strong Wolfe LS α",    
    11: "L-BFGS (m=10), Armijo LS α",    
}

base = "/data2/vnikitin"

colors = ["#e6194b", "#3cb44b", "#ffd700", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#a9a9a9", "#808000", "#00ced1", "#ff69b4", "#8b4513", "#228b22"]
method_color = {m: colors[i] for i, m in enumerate(labels)}

# Line style encodes α strategy so methods with the same β are still distinguishable
alpha_ls = {"BH α": "-", "exact LS α": (0,(8,2)), "strong Wolfe LS α": (0,(3,1,1,1)), "Armijo LS α": (0,(1,1))}
method_ls = {m: alpha_ls[next(k for k in alpha_ls if labels[m].endswith(k))] for m in labels}
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11})

fig, axes = plt.subplots(1, 4, figsize=(22, 12))
fig.suptitle("Brain data [1125, 4, 512, 512]", fontsize=15, fontweight="bold")
ax_err, ax_alpha_cg, ax_alpha_lbfgs, ax_beta = axes

for m in labels:
    path = os.path.join(base, f"comp_method{m}_bin2", "conv.csv")
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    df = pd.read_csv(path)
    df_pos = df.reset_index(drop=True)
    label = labels[m]
    color = method_color[m]
    ls = method_ls[m]
    lw = 6 if m == 4 else (4 if m == 0 else 2)

    mask = df["iter"] <= 100
    mask_pos = df_pos["iter"] <= 100
    ax_err.semilogy(df["iter"][mask], df["err"][mask], label=label, color=color, ls=ls, lw=lw)
    if m in (0, 1, 2, 4, 5, 7, 10):
        ax_alpha_cg.semilogy(df_pos["iter"][mask_pos & (df_pos.index >= 1)], df_pos["alpha"].abs()[mask_pos & (df_pos.index >= 1)], label=label, color=color, ls=ls, lw=lw)
        ax_beta.plot(df_pos["iter"][mask_pos & (df_pos.index >= 1)], df_pos["beta"][mask_pos & (df_pos.index >= 1)], label=label, color=color, ls=ls, lw=lw)
    if m in (3, 6, 8, 9, 11, 12):
        ax_alpha_lbfgs.semilogy(df_pos["iter"][mask_pos & (df_pos.index >= 1)], df_pos["alpha"].abs()[mask_pos & (df_pos.index >= 1)], label=label, color=color, ls=ls, lw=lw)
    
    

ax_err.set_title("Error")
ax_err.set_xlabel("Iteration")
ax_err.set_ylabel("err")
ax_err.legend()
ax_err.grid(True, which="both", ls="--", alpha=0.4)

ax_alpha_cg.set_title("Alpha, non-LBFGS")
ax_alpha_cg.set_xlabel("Iteration")
ax_alpha_cg.set_ylabel("alpha")
ax_alpha_cg.legend()
ax_alpha_cg.grid(True, which="both", ls="--", alpha=0.4)

ax_alpha_lbfgs.set_title("Alpha, LBFGS")
ax_alpha_lbfgs.set_xlabel("Iteration")
ax_alpha_lbfgs.set_ylabel("alpha")
ax_alpha_lbfgs.legend()
ax_alpha_lbfgs.grid(True, which="both", ls="--", alpha=0.4)

ax_beta.set_title("Beta, non-LBFGS")
ax_beta.set_xlabel("Iteration")
ax_beta.set_ylabel("beta")
ax_beta.legend()
ax_beta.grid(True, ls="--", alpha=0.4)

plt.tight_layout()
out = os.path.join(base, "comp_methods.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
