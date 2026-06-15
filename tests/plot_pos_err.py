"""Plot position-error runs: position errors, additional error, recovered,
and recovered + position errors."""
import numpy as np
import matplotlib.pyplot as plt

# (1) Position errors — iter=608
pos_y = np.array([-0.0013,  0.0023, -0.0089,  0.0174, -0.0131,  0.0633,  0.0652,  0.0548,
                  -0.0029,  0.0512,  0.0505,  0.0639,  0.0022,  0.0776,  0.0511,  0.0194])
pos_x = np.array([ 0.0827,  0.0526,  0.0007,  0.0545,  0.0126, -0.05  , -0.0644,  0.0278,
                  -0.0414, -0.075 , -0.0671, -0.0452,  0.1246,  0.1026,  0.0363,  0.0812])

# (2) Additional error — iter=480
add_y = np.array([-0.0057,  0.2173,  0.0366, -0.1517, -0.1492,  0.1877,  0.2277, -0.142 ,
                   0.0354,  0.295 ,  0.0098,  0.0803, -0.0956, -0.1077, -0.1109,  0.1308])
add_x = np.array([ 0.1183,  0.0559, -0.1495,  0.3051, -0.084 ,  0.1321, -0.13  , -0.0232,
                  -0.1129, -0.0737, -0.2303, -0.2291,  0.4115,  0.0265, -0.0683,  0.1691])

# (3) Recovered (sign flipped)
rec_y = -np.array([ 0.0151, -0.203 , -0.0336,  0.1815,  0.1482, -0.11  , -0.1509,  0.2092,
                   -0.0302, -0.2331,  0.0486, -0.0076,  0.1071,  0.1936,  0.1688, -0.1036])
rec_x = -np.array([ 0.0206,  0.0523,  0.2049, -0.194 ,  0.1502, -0.129 ,  0.1173,  0.1051,
                    0.1231,  0.0499,  0.2133,  0.2341, -0.2323,  0.1303,  0.1562, -0.0337])

# (4) Recovered + position errors
sum_y = rec_y + pos_y
sum_x = rec_x + pos_x

idx = np.arange(len(pos_y))

fig, axs = plt.subplots(2, 2, figsize=(13, 9))
panels = [
    (axs[0, 0], pos_y, pos_x, 'position errors'),
    (axs[0, 1], add_y, add_x, 'additional error'),
    (axs[1, 0], rec_y, rec_x, 'recovered'),
    (axs[1, 1], sum_y, sum_x, 'recovered + position errors'),
]

# Shared y-limits for visual comparison across panels.
all_vals = np.concatenate([pos_y, pos_x, add_y, add_x, rec_y, rec_x, sum_y, sum_x])
ylim = (all_vals.min() - 0.05, all_vals.max() + 0.05)

for ax, ey, ex, title in panels:
    ax.plot(idx, ey, 'o-', label=f'err y  (mean {ey.mean():+.4f}, std {ey.std():.4f})')
    ax.plot(idx, ex, 's-', label=f'err x  (mean {ex.mean():+.4f}, std {ex.std():.4f})')
    ax.axhline(0, color='k', lw=0.7, ls=':')
    ax.set_xlabel('position index')
    ax.set_ylabel('error [px]')
    ax.set_title(title)
    ax.set_ylim(ylim)
    ax.legend()
    ax.grid(True, alpha=0.4)

plt.tight_layout()
out = '/home/beams2/VNIKITIN/holotomocupy_mpi/tests/pos_err.png'
plt.savefig(out, dpi=120)
print(f'saved {out}')
