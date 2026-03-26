import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cols = [
    ('figs/mreczz.png',  'figs/preczz.png',  'figs/profile_zz.png'),
    ('figs/mreczz2.png', 'figs/preczz2.png', 'figs/profile_zz2.png'),
    ('figs/mrecyy.png',  'figs/precyy.png',  'figs/profile_yy.png'),
    ('figs/mrecyy3.png', 'figs/precyy3.png', 'figs/profile_yy3.png'),
]
row_titles = ['proposed', 'conventional', '']

# Load all images to get actual pixel dimensions
imgs = [[mpimg.imread(fpath) for fpath in col] for col in cols]

col_widths  = [imgs[c][0].shape[1] for c in range(4)]   # width from row 0
row_heights = [max(imgs[c][r].shape[0] for c in range(4)) for r in range(3)]

dpi = 150
fig_w = sum(col_widths)  / dpi
fig_h = sum(row_heights) / dpi

fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
gs  = fig.add_gridspec(3, 4,
                        height_ratios=row_heights,
                        width_ratios=col_widths,
                        hspace=0.02, wspace=0.02)

for c, col_imgs in enumerate(imgs):
    for r, img in enumerate(col_imgs):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(img)
        ax.axis('off')
        if c == 0 and row_titles[r]:
            ax.set_ylabel(row_titles[r], fontsize=16, labelpad=6)

base = 'figs/comparison'
i = 0
while os.path.exists(f'{base}_{i:03d}.png'):
    i += 1
out = f'{base}_{i:03d}.png'
plt.savefig(out, dpi=dpi, bbox_inches='tight', pad_inches=0.02)
plt.close()
print(f"Done — {out} saved")
