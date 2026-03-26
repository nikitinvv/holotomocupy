import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib as mpl

mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        16,
})

imgs = {}
for name in ['mrecz', 'precz', 'mrecy', 'precy']:
    imgs[name] = mpimg.imread(f'figs/{name}.png')

h_z = imgs['mrecz'].shape[0]
h_y = imgs['mrecy'].shape[0]

fig = plt.figure(figsize=(14, 14 * (h_z + h_y) / (imgs['mrecz'].shape[1] * 2)))
gs  = gridspec.GridSpec(2, 2, height_ratios=[h_z, h_y], hspace=0.02, wspace=0.02)

layout = [('precz', 'mrecz'), ('precy', 'mrecy')]

for ri, (left_key, right_key) in enumerate(layout):
    for ci, key in enumerate([left_key, right_key]):
        ax = fig.add_subplot(gs[ri, ci])
        ax.imshow(imgs[key])
        ax.axis('off')


fig.savefig('figs/compare_full.png', dpi=200, bbox_inches='tight', pad_inches=0.05)
print('Saved figs/compare_full.png')
