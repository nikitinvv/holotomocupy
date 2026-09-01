# ctxl — cortex tissue, 4-distance HT, 7.5 nm voxels

ESRF ID16A, proposal **ihls3888**, scan taken **2026-08-31 08:23**.
Cortex tissue, four propagation distances, 4000 projections over 180°,
±300 px random sample displacement, 4096² detector at 17.1 keV.

Raw data: `/eagle/APS_IRI/vnikitin/20260829/ctxl/ctxl_HT_4K_RD300_007p5nm_0001*`

This folder is modelled on [`../Y350a_largedisp_006nm`](../Y350a_largedisp_006nm)
(the pipeline, the estimator scripts, the config layout) and on
[`../Y350a_HT`](../Y350a_HT) (the 4-distance step-6 parameters). It has never
been reconstructed; every number below was either measured from the raw data on
2026-08-31 or carried over from a named sibling and marked as such.

> **This is the first 2026 scan in the repository, and its directory layout is
> not the 2025 one.** ESRF moved from *bliss* to *ewoks* between the beamtimes.
> All of that difference is confined to [`esrf_layout.py`](esrf_layout.py) — see
> the table in its docstring — so `steps15.py` and the estimators read both
> flavours and nothing in the raw tree has to be copied or renamed.

| | ctxl (here) | `../Y350a_HT` | `../Y350a_largedisp_006nm` |
|---|---|---|---|
| layout | **ewoks (2026)** | bliss (2025) | bliss (2025) |
| distances | 4 | 4 | 1 |
| energy | 17.1 keV | 33.35 keV | 33.35 keV |
| voxel size | **7.500 nm** | 20 nm | 6.000 nm |
| random displacement | ±300 px | none | ±300 px |
| `nobj` | **5056** | 4608 | 4736 |
| `rotation_center_shift` | **−19.22 px** (weak, refine it) | −10.386 px | −37.50 px |
| `paganin` | **20** | 60 | 40 |
| `correct_motion.txt` | **yes, from ESRF** | yes | estimated locally |
| shrinkage correction | none — *measured* as absent | `rho[tp]` non-zero | none — measured as absent |

## Geometry

Read off the raw data with `python show_geometry.py config_steps15.conf`, and
cross-checked against the four `.info` sidecars, which are written by different
beamline software:

```
energy 17.1 keV   detector pixel 1.47601 um   focus->detector 1212.9965 mm
 k    z1 [mm]    z2 [mm]       mag  norm mag  voxel [nm]  prop [mm]  fringe px
 1     6.1635   1206.833   196.802   1.00000      7.5000     6.1322       88.9
 2     6.4280   1206.569   188.707   0.95886      7.8217     6.3939       87.0
 3     7.4856   1205.511   162.044   0.82338      9.1087     7.4394       80.6
 4     9.6817   1203.315   125.288   0.63662     11.7810     9.6044       70.8
```

`focus->detector` agrees to 7 digits across all four planes, and the derived
voxel sizes reproduce the `.info` `PixelSize` values exactly — which is the
independent check that `z1` was read with the right sign, since the NXtomo
stores it as a *negative* source distance.

Field of view 30.72 µm; angular step 0.045°; exposure 0.1 s; scan duration
0.44 h for all four distances; 20 flats and 20 darks per distance;
4003 EDF frames × 4 planes ≈ 500 GiB raw.

![data overview](scan_overview.png)

Regenerate with
`python scan_overview.py config_steps15.conf --sample "ctxl (cortex tissue)" --proposal "ihls3888 @ ESRF ID16A"`.

Two things in that figure are worth a second look before committing GPU hours.
The mean transmission at distance 1 climbs from 0.98 to 1.25 over the first
~500 frames and then falls back linearly to 1.00 — the flats themselves differ
by 2.6 % between the start and end of the scan, so this is illumination drift
that the 20-flat average does not track. Step 2's flat-field correction is
what it is; the residual shows up as the vertical striping visible in every
projection panel, and it is the probe's job in step 6 to absorb it. And the
sample is a low-contrast one: the only high-contrast feature in the frame is a
near-vertical edge, everything else is fringe-level texture. That is why the
rotation-centre measurement below is as weak as it is.

## Shifts

Step 3 combines four sources. Three of the four are present:

| source | file | present? |
|---|---|---|
| random displacement | `<pfile>/projections/<pfile>_000k.txt` | **yes**, 4003 rows per plane |
| inter-plane (RHAPP) | `<pfile>_/rhapp.mat` | **yes**, 2 × 4 × 4003 |
| slow drift | `<pfile>_2_/correct_motion.txt` | **yes**, from ESRF |
| 3-D tomographic | `<pfile>_/correct_correct3D.txt` | no → step 3 uses zeros |

The commanded displacement is a full ±300 detector px at every plane, redrawn
every frame — consecutive frames jump the whole range, which is the point of
the scheme. In the object frame that is ±300 px at plane 1 and ±471 px at
plane 4 (300/0.63662), i.e. ±3.53 µm of sample motion. `nobj` is sized for it;
see below.

**The drift, and one disagreement worth knowing about.** Unlike
`../Y350a_largedisp_006nm`, ESRF supplied `correct_motion.txt` for this scan, so
nothing had to be fitted. It is the plane-2 random displacement plus a drift, in
exactly the units and column order step 3 assumes. Amplitude over the whole
scan: **2.11 px ptp in x, 1.62 px ptp in y**. It is analytic — consecutive rows
differ by |2nd difference| < 1e-4 px, so ESRF fitted a smooth function rather
than following the retakes frame by frame — but it is *not* a straight line;
taking a linear ramp out leaves a clearly curved residual in both columns.
Their fit source is `quali.mat`:

```
corr_imagesafterscan = [0.769 -2.182;  3.207 -1.429;  0 0]   (bin-2 px)
rot_positions        = [0 90 180],  index [0 2000 4000]
```

[`estimate_motion.py --validate`](estimate_motion.py) re-measures the same drift
independently, by correlating the three post-scan retakes against the scan
frames they repeat. It **confirms the vertical and contradicts the horizontal**:

| column | measured from the retakes | ESRF's file |
|---|---|---|
| y (vertical) | 1.572 px ptp | 1.620 px ptp |
| x (horizontal) | **0.080 ± 0.053 px ptp** | **2.109 px ptp** |

Agreement to 3 % in y is a real cross-validation of both. The x column of
`correct_motion.txt` therefore carries something besides sample drift — most
plausibly ESRF's own axis correction, which their pipeline folds into the same
file. **We use their file verbatim**, exactly as their pipeline does, and the
rotation centre below was measured *with it installed*, so the two are
consistent. Do not mix one with the other. A purely retake-derived alternative
is written to `./correct_motion.txt` by the `--validate` run if a reconstruction
ever shows horizontal streaking that points back here; step 3 would pick it up
only if it were copied into `<pfile>_2_/`.

![drift](motion_estimate.png)

## Shrinkage: off, and measured to be off

`rho[tp]=0` at every level. There is no `shrink_list.mat`, so
`load_shrink_from_mats` starts the linear model at A=B=0 regardless — but the
reason it is also *frozen* there is a measurement, not a default.

[`estimate_shrink.py`](estimate_shrink.py) reads it off the scan's own post-scan
retakes. A sample that shrank makes the retake a **scaled** copy of the frame it
repeats, so the residual displacement field between them stops being a constant
translation and acquires a slope; block-wise correlation plus an affine fit
reads that slope off directly.

| pair | spans | A_y | A_x |
|---|---|---|---|
| 4002 (ω=0) vs frame 0 | whole scan | **+107 ± 210 ppm** | +3 ± 141 ppm |
| 4001 (ω=90) vs frame 2000 | second half | +54 ± 294 ppm | −30 ± 389 ppm |

Nothing, at a 2σ limit of **\|A\| < 420 ppm** over the whole scan — 0.86 px at
the frame edge. The two pairs span different fractions of the scan, so a linear
shrink would have to show twice as much at ω=0 as at ω=90; they agree to 0.1σ.

![shrinkage](shrink_estimate.png)

In the figure the measured block displacements (blue) are ~0.5–1 px and point
in random directions, while the fitted affine part (red) is negligible: there is
no coherent radial expansion, only correlation noise.

**How much sensitivity is behind that null.** `--inject 500` rescales the retake
by a known 500 ppm before measuring. The fit responds at 7–11σ, moving from
+27/+107 ppm to **+815/+812 ppm** — so a shrinkage of this size could not have
been missed. The response is not 1:1 though: driven with synthetic frames the
same code returns ~750 ppm for an injected 500 and ~930 for an injected 1000.
That is the parabolic sub-pixel peak fit in `local_peak` inflating sub-pixel
displacements (500 ppm at a block 1200 px out is only 0.6 px). It is an
*over*-response, which makes a null result stronger rather than weaker — a
measured +107 ppm bounds the truth from above. The note above `peak_near()` in
[`estimate_shrink.py`](estimate_shrink.py) spells this out, and says what to fix
if the script is ever used to *measure* a shrinkage instead of bounding one.

For scale on what leaving `rho[tp]` free costs when there is nothing to fit: on
`../Y350a_largedisp` the optimizer invents ≈2200 ppm of edge displacement, 8×
that scan's upper limit, by absorbing displacement the position refinement
should own — [`../fig09_shrink_vs_noshrink.py`](../fig09_shrink_vs_noshrink.py)
measures the resulting 15–24 % loss. With ±300 px of random displacement, this
scan is exposed to exactly that failure mode.

## Rotation centre — measured, but weak

**`rotation_center_shift = −19.22 px`** (bin-0 detector px, convention
`c − n/2`), measured with correct_motion.txt installed:

```
python estimate_center.py config_steps15.conf --pairs 1 --motion
```

| crop | result | individual bands |
|---|---|---|
| 2048 | −19.22 ± 1.70 | −17.5, −21.6, −18.6 |
| 2560 | −20.18 ± 1.30 | −18.4, −21.6, −20.5 |
| 3072 | −18.16 ± 3.27 | −14.5, −17.6, −22.4 |

All nine bands together: mean −19.2, scatter 2.4 px. Without
`correct_motion.txt` it is −19.72 ± 1.26 — the drift barely moves it here,
unlike `../Y350a_largedisp_006nm` where the same swap was worth 2.3 px, because
this scan's horizontal drift is only 2.1 px ptp and an opposed pair sees half of
it. More bands does not help: at 5 bands the strips stop containing enough
sample and the answer falls apart (−5.4 ± 52.9).

![rotation centre](center_estimate.png)

⚠️ **Refine this before the step-6 ladder.** 4000 projections over 180° give
exactly *one* exactly-opposed pair, so there is no averaging to be had, and the
correlation peak is only about twice the noise floor (A−B RMS improves from
0.0819 to 0.0776, a 5 % gain). ±1.7 px is 14 nm at a 7.5 nm voxel — enough to
soften the reconstruction. Run steps 1–5 first, then
`python step5_center_sweep.py config_steps15.conf`, and put the winner in
`config_steps15.conf` **and** all three `config_step6_bin*.conf` before starting
bin 2.

## `nobj` = 5056

The grid has to hold the sample plus the whole displacement sweep:
4096 + 2 × 471 = 5038, rounded up to **5056 = 79 × 64**. It bins cleanly over
all three levels (2528, 1264), and leaves (5056 − 4096)/2 = 480 px of margin per
side against a worst-case 471 px displacement — 9 px of slack.

What it deliberately does **not** cover is the object-plane footprint of the
demagnified planes. Plane 4 has `eff_demag = 1/0.63662 = 1.571`, so its detector
back-maps onto 4096 × 1.571 = 6434 object px. Covering that would need
nobj = 7424: 1.47× linearly, 3.2× the volume. It is not worth it and not needed
— the outer ring of a demagnified plane looks at object coordinates the
reference plane never sees, so nothing there is constrained by the rest of the
data anyway. `mask_oob=1` drops those detector pixels from the fit. Expected
kept fractions, which `Rec._build_data_mask` logs at startup:

| plane | eff_demag | object footprint | kept |
|---|---|---|---|
| 1 | 1.000 | 4096 px | 1.000 |
| 2 | 1.043 | 4272 px | 1.000 |
| 3 | 1.215 | 4975 px | 0.87 – 1.00 (displacement-dependent) |
| 4 | 1.571 | 6434 px | 0.616 (flat — the box just translates) |

`../Y350a_HT` has run in exactly this situation since the start (same 0.637
magnification spread, nobj=4608 against a ~6500 px plane-4 footprint). Note
that `F0` keeps its `1/(ntheta*ndist*nz*n)` normalization, so `err` in
`conv.csv` scales with the kept fraction and is **not** comparable across runs
with different `nobj`.

## Running it on Polaris

```bash
qsub polaris_run.sh                                   # steps15 + bin2 + bin1 + bin0
qsub -v STAGES=steps15 polaris_run.sh                 # just steps 1-5
qsub -v CONFIG=config_step6_bin1.conf polaris_run.sh  # just one step-6 level
```

Recommended first pass, because of the rotation centre:

```bash
qsub -v STAGES=steps15 polaris_run.sh
python step5_center_sweep.py config_steps15.conf      # on a login node
# edit rotation_center_shift in all four configs
qsub -v STAGES=bin2,bin1,bin0 polaris_run.sh
```

The three step-6 levels share one `path_out`, so each seeds itself from the
previous level's checkpoint via `start_iter` and `Reader.read_checkpoint`
upsamples obj/prb/pos onto the finer grid. Iteration numbering is cumulative:

| config | bin | n | nobj | start_iter | niter |
|---|---|---|---|---|---|
| `config_step6_bin2.conf` | 2 (4×4) | 1024 | 1264 | 0 | 513 |
| `config_step6_bin1.conf` | 1 (2×2) | 2048 | 2528 | 512 | 769 |
| `config_step6_bin0.conf` | 0 (1×1) | 4096 | 5056 | 768 | 1025 |

`checkpoint_step=32` divides every `start_iter` and every `niter−1`, so the
handoff checkpoints are guaranteed to exist. Preemption is survivable: a
resubmit loses at most 32 iterations.

**Walltime.** No timings exist for this scan. The closest measured ladder is
`../Y350a_HT` (4 distances, ntheta=4000, n=4096, nobj=4608) on 2 nodes / 8
ranks: 13 min + 22 min + 1 h 47 min ≈ 2 h 25 min of step 6. Scaling by the
1.20× area ratio gives ≈2 h 55 min here. Steps 1–5 read 537 GB of EDF and write
~2.1 TB back and were never timed on any scan — that is the real unknown, and
why the script asks for 18 h. Trim it once the first `.out` file exists.

**Disk**, in the steps15 `path_out`:

| file | size |
|---|---|
| `<pfile>.h5` (4000 × 4096² × 2 B × 4 dist) | 537 GB |
| bin-0 pdata (× 4 B × 4 dist) | 1074 GB |
| `<pfile>_obj.h5` (5056³ × 4 B × 2) | 1034 GB |

≈2.7 TB, against 240 TB free on eagle as of 2026-08-31. Set
`start_level_rec=1` to stop at the 2×2 level if that changes.

## Probe

**Not wired up, and it cannot be from what was transferred.** `prb_file` is left
unset, so step 6 starts from a flat probe and refines it (`rho[1] = 0.05`).

There *are* NFP companions, one per plane —
`projections/<pfile>_NFP_before_000k.nx`, each `70 x 4096 x 4096` uint16 with
`image_key` = 50 projections + 20 darks, and each carrying its own
`source/distance` matching the four `z1` values. But those datasets are HDF5
**virtual** datasets whose sources are

    ../../../../RAW_DATA/ctxl/<pfile>/scan000{4,5}/balor_0000.h5

and `RAW_DATA/` was never copied to eagle — `/eagle/APS_IRI/vnikitin/20260829/`
contains only `ctxl/`. Every frame therefore reads back as zeros. The same is
true of the *projection* `.nx` files (`entry_0001/instrument/detector/data`,
`4063 x 4096 x 4096`, virtual, all zeros): under ewoks the `.nx` files hold
**geometry only**, and the pixels live in the `<pfile>_k_/*.edf` trees, which is
what `esrf_layout.py` reads. Do not point anything at `.nx` for image data.

So there are two prerequisites for a measured probe, in order:

1. copy `RAW_DATA/ctxl/<pfile>/scan0004` and `scan0005` (per plane) from ESRF,
   or the NFP EDFs if they exist there;
2. adapt `step0.py`, which was copied verbatim from the 2025 folders and expects
   the bliss `<pfile>_NFPwS_1_/` EDF directory. Against the ewoks layout it
   would instead read `entry_NFP_before_000k/instrument/detector/data`, split on
   `image_key` (0 = NFP, 2 = dark), and take `z1` from
   `instrument/source/distance` — note the root group is
   `entry_NFP_before_000k`, **not** `entry`.

Until then no `config_step0.conf` is provided. The commented `prb_file` line in
`config_step6_bin2.conf` is where it would go once it exists; bins 1 and 0
inherit the probe from the previous level's checkpoint, so it only ever goes in
the bin-2 config.

## `paganin` = 20

`delta/beta = 20`, from Peter's octave driver for this scan
(`<pfile>_/ht_<pfile>.m`, `delta_beta=20`) — the 2025 Y350a scans used 40–60,
but those were at 33 keV. The other settings there corroborate the config:
`nvue=4000`, `refon=4000`, `numbers=[1 2 3 4]`, `random_disp=1`,
`reference_plane=2` (hence `ref_dist=1`, 0-based), `correct_shrink=0`.

## Files

| file | what |
|---|---|
| [`esrf_layout.py`](esrf_layout.py) | **the only place that knows bliss from ewoks**; filenames + geometry |
| [`config_steps15.conf`](config_steps15.conf) | steps 1–5 |
| [`config_step6_bin{2,1,0}.conf`](config_step6_bin2.conf) | the BH ladder |
| [`polaris_run.sh`](polaris_run.sh) | PBS job, `STAGES=` selects the stages |
| [`show_geometry.py`](show_geometry.py) | prints the derived geometry and the per-level config blocks |
| [`scan_overview.py`](scan_overview.py) | the overview figure above |
| [`estimate_center.py`](estimate_center.py) | rotation centre from opposed projections |
| [`estimate_motion.py`](estimate_motion.py) | drift from the post-scan retakes |
| [`estimate_shrink.py`](estimate_shrink.py) | shrinkage from the post-scan retakes |
| [`step5_center_sweep.py`](step5_center_sweep.py) | centre refinement from the FBP volume |
| [`steps15.py`](steps15.py) | steps 1–5 driver |
| [`step6.py`](step6.py) | BH reconstruction driver |
| [`step0.py`](step0.py) | NFP probe retrieval — **bliss-only, unusable here**, see Probe |
