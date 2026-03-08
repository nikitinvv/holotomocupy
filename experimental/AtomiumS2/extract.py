import h5py
import numpy as np
import glob
import os
import sys

def extract_middle_slices(path_out, out_file):
    """Extract the middle slice of obj_re from all checkpoints and save to a single h5 file."""
    checkpoints = sorted(glob.glob(os.path.join(path_out, "checkpoint_*.h5")))
    if not checkpoints:
        print(f"No checkpoints found in {path_out}")
        sys.exit(1)

    # determine middle slice indices from first checkpoint
    with h5py.File(checkpoints[0], 'r') as f:
        nzobj, nobj, _ = f['obj_re'].shape
        mid_h = nzobj // 2   # horizontal (z) middle
        mid_v = nobj  // 2   # vertical middle
        has_im = 'obj_im' in f

    # determine number of probe distances
    with h5py.File(checkpoints[0], 'r') as f:
        ndist = f['prb_abs'].shape[0]

    slices_re_h, slices_re_v = [], []
    slices_im_h, slices_im_v = [], []
    prb_abs   = [[] for _ in range(ndist)]
    prb_phase = [[] for _ in range(ndist)]
    iters = []
    for cp in checkpoints:
        with h5py.File(cp, 'r') as f:
            slices_re_h.append(f['obj_re'][mid_h])
            slices_re_v.append(f['obj_re'][:, mid_v, :])
            if has_im:
                slices_im_h.append(f['obj_im'][mid_h])
                slices_im_v.append(f['obj_im'][:, mid_v, :])
            for k in range(ndist):
                prb_abs[k].append(f['prb_abs'][k])
                prb_phase[k].append(f['prb_phase'][k])
            iters.append(int(f.attrs['iter']))

    with h5py.File(out_file, 'w') as f:
        ds = f.create_dataset('obj_re_mid_h', data=np.stack(slices_re_h, axis=0), dtype='float32')
        ds.attrs['mid_slice'] = mid_h
        ds = f.create_dataset('obj_re_mid_v', data=np.stack(slices_re_v, axis=0), dtype='float32')
        ds.attrs['mid_slice'] = mid_v
        if has_im:
            ds = f.create_dataset('obj_im_mid_h', data=np.stack(slices_im_h, axis=0), dtype='float32')
            ds.attrs['mid_slice'] = mid_h
            ds = f.create_dataset('obj_im_mid_v', data=np.stack(slices_im_v, axis=0), dtype='float32')
            ds.attrs['mid_slice'] = mid_v
        for k in range(ndist):
            f.create_dataset(f'prb_abs{k}',   data=np.stack(prb_abs[k],   axis=0), dtype='float32')
            f.create_dataset(f'prb_phase{k}', data=np.stack(prb_phase[k], axis=0), dtype='float32')
        f.create_dataset('iter', data=np.array(iters, dtype='int32'))

    print(f"Saved {len(checkpoints)} checkpoints (z={mid_h}, y={mid_v}) → {out_file}")


if __name__ == "__main__":
    path_out = "/data2/vnikitin/atomium_rec/20240924/AtomiumS2/test1"
    out_file  = os.path.join(path_out, "mid_slices.h5")
    extract_middle_slices(path_out, out_file)
