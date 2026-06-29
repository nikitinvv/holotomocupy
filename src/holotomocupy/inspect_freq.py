#!/usr/bin/env python3
"""Inspect a 2D image's spectrum and remove the highest-frequency rows/cols.

Usage:
    python inspect_nyquist.py image.npy
    python inspect_nyquist.py image.tif --band 1
    python inspect_nyquist.py recon.h5 --dataset /exchange/data --index 12
    python inspect_nyquist.py recon.h5 --component phase --save-cleaned clean.tif
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_image(path, dataset=None, index=None):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        a = np.load(path)
    elif ext in (".tif", ".tiff"):
        try:
            import tifffile
            a = tifffile.imread(path)
        except ImportError:
            import imageio.v3 as iio
            a = iio.imread(path)
    elif ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            if dataset is None:
                for name in ("data", "exchange/data", "object", "image", "reconstruction"):
                    if name in f:
                        dataset = name
                        break
                else:
                    def first_ds(g, prefix=""):
                        for k in g:
                            full = f"{prefix}/{k}".lstrip("/")
                            if isinstance(g[k], h5py.Dataset):
                                return full
                            r = first_ds(g[k], full)
                            if r:
                                return r
                    dataset = first_ds(f)
                print(f"auto-picked dataset: {dataset}")
            a = f[dataset][...]
    else:
        raise ValueError(f"unsupported extension: {ext}")

    if a.ndim == 3:
        if index is None:
            index = a.shape[0] // 2
        print(f"3D input {a.shape} → taking slice [{index}]")
        a = a[index]
    if a.ndim != 2:
        raise ValueError(f"expected 2D after slicing, got shape {a.shape}")
    return a


def kill_high_freq(img, band=1):
    """Zero out the `band` rows/cols closest to Nyquist on each axis.

    band=1 zeros only the Nyquist row + Nyquist col (the unconstrained modes
    from a half-pixel shift). Larger band also removes near-Nyquist frequencies.
    Operates on fftshifted spectrum where DC is at the center, Nyquist at index 0.
    """
    C = np.fft.fftshift(np.fft.fft2(img))
    for k in range(band):
        if k == 0:
            C[0, :] = 0
            C[:, 0] = 0
        else:
            C[k, :] = 0
            C[-k, :] = 0
            C[:, k] = 0
            C[:, -k] = 0
    out = np.fft.ifft2(np.fft.ifftshift(C))
    return out.real if np.isrealobj(img) else out.astype(img.dtype)


def save_image(path, img):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, img)
    elif ext in (".tif", ".tiff"):
        try:
            import tifffile
            tifffile.imwrite(path, img)
        except ImportError:
            import imageio.v3 as iio
            iio.imwrite(path, img)
    else:
        raise ValueError(f"can only save .npy or .tif/.tiff, got {ext}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help=".npy, .tif/.tiff, or .h5/.hdf5")
    ap.add_argument("--dataset", help="HDF5 dataset path (auto-detected otherwise)")
    ap.add_argument("--index", type=int, default=None,
                    help="3D slice index (default: middle)")
    ap.add_argument("--component", choices=("real", "imag", "abs", "phase"),
                    default="real", help="Component to extract from complex input")
    ap.add_argument("--band", type=int, default=1,
                    help="Number of highest-|k| rows/cols to zero per axis "
                         "(1 = just Nyquist, the unconstrained modes)")
    ap.add_argument("--save-cleaned", metavar="PATH",
                    help="Write cleaned image to PATH (.npy or .tif)")
    ap.add_argument("--save-fig", metavar="PATH",
                    help="Write comparison figure to PATH (.png)")
    ap.add_argument("--no-show", action="store_true",
                    help="Don't open a window (use with --save-fig on headless nodes)")
    args = ap.parse_args()

    img = load_image(args.path, dataset=args.dataset, index=args.index)
    if np.iscomplexobj(img):
        comp = {"real": np.real, "imag": np.imag,
                "abs": np.abs, "phase": np.angle}[args.component]
        img = comp(img)
        print(f"complex input: using {args.component}")
    img = img.astype(np.float32)

    cleaned = kill_high_freq(img, band=args.band).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(img))
    F_clean = np.fft.fftshift(np.fft.fft2(cleaned))
    eps = np.float32(1e-30)

    print(f"shape: {img.shape},  range orig: [{img.min():.3g}, {img.max():.3g}],  "
          f"cleaned: [{cleaned.min():.3g}, {cleaned.max():.3g}]")
    print(f"|FFT| at Nyquist row before/after: "
          f"{np.abs(F[0]).max():.3g} / {np.abs(F_clean[0]).max():.3g}")

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(img, cmap="gray")
    ax[0, 0].set_title(f"original  {img.shape}")
    im = ax[0, 1].imshow(np.log(np.abs(F) + eps), cmap="viridis")
    ax[0, 1].set_title("log |FFT2|  (DC at center, Nyquist at edges)")
    plt.colorbar(im, ax=ax[0, 1], fraction=0.046)
    ax[1, 0].imshow(cleaned, cmap="gray")
    ax[1, 0].set_title(f"cleaned  (band={args.band} rows/cols zeroed)")
    im = ax[1, 1].imshow(np.log(np.abs(F_clean) + eps), cmap="viridis")
    ax[1, 1].set_title("log |FFT2| of cleaned")
    plt.colorbar(im, ax=ax[1, 1], fraction=0.046)
    for a in ax.flat:
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()

    if args.save_fig:
        fig.savefig(args.save_fig, dpi=150, bbox_inches="tight")
        print(f"wrote {args.save_fig}")
    if args.save_cleaned:
        save_image(args.save_cleaned, cleaned)
        print(f"wrote {args.save_cleaned}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
