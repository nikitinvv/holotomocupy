#!/bin/bash
# Report what each candidate python on Polaris already provides, so the setup
# only compiles what is genuinely missing.  Read-only; touches nothing.
#
#   bash probe_polaris.sh

module use /soft/modulefiles >/dev/null 2>&1
module load PrgEnv-gnu cray-hdf5-parallel cudatoolkit-standalone >/dev/null 2>&1

echo "=== node: $(hostname)"
echo "=== nproc: soft=$(ulimit -Su) hard=$(ulimit -Hu); threads for $USER: $(ps -u "$USER" -L --no-headers 2>/dev/null | wc -l)"
printf "=== can fork a child: "
if (exec true) 2>/dev/null; then echo yes; else echo "NO -- this node is out of process slots"; fi

probe() {
    local py="$1" label="$2"
    [ -x "$py" ] || return 0
    echo
    echo "--- $label"
    echo "    $py"
    "$py" - <<'PY' 2>&1 | sed 's/^/    /'
import importlib, sys
print("python  :", sys.version.split()[0])
def line(mod, fn):
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        print(f"{mod:8}: MISSING ({type(e).__name__})"); return
    try:
        print(f"{mod:8}: {fn(m)}")
    except Exception as e:
        print(f"{mod:8}: present but broken ({type(e).__name__}: {e})")

line("numpy",  lambda m: m.__version__)
line("mpi4py", lambda m: (lambda v: f"{m.__version__} -- {v.splitlines()[0][:70]}"
                          + ("  [CRAY]" if "CRAY" in v.upper() else "  [NOT CRAY]"))(
                          __import__("mpi4py.MPI", fromlist=["MPI"]).Get_library_version().strip()))
line("h5py",   lambda m: f"{m.version.version}, hdf5 {m.version.hdf5_version}, mpi={m.get_config().mpi}"
                         + ("" if m.get_config().mpi else "   [no mpio driver]"))
line("cupy",   lambda m: f"{m.__version__}, CUDA {m.cuda.runtime.runtimeGetVersion()}")
for extra in ("scipy", "tifffile", "psutil", "nvtx", "pandas", "matplotlib"):
    line(extra, lambda m: getattr(m, "__version__", "ok"))
PY
}

SITE_CONDA=$(ls -1dt /soft/applications/conda/*/mconda3 2>/dev/null | head -1)
probe "$SITE_CONDA/bin/python"        "ALCF site conda base  ($SITE_CONDA)"
probe "$HOME/miniforge3/bin/python"   "miniforge base"
for v in "$HOME"/venvs/*/bin/python; do probe "$v" "venv $(basename "$(dirname "$(dirname "$v")")")"; done

echo
echo "=== nvcc: $(command -v nvcc || echo '<none>')"
echo "=== mathDX: ${MATHDX_ROOT:-<unset>}"
for d in /eagle/APS_IRI/vnikitin/nvidia/nvidia-mathdx-*/nvidia/mathdx/*/; do
    [ -d "$d" ] && echo "    found: $d"
done
