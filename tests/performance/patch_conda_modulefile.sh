#!/bin/bash
# Private copy of the site `conda` modulefile with its retired version pins
# repointed at what the system actually has today.
#
#   ./patch_conda_modulefile.sh                    # newest /soft conda module
#   ./patch_conda_modulefile.sh /soft/modulefiles/conda/2025-09-25.lua
#
# Why this works: `module use` PREPENDS to MODULEPATH and Lmod resolves a bare
# `module load conda` in the first MODULEPATH directory that has a conda/, so
# ~/modulefiles/conda/<ver>.lua shadows the site file entirely.  Nothing under
# /soft is modified.
#
# Undo with:  rm -rf ~/modulefiles/conda
set -u

DEST_ROOT=${DEST_ROOT:-$HOME/modulefiles}
SITE_ROOT=${SITE_ROOT:-/soft/modulefiles}

. /etc/profile.d/z00_lmod.sh 2>/dev/null || . /usr/share/lmod/lmod/init/bash 2>/dev/null || true
module use "${SITE_ROOT}" 2>/dev/null

# ---------------------------------------------------------------- source file
SRC=${1:-}
if [ -z "${SRC}" ]; then
    SRC=$(ls -1 "${SITE_ROOT}"/conda/*.lua "${SITE_ROOT}"/conda/* 2>/dev/null \
          | grep -vE '/\.|modulerc' | sort -V | tail -1)
fi
if [ ! -f "${SRC}" ]; then
    echo "ERROR: no conda modulefile found (looked in ${SITE_ROOT}/conda)." >&2
    echo "       Pass one explicitly: $0 /soft/modulefiles/conda/<ver>.lua" >&2
    exit 1
fi
echo "source:  ${SRC}"

DEST_DIR="${DEST_ROOT}/conda"
DEST="${DEST_DIR}/$(basename "${SRC}")"
mkdir -p "${DEST_DIR}"
cp -f "${SRC}" "${DEST}"
chmod u+w "${DEST}"
echo "copy:    ${DEST}"

# ------------------------------------------------- what versions exist today?
# `module -t avail foo/` is terse and anchored, so it cannot match foo-bar/.
# --ignore_cache matters: the Lmod spider cache goes stale across maintenances.
versions() {
    module --ignore_cache --redirect -t avail "$1/" 2>/dev/null \
        | sed -e 's/(.*)//' -e 's/:$//' -e 's/[[:space:]]*$//' \
        | grep -E "^$1/[^/]+$" | sort -V -u
}

# --------------------------------------------------------- repin what is gone
# Every "name/version" argument of depends_on/load/prereq/always_load.
PINS=$(grep -oE '(depends_on|always_load|prereq|load)[[:space:]]*\(?[^)]*' "${DEST}" \
       | grep -oE '"[^"]+/[^"]+"' | tr -d '"' | sort -u)
# Tcl modulefiles quote nothing; catch those too.
PINS="${PINS}
$(grep -oE '^[[:space:]]*(module[[:space:]]+load|prereq)[[:space:]]+[A-Za-z][A-Za-z0-9_.+-]*/[0-9][A-Za-z0-9_.-]*' "${DEST}" \
  | grep -oE '[A-Za-z][A-Za-z0-9_.+-]*/[0-9][A-Za-z0-9_.-]*')"
PINS=$(printf '%s\n' "${PINS}" | grep -E '.' | sort -u)

if [ -z "${PINS}" ]; then
    echo "NOTE: no name/version pins found in the modulefile -- nothing to repin."
fi

CHANGED=0
for pin in ${PINS}; do
    name=${pin%%/*}
    have=$(versions "${name}")
    if [ -z "${have}" ]; then
        echo "  !! ${pin}: no ${name} at all on this system -- left alone, load will still fail"
        continue
    fi
    if printf '%s\n' "${have}" | grep -qx "${pin}"; then
        echo "  ok ${pin}"
        continue
    fi
    new=$(printf '%s\n' "${have}" | tail -1)
    esc=$(printf '%s' "${pin}" | sed 's/[][\.*^$/]/\\&/g')
    sed -i "s|${esc}|${new}|g" "${DEST}"
    echo "  -> ${pin}  =>  ${new}"
    CHANGED=$((CHANGED + 1))
done
echo "repinned ${CHANGED} dependenc$([ ${CHANGED} -eq 1 ] && echo y || echo ies)"

# ------------------------------------------------------------------- verify
echo
echo "verifying in a subshell:"
(
    module use "${DEST_ROOT}" 2>/dev/null
    module --ignore_cache load conda 2>&1 | sed 's/^/    /'
    if command -v python >/dev/null 2>&1; then
        echo "    OK: python = $(command -v python)"
        module -t --redirect list 2>&1 | sed 's/^/      /'
    else
        echo "    FAILED: conda module still does not load (see above)." >&2
        exit 1
    fi
)
rc=$?

cat <<TXT

Use it by prepending your modulefiles before loading conda:

    module use /soft/modulefiles
    module use \$HOME/modulefiles     # shadows the site conda/
    module load conda

Remove the override once ALCF fixes the site modulefile:

    rm -rf ${DEST_DIR}
TXT
exit ${rc}
