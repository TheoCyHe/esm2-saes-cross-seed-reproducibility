# Shared environment activation for Bouchet jobs. Source this, don't execute it:
#     source env/activate.sh
#
# The LD_PRELOAD line is load-bearing, not cargo cult. Bouchet's system
# libstdc++ (/usr/lib64/libstdc++.so.6.0.29) provides CXXABI up to 1.3.13, but
# conda-forge's scipy needs 1.3.15. The env ships its own libstdc++ 15.2 and
# scipy's RPATH finds it correctly *in isolation* -- so `import scipy` alone
# works. But torch loads the system libstdc++ first, and once that soname is
# resolved the loader reuses it for everything after, so `import torch` before
# scipy fails with a CXXABI_1.3.15 error. Preloading the env's copy makes the
# newer library win regardless of import order. LD_LIBRARY_PATH does NOT fix
# this -- by the time it matters the soname is already bound.

module purge
module load miniconda/24.11.3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate esm-interp

export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"

# Fixed, never per-seed: the ensemble varies ONLY the SAE seed, so hash
# randomization must not be another moving part.
export PYTHONHASHSEED=0

# Unbuffered stdout/stderr: SLURM redirects stdout to a file, which makes Python
# BLOCK-buffer it, so training progress (mse, dead count, b_dec convergence) does
# not reach the log until the buffer fills -- effectively invisible until a run
# ends. This keeps job logs live for monitoring.
export PYTHONUNBUFFERED=1

# UTF-8 for every open() and for stdout, whatever the node's locale: a record
# document written, read back under a single-byte codec and written again is
# still valid UTF-8, so nothing fails on it (CLAUDE.md gotcha 46). This is the
# backstop; passing encoding="utf-8" explicitly is still the control.
export PYTHONUTF8=1
