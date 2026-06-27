# kernels

GPU kernels (Triton now, CUDA later), each benchmarked against a PyTorch reference.
Authored on a Mac, run on a Linux + NVIDIA CUDA box.

```
src/kernels/
├── ops/         # kernel implementations
└── benchmarks/  # benchmarks vs PyTorch
```

Installed via hatchling (src-layout), so absolute imports work everywhere:

```python
from kernels.ops.rmsnorm import rmsnorm_triton
```

## Setup

Mac (authoring only):

```sh
uv sync
bash scripts/vendor-triton.sh
```

Linux + CUDA (execution):

```sh
uv sync --extra gpu
uv run python -m kernels.benchmarks.<name>
```

On the GPU box torch installs the CUDA 12.8 wheel (`pytorch-cu128` index); the Mac
gets the default CPU/MPS build. Triton has no macOS wheels, so it's gated to Linux by
a `sys_platform` marker in the `gpu` extra — `uv sync` never fails on the Mac.
`vendor-triton.sh` shallow-clones
Triton's Python source into `.triton-src/` (gitignored) for autocomplete only; rerun
it after a fresh clone or a Triton bump.

## Workflow

Author on Mac → `git push` → `git pull` on GPU box → `uv sync --extra gpu` → run.
Commit `uv.lock`.
