# SoCDAML Part III - TA reference solution for `iLeakyReLU`

The complete working `iLeakyReLU` operator (parser, template, binding,
mapper, tile constraint, scalar kernel, SIMD kernel, ONNX + golden
artifacts, and a one-shot deploy script). Use it to demo the lab
end-to-end, and unblock students who get stuck.

## What's in here

| File | Purpose |
|------|---------|
| `generate.py` | Builds `network.onnx`, `inputs.npz`, `outputs.npz` for the single-node test |
| `network.onnx` | Single-node ONNX with op_type `iLeakyReLU` (`mul=1`, `shift=3`), shape `(1, 16, 64, 64)` |
| `inputs.npz`  | Int8 input tensor named `data_in` |
| `outputs.npz` | Int8 golden output tensor named `data_out` |
| `iLeakyReLU.h` | Kernel header |
| `iLeakyReLU.c` | Scalar baseline kernel (Step 3) |
| `iLeakyReLU_simd.c` | XPULP SIMD kernel (Step 6b) |
| `iLeakyReLUParser.py` | Full parser class for `Deeploy/Targets/Generic/Parsers.py` |
| `iLeakyReLUTemplate.py` | Full Mako template for `Deeploy/Targets/PULPOpen/Templates/` |
| `iLeakyReLUTileConstraint.py` | Full tile + perf constraint for `Deeploy/Targets/PULPOpen/TileConstraints/` |
| `iLeakyReLU-core.patch` | The edits that wire the op into `Parsers.py`, `Bindings.py`, `Tiler.py`, `Platform.py` and `DeeployPULPMath.h` |
| `deploy.sh` | One-shot script that copies the kernel/template/constraint into the live tree and applies `iLeakyReLU-core.patch` with `git apply` |

## Quick start (TA workflow)

From this directory, inside the Singularity shell:

```bash
# 1) (Re)generate the test artifacts
python generate.py

# 2) Apply the SCALAR solution into the live source tree
./deploy.sh

# 3) Verify (all four runs should report 0 errors and the cycle counts
#    in the table below). See "Verification" section for the commands.

# 4) Swap to the SIMD kernel for Step 6b
./deploy.sh simd

# 5) Roll back everything if you ever need to clean up
./deploy.sh undo
# (reverts the core-library patch and removes the copied files)
```

The edits to Deeploy's own sources live in `iLeakyReLU-core.patch` and are
applied with `git apply`. If upstream has moved since the patch was written,
you get ordinary conflict markers to resolve rather than a half-applied tree,
and `undo` reverts the patch with `git apply --reverse`.

`deploy.sh` is idempotent, i.e. running it a second time is a no-op for
the source patches. Re-running `./deploy.sh` after `./deploy.sh simd`
will overwrite the kernel back to scalar (and vice versa), so you can
flip between the two with one command.

## Verification

Reproduce every number in the lab's "Stacked speedup" table from
`DeeployTest/`:

```bash
cd /app/Deeploy/Tutorials/PartIII_solution/iLeakyReLU
./deploy.sh
cd /app/Deeploy/DeeployTest

echo "=== Baseline (1 core, scalar, untiled) ===";  python deeployRunner_siracusa.py       -t Tests/Kernels/Integer/LeakyReLU/Regular --cores=1 2>&1 | grep -E "Runtime|Errors"
echo "=== Step 4   (8 cores, scalar, untiled) ==="; python deeployRunner_siracusa.py       -t Tests/Kernels/Integer/LeakyReLU/Regular --cores=8 2>&1 | grep -E "Runtime|Errors"
echo "=== Step 5   (8 cores, scalar, tiled)   ==="; python deeployRunner_tiled_siracusa.py -t Tests/Kernels/Integer/LeakyReLU/Regular --cores=8 --l1=32768 --defaultMemLevel=L2 2>&1 | grep -E "Runtime|Errors"

cd /app/Deeploy/Tutorials/PartIII_solution/iLeakyReLU
./deploy.sh simd
cd /app/Deeploy/DeeployTest

echo "=== Step 6   (8 cores, SIMD,   tiled)   ==="; python deeployRunner_tiled_siracusa.py -t Tests/Kernels/Integer/LeakyReLU/Regular --cores=8 --l1=32768 --defaultMemLevel=L2 2>&1 | grep -E "Runtime|Errors"
```

### Expected output

Every run reports `Errors: 0 out of 65536`. Cycle counts:

| Step | Configuration | Cycles | vs baseline |
|------|---|---|---|
| baseline | 1 core, scalar, untiled | **2 492 970** | 1.00× |
| Step 4   | 8 cores, scalar, untiled | **313 541** | 7.95× |
| Step 5   | 8 cores, scalar, tiled (`--l1=32768`) | **108 090** | 23.06× |
| Step 6   | 8 cores, SIMD, tiled (`--l1=32768`) | **43 005** | 57.97× |

If any count drifts by more than a few percent or a run reports any
errors, something in the deploy is off. Try `./deploy.sh undo`, then
re-deploy from scratch.

## Files NOT in this directory (live-tree edits applied by deploy.sh)

`deploy.sh` modifies these files in the live tree. They are NOT
duplicated here, i.e. `deploy.sh` is the source of truth.

- `Deeploy/Targets/Generic/Parsers.py`: appends `iLeakyReLUParser`
- `Deeploy/Targets/PULPOpen/Bindings.py`: appends `PULPiLeakyReLUBindings`
- `Deeploy/Targets/PULPOpen/Tiler.py`: appends `PULPiLeakyReLUTilingReadyBindings`
- `Deeploy/Targets/PULPOpen/Platform.py`: adds parser/layer imports, mapper, and `PULPMapping` entry
- `TargetLibraries/PULPOpen/inc/DeeployPULPMath.h`: adds the kernel include
