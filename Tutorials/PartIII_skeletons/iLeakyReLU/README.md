# SoCDAML Part III - Student skeletons for `iLeakyReLU`

These files are your starting points for the Part III lab. Each one
contains the surrounding boilerplate; the conceptually interesting
parts are marked with `TODO(student)` comments and short hints.

| File | What's in it | What to do |
|------|--------------|------------|
| `generate.py` | Complete ONNX + golden-value generator | Run it (Step 1) |
| `iLeakyReLU.h` | Complete kernel header | Copy to `TargetLibraries/PULPOpen/inc/kernel/` (Step 3) |
| `iLeakyReLU.c` | Multi-core chunking provided; inner loop TODO | Fill the TODO, copy to `TargetLibraries/PULPOpen/src/` (Step 3) |
| `iLeakyReLU_simd.c` | SIMD chunking + load/max/store provided; one TODO line | Fill in Step 6b after the scalar works |
| `iLeakyReLUParser.py` | `parseNode` and `parseNodeCtxt` are TODO | Fill in, paste class into `Deeploy/Targets/Generic/Parsers.py` (Step 2) |
| `iLeakyReLUTemplate.py` | Mako template body is TODO | Fill in, copy to `Deeploy/Targets/PULPOpen/Templates/` (Step 4) |
| `iLeakyReLUTileConstraint.py` | Inherits `UnaryTileConstraint`; performance constraint TODO | Fill in (Step 5 + Step 6a), copy to `Deeploy/Targets/PULPOpen/TileConstraints/` |

The **Binding** (in `Bindings.py`), **Mapper + PULPMapping entry** (in `Platform.py`), the **`TilingReadyNodeBindings` registration** (in `Tiler.py`), and the **aggregator include** in `DeeployPULPMath.h` are *not* shipped as paste-in snippets — you'll write them yourself with the markdown's guidance and `<details>` solutions.

The companion `Deeploy/Tutorials/SoCDAML.md` (Part III) walks through
the six steps in order and includes collapsed solutions to peek at
when you're stuck.

If you really need the answer key, look in
`Deeploy/Tutorials/PartIII_solution/iLeakyReLU/` -- but try the lab
first; you'll learn far more.
