# Changelog

## Unreleased

## Improve Documentation and VSCode Support

### Added
- Tutorial section in the documentation
- Guide on using the debug print topology pass and code transformation
- VSCode configuration files for improved IDE support
-Multi-branch GitHub Pages deployment support

### Changed
- Updated `README.md` description to use a persistent development container
- Symlinking of the latest build and source files into `TEST_RECENT`
- Disabled `CMAKE_VERBOSE_MAKEFILE` by default for cleaner builds.

### Fixed
- Removed unsupported `-MMD` compiler flag in LLVM-based toolchains.

## Fix Debug Print Topology Pass and Code Transformation

### Added
- Test for the `DebugPrintTopologyPass`.
- Test for `PrintInputGeneration`, `PrintOutputGeneration`, `MemoryAwarePrintInputGeneration`, `MemoryAwarePrintOutputGeneration`

### Changed
- Refactored` IntrospectiveCodeTransformationMixIn` to allow extracting dynamic references to global variables

### Fixed
- Fix `DebugPrint` topology pass
- Fix `PrintInput` code transformations to work with global variables

## Bunch of fixes and changes

### Added
- check for CMAKE variable and fallback to searching for cmake
- tensor name mangling
- identity operation removal
- _unpack_const helper function to NodeParser to allow for node attributes that are direct Constant tensors or direct numpy values
- load_file_to_local in dory_mem as a way to load values directly to a local memory (not ram). needed for copying values from flash to wmem needed for Neureka v2

### Changed
- duplicateConstants now also duplicates constant nodes
- check float output define in DeeployTest Generic platform
- kernel_shape now inferred from weight shape if not present as per ONNX spec
- USE_NEUREKA moved into TargetLibraries where it's closer to pulp-nnx
- hex dumping logic for pulp platforms in prep for neureka v2 where I need to save weights to flash and move them during runtime to wmem
- add_gvsoc_emulation macro now requires an additional target argument and abstracted adding flags to gvsoc through the GVSOC_EXTRA_FLAGS variable

### Fixed
- RequantShift when log2d is 0
- missing math.h headers
- clang on mac doesn't support `-Wl,--gc-sections` flag, moved it into each target and for host it's checking now for host system
- `--ffast-math` caused numerical errors on generic so moved into each target and removed from that one since I'm imagining it as the _debug_ target
- Gather kernel on generic target


## Deploy Documentation with Sphinx and GitHub Pages

### Added
- Add the `documentation.yml` workflow to deploy doc pages.

### Changed
- Updated `README.md` with direct link to the documentation page.


## Fix Generic Softmax Kernel

### Fixed
- Fix broken softmax kernel for generic platform ([#2](https://github.com/pulp-platform/Deeploy/pull/2)).


## Minor CI and Readme Improvements

### Added
- Improved README with more detailed `Getting Started` section, a section listing related publications, and a list of supported platforms.
- Schedule a CI run every 6 days at 2AM CET to refresh the cache (it expires after 7 days if unused).
### Fixed
- Update the link of the Docker container used to run the CI with the Docker published by this repo instead of my fork.
- Add a retry on timeout step for large network tests. This is a temporary fix to address the sporadic freeze happening at the compilation stage, see [this issue](https://github.com/pulp-platform/Deeploy/issues/9).


## Floating Point Support

### Added
- Add the `FloatImmediate` `AbstractType`
- Define fp64, fp32, fp16, and bf16
- Add float binding for the Adder in the Generic platform
- Add a FloatAdder test to the CI for Siracusa and Generic platforms
- Extend `testType.py` with float tests
- LIMITATION: Current LLVM compiler does not support bfp16 and fp16, these types are commented in the library header


## Snitch Cluster Support

### Added
 - cMake Flow for the Snitch Cluster
 - Added `snitch_cluster` to Makefile
 - New Snitch platform with testing application
 - Testrunner for tiled and untiled execution (`testRunner_snitch.py`, `testRunner_tiled_snitch.py`)
 - Minimal library with CycleCounter and utility function

### Changed
- Update the Banshee's commit to include a recent PR.


## Snitch Cluster Tiling Support

### Added
- Support for single-buffered tiling from L2.
- Parsers, Templates, TypeCheckers, Layers, and TCF for the newly supported operators.
- A code transformation pass to filter DMA cores or compute cores for an `ExecutionBlock`.
- A code transformation pass to profile an `ExecutionBlock`.
- Test for single kernels, both with and without tiling.
- Adds the `--debug` flag to `cargo install` when installing Banshee to get the possibility of enabling the debug prints.
- New tests for the `snitch_cluster` platform.
- Add macros to `main.c` to disable printing and testing (convenient when running RTL simulations).

### Changed
- Add the possibility of changing the simulator when using the snitch-tiled test runner.


## GVSOC support for the Snitch Cluster Platform

### Added
- gvsoc in the Makefile and dockerfile
- cmake flow for gvsoc
- CI tests regarding Snitch run on GVSOC as well

### Changed
- Add the RTL library to the snitch_cluster build process in the Makefile, required for GVSOC simulation


## Add Float Support & Float GEMM for Generic and PULP

### Added
- Float Support for Constbuffer
- Simple Float GEMM on Generic and Pulp
- FP GEMM to CI
- FP GEMM Tiling on PULP

### Fixed
- Float bug on Testslice, CMSIS TestUtil, DivInterger
- AbstractDatayType Float Bugs

## Fix main.c Float Cast Bugs

### Added
- Add one new #define OUTPUTTYPE to testoutput.h

### Changed
Change main.c to use OUTPUTTYPE instead of float

## Add Operators for CCT Model

### Added
- Float Template, binding and parser, test for Conv2D, LayerNorm, Div, Relu, Softmax, MaxPool, Matmul, Transpose, Gelu, Mul, Reshape, Gather, Squeeze, Padding
- CCT model test to Generic Target
- Math Lib link on Generic Target

### Changed
- float infinity macro #define inf
- Signprop depend on float check and platform

### Fixed
- MaxPool Padding Extract Pass for float and interger
- Testinput, testoutput, weight type casted from double to float warning

## Add Float GEMM and Softmax for Snitch platform

### Added
- New templates for GEMM and Softmax.
- Added GEMM and Softmax to TargetLibraries, including case for GEMM with a transposed B matrix.
- Added new CI tests for GEMM and Softmax.

### Changed
- Adapted snitch Bindings and Platform files.

### Fixed
- Relaxed the error threshold between expected and actual values in deeploytest.
## Add Tiling Support to All CCT Kernels and Fix CCT Operators on Siracusa Platform for L2

### Added
- Float Bindings, Tilers of CCT kernels for Pulp Target
- Float Convolution, MaxPool Parser, Template, Kernel with HWC layout and padding integrated
- Added tiling constraints for conv gather and layernorm and exisitng constraints for other kernels
- profileuntiled arg
- CCT onnx tests with img size of 16 and 32

### Fixed
- CycleMeasure Pass for Siracusa Untiled Profilling
- GEMM Tiling Constraints transA and `transB' not supported
- MatMul layer Multi-Dimensional Input Issue
- Add Layer for Broadcasted Bias
- Resolved an issue where concatenation of float32 with f caused inf errors during code generation

## Add CODEOWNERS

### Added
- CODEOWNERS file to control who is responsible for reviewing future PRs.

## Memory Allocation Strategies and Visualization

### Added
- A visualization of the memory allocation solution generated by Deeploy at each level of memory. I use Plotpy to generate a static `html` file and save it to the `DeeployState` directory.
- An initialization strategy for the variable in the tiling to randomize the variables related to the permutation matrix.
- New interface to `testRunner_tiled_siracusa` to control the generation of the memory allocation visualization, the memory allocation strategy, and the search strategy.
- Export a new docker container with `plotpy` as dependency.

### Changed
- Removed unused `TilerAwareDeployer` class.

### Fixed
- Fixed a bug in the MemoryScheduler where the CP problem was solved more time that it was needed.

## Fix Float CCT Bugs on L3

### Added
- Added multiple CCT settings for testing.
- Added CCT L3 test to CI to ensure correctness for img size of 16 and 32.
- Added NaN check for deeploytest diff to improve result validation.

### Changed
- Regenerated CCT ONNX files without "output" & "input" in their names to avoid triggering the dumphex parser bug.
- Regenerated CCT ONNX file with 3 branches for attention, transforming the attention computation graph into three branches.
- Changed code generation for Hex output to properly handle float values.

### Fixed
- Updated printinput nodetemplate for float handling.

## Add MiniMalloc and Decouple Memory Allocation and Tiling

## Added
-  Installation and compilation flow for MiniMalloc through Makefile.
- Adapt the docker to install MiniMalloc and declare necessary symbols.
- Add the `constraintTileBuffersWithOverlappingLifetime` method to the memory scheduler to add the necessary memory constraint when we decouple memory allocation and tiling.
- Add the `minimalloc` method to the `Tiler` class. MiniMalloc comes as a precompiled cpp library using CSV for I/O. Hence, this method converts Deeploy's memory map to MiniMalloc's CSV representation, calls a subprocess to run MiniMalloc, reads the output CSV, and translates it back to Deeploy's memory map.
- Add MiniMalloc to the memory allocation strategies and add a new argument to the test runner to control the L2 size.

## Fixed
- Fix `testMVP.py` to get a proper should fail test.

## Implemented Quant Layer for Generic and Siracusa

### Added
- New `Quant` operation to handle quantization pattern in ONNX models
- Implementation for both Generic and Siracusa targets in the Deeploy framework
- Custom `QuantPatternPass` class to replace matched patterns with a single `Quant` operator
- Parser implementation in `Parsers.py` to extract quantization parameters
- C template implementation in `QuantTemplate.py` for efficient quantization
- Type checker implementation in `TypeCheckers.py` to handle bit-width and signedness

## Implemented Dequant Layer for Generic and Siracusa

### Added
- New `Dequant` operation to handle dequantization pattern in ONNX models
- Implementation for both Generic and Siracusa targets in the Deeploy framework
- Custom `DequantPatternPass` class to replace matched patterns with a single `Dequant` operator
- Parser implementation in `Parsers.py` to extract dequantization parameters
- C template implementation in `DequantTemplate.py` for efficient dequantization
- Type checker implementation in `TypeCheckers.py` to handle bit-width and signedness

## Fix L3 Bugs: DMA Struct Datatype and Maxpool Margin Error

### Added
- New Test Cases: Added and passed tests for 16×16 64 and 16×16 128 configurations to validate correctness.

### Fixed
- Maxpool Tile Calculation Error: The last dimension padding was incorrectly calculated due to L3 wraptiling solution. This has been fixed by updating serializeTilingSolution of Maxpool to avoid incorrect padding of Maxpool and prevent potential DMA 3D transfer issues of Maxpool.

- DMA 1D Copy Assertion Issue: Updated the DMA length datatype from uint16 to uint32 to avoid assertion failures when dealing with large block transfers.

## Implemented Updates for handling Quantized Linear DNN

### Added
- New `_sanitizeGraphNames` function to sanitize the names of the nodes and tensors of the graph
- Implementation for both Generic and Siracusa targets in the Deeploy framework
- Modified the binding of dequant in `Bindings.py` to handle int32 after GEMM operation

## Add CCT Classifier Training Support
### Added
- New test cases: testTrainCCT/CCT_GEMM_Weight_Bias_1_16_16_8, testFloatReduceSum, testFloatSoftmaxGrad, testFloatSoftmaxCrossEntropy, testFloatSoftmaxCrossEntropyGrad
- New kernels: SoftmaxCrossEntropy, SoftmaxCrossEntropyGrad, SoftmaxGrad, ReduceSum
- Refinements in operator parsers and computeShape logic for: Softmax, Mul, Reducesum

### Changed
- Enhanced layernorm operator to support three outputs (layernormout, mean, std) for compatibility with training-related layernormgrad in the future.
## Implemented Updates for handling Quantized Linear DNN

### Added
- New `_sanitizeGraphNames` function to sanitize the names of the nodes and tensors of the graph
- Implementation for both Generic and Siracusa targets in the Deeploy framework
- Modified the binding of dequant in `Bindings.py` to handle int32 after GEMM operation

## Add Support for CCT Last Layer Training with Dim 8-128

### Added
- Support for SoftmaxCrossEntropyLoss and SoftmaxCrossEntropyLossGrad with tiling.
- Implementation of SGD updates for CCT training.
- Test for one iteration of CCT last-layer training with dimensions from 8 to 128.

### Changed
- Modified the outputs of LayerNorm and SoftmaxCrossEntropyLoss nodes to a single output for better tiling compatibility.
- Added SGD parameter updates to the CCT training graph.

## One GVSoC to Simulate Them All

### Added
- All Banshee dependencies now have a frozen version. This improves maintainability as some packages get yanked for the old versions of Rust.
- Increase the L2 buffer size for loading files from Flash to RAM. This speeds up the simulation setup time.
- Align the GVSoC simulation command and build command for the new version.
- Bump new version of GVSoC and PULP-SDK

## One LLVM To Compile Them All

### Added
- Build flow and its Docker integration for LLVM 15 tagged `15.0.0-snitch-0.1.0'
- Picolibc build flow for v32im, v32ima, rv32imc and rv32imafd. Previously, it was only for rv32imc.
- LLVM Compiler RT for rv32im, rv32ima, and rv32imafd.
- Appropriate linking of picolibc and compiler RT.
- Build and install a flow for XTensor, XTL, and XSIMD. These libraries are used in some GVSoC models, and they used to live in the PULP SDK, as a header-only library. Keeping only the library headers in the PULP SDK makes it hard to bump new versions.

### Changed
- Officially depreciate Banshee as a simulator for Snitch Cluster in the CI. Maintaining this is a burden and unnecessary, as GVSoC is now the standard simulator. Additionally, newer versions of the Snitch runtime don't support Banshee anymore.
- Bump XTensor's version to `0.25.0` to fix a bug with Intel's SSE.
- Update snitch cluster patch to link to picolibc and add explicit target.
- Update README to include Snitch in the Getting Started and the D&T Journal.

### Removed
- Remove the link to the precompiled LLVM 12 in the `testRunner` for Snitch and in the CI.
- Remove the sourcing of the cursed PULP SDK script.

## rv32imf_xpulpv2 ISA support for Siracusa platform

### Changed
- The ISA for the Siracusa platform has been updated from rv32imc_zfinx_xpulpv2 to rv32imf_xpulpv2.
- All floating-point comparison tasks in deeploytest.c are now offloaded to Cluster 0 for execution.

## Small changes
- HOTFIX: Deeploy subdirectories installed when installing Deeploy with pip install
