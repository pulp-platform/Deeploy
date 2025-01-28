# Changelog

## Unreleased

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