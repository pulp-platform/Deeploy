# Deeploy

Deeploy is a python tool to generate low-level optimized C Code for multi-cluster, heterogeneous SoCs. Its goal is to enable configurable deployment flows from a bottom-up compiler perspective, modelling target hardware in a fine-grained and modular manner.

Deeploy is developed as part of the PULP project, a joint effort between ETH Zurich and the University of Bologna.

## License

Unless specified otherwise in the respective file headers, all code checked into this repository is made available under a permissive license. All software sources and tool scripts are licensed under Apache 2.0, except for files contained in the `scripts` directory, which are licensed under the MIT license, and files contained in the `DeeployTest/Tests`directory, which are licensed under the [Creative Commons Attribution-NoDerivates 4.0 International](https://creativecommons.org/licenses/by-nd/4.0) license (CC BY-ND 4.0).

## Getting started

To install Deeploy, simply run
```
pip install -e .
```
to download and install the Python dependencies. Run
```
make docs
```
and open `docs/_build/html/index.html` for more extensive documentation & getting started guides.
