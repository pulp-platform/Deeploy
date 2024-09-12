# Quickstart

Even though Deeploy is a pure Python library, it uses system dependencies, including a [LLVM](https://llvm.org/) cross-compiler, to test its code generation. Deeploy's testing framework further uses [picolibc](https://github.com/picolibc/picolibc) for embedded `libc` implementations and [CMake](https://cmake.org/) for its testing build flow.

Deeploy's embedded platform targets support software emulators, in the case of [ARM Cortex-M](https://www.arm.com/products/silicon-ip-cpu/cortex-m/cortex-m4) we use [QEMU](https://www.qemu.org/), for [MemPool](https://github.com/pulp-platform/mempool) and the [Snitch Cluster](https://github.com/pulp-platform/snitch_cluster) we use [Banshee](https://github.com/pulp-platform/banshee). For the PULP-Open, N-EUREKA, and Siracusa targets, we use GVSoC within the [PULP-SDK](https://github.com/pulp-platform/pulp-sdk).

To install these various dependencies, we prove instructions below, and a `Makefile` setup.

## Library Installation

From a newly setup Ubuntu 20.04 installation, you may run the following sequence to install the necessary dependencies.
For ARM64 machines, as of August 2024, `gcc-multilib` is only supported on Ubuntu 20.04. For x86_64, `gcc-multilib` should be available on most distributions.

### Installing system dependencies

```
sudo apt install git git-lfs cmake build-essential ccache ninja-build pkg-config libglib2.0-dev libpixman-1-dev cargo python3 python-is-python3 curl protobuf-compiler libftdi-dev libftdi1 doxygen libsdl2-dev scons gtkwave libsndfile1-dev rsync autoconf automake texinfo libtool libsdl2-ttf-dev
```

In case you work on an x86_64 machine, please also install `gcc-multilib`:
```
sudo apt install gcc-multilib
```

In case you work on an ARM64 machine, please install `gcc-multilib-arm-linux-gnueabi`:
```
sudo apt install gcc-multilib-arm-linux-gnueabi
export $C_INCLUDE_PATH:/usr/include:/usr/include/aarch64-linux-gnu:$C_INCLUDE_PATH
```

Other ISA/OS combinations might work, but your mileage may vary.

### Bootstrapping pip

```
curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
python get-pip.py
rm get-pip.py
export PATH=~/.local/bin:$PATH
```

### Installing Deeploy

```
pip install -e .
```

## Testing Framework Installation

Please make sure to use a Rust version that is compatible with LLVM 15, like 1.63.0:

```
sudo snap install rustup --classic
rustup install 1.63.0
rustup default 1.63.0
```

The Makefile expects the environemt variable `CMAKE` to be defined. In case you have no strong preferences, you may run

```
export CMAKE=$(which cmake)
```

to achieve this.

Finally, you should be able to run

```
make all
```

to build all Deeploy dependencies. Make sure to run

```
make echo-bash
```

to get instructions for setting up your environment.

## Getting Started

To get started with Deeploy, you can run any of the regression tests in `deeploytest`.
For example, you can run

```
cd DeeployTest
python testRunner_generic.py -t Tests/simpleRegression
```

to run the `simpleRegression` test on your workstation. Various other tests are available and compatibility between tests and platforms is tested in the `.gitlab-ci.yml` file.
