########## Stage 1: Large image to build toolchains and emulator ##########
FROM ubuntu:22.04 AS builder

ARG PYTHON_VERSION=3.8.0
ARG DEBIAN_FRONTEND=noninteractive

ARG UBUNTU_VERSION=22.04
ARG BENDER_VERSION=0.28.1
ARG SNITCH_LLVM_VERSION=15.0.0-snitch-0.1.0

ENV TZ=Etc/UTC

WORKDIR /app
COPY toolchain/ toolchain/
COPY Makefile ./

RUN apt-get upgrade
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git-lfs \
    build-essential \
    ccache \
    ninja-build \
    pkg-config \
    ibglib2.0-dev \
    libpixman-1-dev \
    python3 \
    python-is-python3 \
    python3.10-venv \
    python3.10-distutils \
    curl \
    protobuf-compiler \
    libftdi-dev \
    libftdi1 \
    doxygen \
    libsdl2-dev \
    scons \
    gtkwave \
    libsndfile1-dev \
    rsync \
    autoconf \
    automake \
    texinfo \
    libtool \
    libsdl2-ttf-dev \
    gcc-multilib \
    wget \
    clang-format

# Install cmake 3.31.1
RUN wget https://github.com/Kitware/CMake/releases/download/v3.31.1/cmake-3.31.1-linux-x86_64.sh && \
    chmod +x cmake-3.31.1-linux-x86_64.sh && \
    ./cmake-3.31.1-linux-x86_64.sh --prefix=/usr --skip-license

# Compile minimalloc
RUN make minimalloc

# bowwang: Install build dependencies for Python venv + ensurepip
RUN apt-get update && \
    apt-get install -y \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev \
    uuid-dev

# Install Python
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
RUN tar xzf Python-${PYTHON_VERSION}.tgz
RUN cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --prefix=/opt/python/ && \
    make install -j

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Build Rust tools
RUN apt remove cargo -y
RUN apt autoremove -y
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup install 1.63.0
RUN rustup default 1.63.0
RUN rustup component add rust-src

# Compile toolchains
RUN pip install meson
RUN make llvm
RUN make llvm-compiler-rt-riscv
RUN make llvm-compiler-rt-arm
RUN make picolibc-arm
RUN make picolibc-riscv

# Free spacce before softhier installation
# RUN rm -rf /app/toolchain/*
RUN make SOFTHIER_INSTALL_DIR=/app/install/softhier softhier && \
    rm -rf /app/install/softhier/third_party/toolchain/toolchain.tar.xz && \
    rm -rf /app/install/softhier/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7.tar.bz2 && \
    rm -rf /app/install/softhier/third_party/toolchain/v1.0.16-pulp-riscv-gcc-centos-7

# # Compile emulators
RUN make pulp-sdk
RUN make qemu
RUN make mempool
RUN make banshee
# Copy gvsoc requirements for later as they are removed when toolchain folder is deleted
RUN make xtensor
RUN make gvsoc && cp -r /app/toolchain/gvsoc/core/requirements.txt /app/core-requirements.txt && cp -r /app/toolchain/gvsoc/gapy/requirements.txt /app/gapy-requirements.txt

# Dependencies needed for compiling Snitch
## Bender's installaton
RUN wget https://github.com/pulp-platform/bender/releases/download/v${BENDER_VERSION}/bender-${BENDER_VERSION}-x86_64-linux-gnu-ubuntu${UBUNTU_VERSION}.tar.gz && \
    tar xzf bender-${BENDER_VERSION}-x86_64-linux-gnu-ubuntu${UBUNTU_VERSION}.tar.gz && cp  /app/bender /bin
ENV PATH=/app/bender:$PATH

## compile snitch
ENV LLVM_INSTALL_DIR=/app/install/llvm
RUN make snitch_runtime && rm -rf /app/snitch_cluster

# Remove toolchain to make the container lighter
RUN rm -rf toolchain

RUN echo "== Top 10 largest install subdirectories ==" && du -h /app/install | sort -hr | head -n 10

########## Stage 2: Lightweight image with precompiled toolchain and emulators ##########
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Export symbols necessary for Deeploy's build flow 
ENV CMAKE=/usr/bin/cmake
ENV PULP_SDK_HOME=/app/install/pulp-sdk
ENV LLVM_INSTALL_DIR=/app/install/llvm
ENV SNITCH_HOME=/app/install/snitch_cluster
ENV GVSOC_INSTALL_DIR=/app/install/gvsoc
ENV SOFTHIER_INSTALL_DIR=/app/install/softhier
ENV MINIMALLOC_INSTALL_DIR=/app/install/minimalloc
ENV MEMPOOL_HOME=/app/install/mempool
ENV PATH=/app/install/qemu/bin:/app/install/banshee:$PATH
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml ./

# Install dependencies
RUN mkdir -p /root/.cargo/bin/ && \ 
apt-get update && \ 
DEBIAN_FRONTEND=noninteractive apt-get install -y git-lfs \
cmake \
ccache \ 
curl \
libpixman-1-dev \
libsdl2-dev \
python-is-python3 \
python3.10-venv \
python3.10-distutils && \
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
python get-pip.py && \
rm get-pip.py && \
pip install nvidia-pyindex && \
pip install toml-to-requirements && \
toml-to-req --toml-file pyproject.toml && \
pip install -r requirements.txt

COPY --from=builder /app/core-requirements.txt ./core-requirements.txt
COPY --from=builder /app/gapy-requirements.txt ./gapy-requirements.txt
RUN pip install -r core-requirements.txt -r gapy-requirements.txt

# Copy pre-built toolchains and emulators
COPY --from=builder /app/install ./install
COPY --from=builder /root/.cargo/bin/banshee /root/.cargo/bin/banshee