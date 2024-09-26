########## Stage 1: Large image to build toolchains and emulator ##########
FROM ubuntu:22.04 AS builder

ARG PYTHON_VERSION=3.8.0
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

WORKDIR /app
COPY toolchain/ toolchain/
COPY Makefile ./

RUN apt-get upgrade
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git-lfs \
    cmake \
    build-essential \
    ccache \
    ninja-build \
    pkg-config \
    ibglib2.0-dev \
    libpixman-1-dev \
    python3 \
    python-is-python3 \
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
    wget

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

# # Compile emulators
RUN make pulp-sdk
RUN make qemu
RUN make mempool
RUN make banshee

# Remove toolchain to make the container lighter
RUN rm -rf toolchain


########## Stage 2: Lightweight image with precompiled toolchain and emulators ##########
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Export symbols necessary for Deeploy's build flow 
ENV CMAKE=/usr/bin/cmake
ENV PULP_SDK_HOME=/app/install/pulp-sdk
ENV LLVM_INSTALL_DIR=/app/install/llvm
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
python-is-python3 && \
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
python get-pip.py && \
rm get-pip.py && \
pip install nvidia-pyindex && \
pip install toml-to-requirements && \
toml-to-req --toml-file pyproject.toml && \
pip install -r requirements.txt

# Copy pre-built toolchains and emulators
COPY --from=builder /app/install ./install
COPY --from=builder /root/.cargo/bin/banshee /root/.cargo/bin/banshee