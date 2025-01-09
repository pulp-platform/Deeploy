# ----------------------------------------------------------------------
#
# File: Makefile
#
# Created: 30.06.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SHELL = /usr/bin/env bash
ROOT_DIR := $(patsubst %/,%, $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

INSTALL_PREFIX        ?= install

DEEPLOY_INSTALL_DIR           ?= ${ROOT_DIR}/${INSTALL_PREFIX}
TOOLCHAIN_DIR         := ${ROOT_DIR}/toolchain

LLVM_INSTALL_DIR      ?= ${DEEPLOY_INSTALL_DIR}/llvm
LLVM_CLANG_RT_ARM      ?= ${LLVM_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/libclang_rt.builtins-armv7m.a
LLVM_CLANG_RT_RISCV_RV32IMC      ?= ${LLVM_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/rv32imc/libclang_rt.builtins-riscv32.a
LLVM_CLANG_RT_RISCV_RV32IM      ?= ${LLVM_INSTALL_DIR}/lib/clang/15.0.0/lib/baremetal/rv32im/libclang_rt.builtins-riscv32.a
PICOLIBC_ARM_INSTALL_DIR      ?= ${LLVM_INSTALL_DIR}/picolibc/arm
PICOLIBC_RISCV_INSTALL_DIR      ?= ${LLVM_INSTALL_DIR}/picolibc/riscv

PULP_SDK_INSTALL_DIR ?= ${DEEPLOY_INSTALL_DIR}/pulp-sdk
QEMU_INSTALL_DIR ?= ${DEEPLOY_INSTALL_DIR}/qemu
BANSHEE_INSTALL_DIR ?= ${DEEPLOY_INSTALL_DIR}/banshee
MEMPOOL_INSTALL_DIR ?= ${DEEPLOY_INSTALL_DIR}/mempool
SNITCH_INSTALL_DIR ?= ${DEEPLOY_INSTALL_DIR}/snitch_cluster
GVSOC_INSTALL_DIR ?= ${DEEPLOY_INSTALL_DIR}/gvsoc

CMAKE ?= cmake

LLVM_COMMIT_HASH ?= 99902b1
PICOLIBC_COMMIT_HASH ?= 31ff1b3601b379e4cab63837f253f59729ce1fef
PULP_SDK_COMMIT_HASH ?= c216298881cee767afc30928e055982b9e40e568
BANSHEE_COMMIT_HASH ?= 0e105921e77796e83d01c2aa4f4cadfa2005b4d9
MEMPOOL_COMMIT_HASH ?= affd45d94e05e375a6966af6a762deeb182a7bd6
SNITCH_COMMIT_HASH ?= e02cc9e3f24b92d4607455d5345caba3eb6273b2

RUSTUP_CARGO ?= $$(rustup which cargo)

all: toolchain emulators docs echo-bash

echo-bash:
	@echo "Please export the following symbols:"
	@echo "PULP_SDK_HOME=${PULP_SDK_INSTALL_DIR}"
	@echo "SNITCH_HOME=${SNITCH_INSTALL_DIR}"
	@echo "LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR}"
	@echo "CMAKE=$$(which cmake)"

	@echo "Please add the following paths to your PATH variable:"
	@echo "${QEMU_INSTALL_DIR}/bin"
	@echo "${BANSHEE_INSTALL_DIR}"

	@echo "For PULP to work, please source the following file:"
	@echo "${PULP_SDK_INSTALL_DIR}/configs/siracusa.sh"

	@echo ""
	@echo "TL/DR: add these lines to run ~/.bashrc"
	@echo "export PULP_SDK_HOME=${PULP_SDK_INSTALL_DIR}"
	@echo "export SNITCH_HOME=${SNITCH_INSTALL_DIR}"
	@echo "export GVSOC_INSTALL_DIR=${GVSOC_INSTALL_DIR}"
	@echo "export LLVM_INSTALL_DIR=${LLVM_INSTALL_DIR}"
	@echo "export PULP_RISCV_GCC_TOOLCHAIN=/PULP_SDK_IS_A_MESS"
	@echo "export MEMPOOL_HOME=${MEMPOOL_INSTALL_DIR}"
	@echo "export CMAKE=$$(which cmake)"
	@echo "export PATH=${QEMU_INSTALL_DIR}/bin:${BANSHEE_INSTALL_DIR}:\$$PATH"
	@echo "export PATH=~/.cargo/bin:\$$PATH"
	@echo "source ${PULP_SDK_INSTALL_DIR}/configs/siracusa.sh"


toolchain: llvm llvm-compiler-rt-riscv llvm-compiler-rt-arm picolibc-arm picolibc-riscv

emulators: snitch_runtime pulp-sdk qemu banshee mempool

${TOOLCHAIN_DIR}/llvm-project:
	cd ${TOOLCHAIN_DIR} && \
	git clone https://github.com/pulp-platform/llvm-project.git \
	 -b main && \
	cd ${TOOLCHAIN_DIR}/llvm-project && git checkout ${LLVM_COMMIT_HASH} && \
	git submodule update --init --recursive && \
	git apply ${TOOLCHAIN_DIR}/llvm.patch

${LLVM_INSTALL_DIR}: ${TOOLCHAIN_DIR}/llvm-project
	cd ${TOOLCHAIN_DIR}/llvm-project && \
	mkdir -p build && cd build && \
	${CMAKE} -G Ninja \
	-DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR} \
	-DLLVM_ENABLE_PROJECTS="clang;lld" \
	-DLLVM_TARGETS_TO_BUILD="ARM;RISCV;host" \
	-DLLVM_BUILD_DOCS="0" \
	-DLLVM_ENABLE_BINDINGS="0" \
	-DLLVM_ENABLE_TERMINFO="0" \
	-DLLVM_OPTIMIZED_TABLEGEN=ON \
	-DLLVM_PARALLEL_LINK_JOBS=2 \
	-DCMAKE_BUILD_TYPE=Release \
	../llvm && \
	${CMAKE} --build . -j && \
	${CMAKE} --install .

llvm: ${LLVM_INSTALL_DIR}

${LLVM_CLANG_RT_RISCV_RV32IMC}: ${TOOLCHAIN_DIR}/llvm-project
	cd ${TOOLCHAIN_DIR}/llvm-project && mkdir -p build-compiler-rt-riscv-rv32imc \
	&& cd build-compiler-rt-riscv-rv32imc; \
	${CMAKE} ../compiler-rt \
	-DCMAKE_C_COMPILER_WORKS=1 \
	-DCMAKE_CXX_COMPILER_WORKS=1 \
	-DCMAKE_AR=${LLVM_INSTALL_DIR}/bin/llvm-ar \
	-DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}/lib/clang/15.0.0 \
	-DCMAKE_ASM_COMPILER_TARGET="riscv32-unknown-elf" \
	-DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
	-DCMAKE_ASM_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
	-DCMAKE_C_FLAGS="-mno-relax -march=rv32imc" \
	-DCMAKE_SYSTEM_NAME=baremetal \
	-DCMAKE_HOST_SYSTEM_NAME=baremetal \
	-DCMAKE_C_COMPILER_TARGET="riscv32-unknown-elf" \
	-DCMAKE_CXX_COMPILER_TARGET="riscv32-unknown-elf" \
	-DCMAKE_SIZEOF_VOID_P=4 \
	-DCMAKE_NM=${LLVM_INSTALL_DIR}/bin/llvm-nm \
	-DCMAKE_RANLIB=${LLVM_INSTALL_DIR}/bin/llvm-ranlib \
	-DCOMPILER_RT_BUILD_BUILTINS=ON \
	-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
	-DCOMPILER_RT_BUILD_MEMPROF=OFF \
	-DCOMPILER_RT_BUILD_PROFILE=OFF \
	-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
	-DCOMPILER_RT_BUILD_XRAY=OFF \
	-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
	-DCOMPILER_RT_BAREMETAL_BUILD=ON \
	-DCOMPILER_RT_OS_DIR="baremetal/rv32imc" \
	-DLLVM_CONFIG_PATH=${LLVM_INSTALL_DIR}/bin/llvm-config && \
	${CMAKE} --build . -j && \
	${CMAKE} --install .

${LLVM_CLANG_RT_RISCV_RV32IM}: ${TOOLCHAIN_DIR}/llvm-project
	cd ${TOOLCHAIN_DIR}/llvm-project && mkdir -p build-compiler-rt-riscv-rv32im \
	&& cd build-compiler-rt-riscv-rv32im; \
	${CMAKE} ../compiler-rt \
	-DCMAKE_C_COMPILER_WORKS=1 \
	-DCMAKE_CXX_COMPILER_WORKS=1 \
	-DCMAKE_AR=${LLVM_INSTALL_DIR}/bin/llvm-ar \
	-DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}/lib/clang/15.0.0 \
	-DCMAKE_ASM_COMPILER_TARGET="riscv32-unknown-elf" \
	-DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
	-DCMAKE_ASM_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
	-DCMAKE_C_FLAGS="-mno-relax -march=rv32im" \
	-DCMAKE_SYSTEM_NAME=baremetal \
	-DCMAKE_HOST_SYSTEM_NAME=baremetal \
	-DCMAKE_C_COMPILER_TARGET="riscv32-unknown-elf" \
	-DCMAKE_CXX_COMPILER_TARGET="riscv32-unknown-elf" \
	-DCMAKE_SIZEOF_VOID_P=4 \
	-DCMAKE_NM=${LLVM_INSTALL_DIR}/bin/llvm-nm \
	-DCMAKE_RANLIB=${LLVM_INSTALL_DIR}/bin/llvm-ranlib \
	-DCOMPILER_RT_BUILD_BUILTINS=ON \
	-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
	-DCOMPILER_RT_BUILD_MEMPROF=OFF \
	-DCOMPILER_RT_BUILD_PROFILE=OFF \
	-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
	-DCOMPILER_RT_BUILD_XRAY=OFF \
	-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
	-DCOMPILER_RT_BAREMETAL_BUILD=ON \
	-DCOMPILER_RT_OS_DIR="baremetal/rv32im" \
	-DLLVM_CONFIG_PATH=${LLVM_INSTALL_DIR}/bin/llvm-config && \
	${CMAKE} --build . -j && \
	${CMAKE} --install .

llvm-compiler-rt-riscv: ${LLVM_CLANG_RT_RISCV_RV32IM} ${LLVM_CLANG_RT_RISCV_RV32IMC}

${LLVM_CLANG_RT_ARM}: ${TOOLCHAIN_DIR}/llvm-project
	cd ${TOOLCHAIN_DIR}/llvm-project && mkdir -p build-compiler-rt-arm \
	&& cd build-compiler-rt-arm; \
	${CMAKE} ../compiler-rt \
	-DCMAKE_C_COMPILER_WORKS=1 \
	-DCMAKE_CXX_COMPILER_WORKS=1 \
	-DCMAKE_AR=${LLVM_INSTALL_DIR}/bin/llvm-ar \
	-DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_DIR}/lib/clang/15.0.0 \
	-DCMAKE_ASM_COMPILER_TARGET="armv7m-none-eabi" \
	-DCMAKE_C_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
	-DCMAKE_ASM_COMPILER=${LLVM_INSTALL_DIR}/bin/clang \
	-DCMAKE_C_FLAGS="-mcpu=cortex-m4 "\
	-DCMAKE_SYSTEM_NAME=baremetal \
	-DCMAKE_HOST_SYSTEM_NAME=baremetal \
	-DCMAKE_C_COMPILER_TARGET="armv7m-none-eabi" \
	-DCMAKE_CXX_COMPILER_TARGET="armv7m-none-eabi" \
	-DCMAKE_SIZEOF_VOID_P=4 \
	-DCMAKE_NM=${LLVM_INSTALL_DIR}/bin/llvm-nm \
	-DCMAKE_RANLIB=${LLVM_INSTALL_DIR}/bin/llvm-ranlib \
	-DCOMPILER_RT_BUILD_BUILTINS=ON \
	-DCOMPILER_RT_BUILD_LIBFUZZER=OFF \
	-DCOMPILER_RT_BUILD_MEMPROF=OFF \
	-DCOMPILER_RT_BUILD_PROFILE=OFF \
	-DCOMPILER_RT_BUILD_SANITIZERS=OFF \
	-DCOMPILER_RT_BUILD_XRAY=OFF \
	-DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON \
	-DCOMPILER_RT_BAREMETAL_BUILD=ON \
	-DCOMPILER_RT_OS_DIR="baremetal" \
	-DLLVM_CONFIG_PATH=${LLVM_INSTALL_DIR}/bin/llvm-config && \
	${CMAKE} --build . -j && \
	${CMAKE} --install .

llvm-compiler-rt-arm: ${LLVM_CLANG_RT_ARM}

${TOOLCHAIN_DIR}/picolibc:
	cd ${TOOLCHAIN_DIR} && \
	git clone https://github.com/picolibc/picolibc.git && \
	cd ${TOOLCHAIN_DIR}/picolibc && git checkout ${PICOLIBC_COMMIT_HASH} && \
	git submodule update --init --recursive

${PICOLIBC_ARM_INSTALL_DIR}: ${TOOLCHAIN_DIR}/picolibc
	cd ${TOOLCHAIN_DIR}/picolibc && mkdir -p build-arm && cd build-arm && \
	cp ${TOOLCHAIN_DIR}/meson-build-script-arm.txt ../scripts && \
	PATH=${LLVM_INSTALL_DIR}/bin:${PATH} meson setup --reconfigure -Dincludedir=include \
	-Dlibdir=lib \
	-Dspecsdir=none \
	-Dmultilib=false \
	--prefix ${PICOLIBC_ARM_INSTALL_DIR} \
	--cross-file ../scripts/meson-build-script-arm.txt && \
	PATH=${LLVM_INSTALL_DIR}/bin:${PATH} meson install

picolibc-arm: ${PICOLIBC_ARM_INSTALL_DIR}

${PICOLIBC_RISCV_INSTALL_DIR}: ${TOOLCHAIN_DIR}/picolibc
	cd ${TOOLCHAIN_DIR}/picolibc && mkdir -p build-riscv && cd build-riscv && \
	cp ${TOOLCHAIN_DIR}/meson-build-script-riscv.txt ../scripts && \
	PATH=${LLVM_INSTALL_DIR}/bin:${PATH} meson setup --reconfigure -Dincludedir=include \
	-Dlibdir=lib \
	-Dspecsdir=none \
	-Dmultilib=false \
	--prefix ${PICOLIBC_RISCV_INSTALL_DIR} \
	--cross-file ../scripts/meson-build-script-riscv.txt && \
	PATH=${LLVM_INSTALL_DIR}/bin:${PATH} meson install

picolibc-riscv: ${PICOLIBC_RISCV_INSTALL_DIR}

${TOOLCHAIN_DIR}/pulp-sdk:
	cd ${TOOLCHAIN_DIR} && \
	git clone https://github.com/Scheremo/pulp-sdk.git -b scheremo && \
	cd ${TOOLCHAIN_DIR}/pulp-sdk && git checkout ${PULP_SDK_COMMIT_HASH} && \
	git submodule update --init --recursive

${PULP_SDK_INSTALL_DIR}: ${TOOLCHAIN_DIR}/pulp-sdk
	mkdir -p ${PULP_SDK_INSTALL_DIR}
	cp -r ${TOOLCHAIN_DIR}/pulp-sdk/ ${PULP_SDK_INSTALL_DIR}/../
	cd ${PULP_SDK_INSTALL_DIR} && \
	source configs/siracusa.sh && \
	make build

pulp-sdk: ${PULP_SDK_INSTALL_DIR}

${TOOLCHAIN_DIR}/snitch_cluster:
	cd ${TOOLCHAIN_DIR} && \
	git clone https://github.com/pulp-platform/snitch_cluster.git && \
	cd ${TOOLCHAIN_DIR}/snitch_cluster && git checkout ${SNITCH_COMMIT_HASH} && \
	git submodule update --init --recursive && \
	git checkout ${SNITCH_COMMIT_HASH} && git apply ${TOOLCHAIN_DIR}/snitch_cluster.patch

${SNITCH_INSTALL_DIR}: ${TOOLCHAIN_DIR}/snitch_cluster
	mkdir -p ${SNITCH_INSTALL_DIR}
	cp -r ${TOOLCHAIN_DIR}/snitch_cluster/ ${SNITCH_INSTALL_DIR}/../
	cd ${SNITCH_INSTALL_DIR} && \
	[ -d /usr/pack/riscv-1.0-kgf/pulp-llvm-0.12.0/bin ] && export LLVM_BINROOT=/usr/pack/riscv-1.0-kgf/pulp-llvm-0.12.0/bin || \
	export LLVM_BINROOT=$(dir $(shell which clang) ) && \
	mkdir tmp && \
	TMPDIR=tmp pip install -r python-requirements.txt && rm -rf tmp && \
	bender vendor init && \
	cd ${SNITCH_INSTALL_DIR}/target/snitch_cluster && \
	make sw/runtime/banshee sw/runtime/rtl sw/math

snitch_runtime: ${SNITCH_INSTALL_DIR}

${TOOLCHAIN_DIR}/gvsoc:
	cd ${TOOLCHAIN_DIR} && \
	git clone https://github.com/gvsoc/gvsoc.git && \
	cd ${TOOLCHAIN_DIR}/gvsoc &&  \
	git submodule update --init --recursive && \
	pip install -r core/requirements.txt && pip install -r gapy/requirements.txt

${GVSOC_INSTALL_DIR}: ${TOOLCHAIN_DIR}/gvsoc
	cd ${TOOLCHAIN_DIR}/gvsoc && \
	make all TARGETS=pulp.snitch.snitch_cluster_single INSTALLDIR=${GVSOC_INSTALL_DIR}

gvsoc: ${GVSOC_INSTALL_DIR}

${TOOLCHAIN_DIR}/qemu:
	cd ${TOOLCHAIN_DIR} && \
	git clone https://github.com/qemu/qemu.git --depth 1 -b stable-6.1 && \
	cd ${TOOLCHAIN_DIR}/qemu && \
	git submodule update --init --recursive

${QEMU_INSTALL_DIR}: ${TOOLCHAIN_DIR}/qemu
	cd ${TOOLCHAIN_DIR}/qemu/ && \
	mkdir -p build && cd build && \
	../configure --target-list=arm-softmmu,arm-linux-user,riscv32-softmmu,riscv32-linux-user \
	--prefix=${QEMU_INSTALL_DIR} && \
	make -j && \
	make install

qemu: ${QEMU_INSTALL_DIR}

${TOOLCHAIN_DIR}/banshee:
	cd ${TOOLCHAIN_DIR} && \
	git clone https://github.com/pulp-platform/banshee.git && \
	cd ${TOOLCHAIN_DIR}/banshee && git checkout ${BANSHEE_COMMIT_HASH} && \
	git submodule update --init --recursive && \
	git apply ${TOOLCHAIN_DIR}/banshee.patch

${BANSHEE_INSTALL_DIR}: ${TOOLCHAIN_DIR}/banshee
	export LLVM_SYS_150_PREFIX=${LLVM_INSTALL_DIR} && \
	cd ${TOOLCHAIN_DIR}/banshee/ && \
	${RUSTUP_CARGO} clean && \
	${RUSTUP_CARGO} install --path . -f

banshee: ${BANSHEE_INSTALL_DIR}

mempool: ${MEMPOOL_INSTALL_DIR}

${TOOLCHAIN_DIR}/mempool:
	cd ${TOOLCHAIN_DIR} && \
	git clone https://github.com/Xeratec/mempool.git && \
	cd ${TOOLCHAIN_DIR}/mempool && git checkout ${MEMPOOL_COMMIT_HASH}

${MEMPOOL_INSTALL_DIR}: ${TOOLCHAIN_DIR}/mempool
	mkdir -p ${MEMPOOL_INSTALL_DIR}/software && \
	cd ${TOOLCHAIN_DIR}/mempool && \
	cp -r ${TOOLCHAIN_DIR}/mempool/software/runtime ${MEMPOOL_INSTALL_DIR}/software

.PHONY: docs clean-docs format

format:
	python scripts/run_clang_format.py -e "*/third_party/*" -e "*/install/*" -e "*/toolchain/*" -ir ./ scripts --clang-format-executable=${LLVM_INSTALL_DIR}/bin/clang-format
	autoflake -i -r --remove-all-unused-imports --ignore-init-module-imports --exclude "*/third_party/**" ./
	yapf -ipr -e "third_party/" -e "install/" -e "toolchain/" .
	isort --sg "**/third_party/*"  --sg "install/*" --sg "toolchain/*" ./

docs:
	make -C docs html
clean-docs:
	rm -rf docs/_autosummary docs/_build
