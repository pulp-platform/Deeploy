export PULP_SDK_HOME=${DEEPLOY_INSTALL_DIR}/pulp-sdk
export LLVM_INSTALL_DIR=${DEEPLOY_INSTALL_DIR}/llvm
export PULP_RISCV_GCC_TOOLCHAIN=
export MEMPOOL_HOME=${DEEPLOY_INSTALL_DIR}/mempool
export CMAKE=/usr/bin/cmake
export PATH=${DEEPLOY_INSTALL_DIR}/qemu/bin:${DEEPLOY_INSTALL_DIR}/banshee:$PATH
export PATH=~/.cargo/bin:$PATH
source ${PULP_SDK_HOME}/configs/siracusa.sh
