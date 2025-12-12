# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(GAP_SDK_HOME $ENV{GAP_SDK_HOME})

set(GAP9_SDK_INCLUDES
  ${GAP_SDK_HOME}/rtos/pmsis/include
  ${GAP_SDK_HOME}/rtos/pmsis/api/include
  ${GAP_SDK_HOME}/rtos/pmsis/api/include/pmsis/chips/gap9
  ${GAP_SDK_HOME}/rtos/pmsis/api/include/pmsis/implem/gap9
  ${GAP_SDK_HOME}/rtos/pmsis/pmsis_bsp/include
  ${GAP_SDK_HOME}/rtos/freeRTOS/vendors/gwt/gap9/pmsis/include
  ${GAP_SDK_HOME}/rtos/freeRTOS/vendors/gwt/gap9/config
  ${GAP_SDK_HOME}/rtos/freeRTOS/freertos_kernel/include
)

set(GAP9_SDK_COMPILE_FLAGS
  -D__riscv__
  -D__GAP9__
  -DCHIP=GAP9_V2
  -DCONFIG_GAP9_V2
)

set(PULP_SDK_INCLUDES ${GAP9_SDK_INCLUDES})
set(PULP_SDK_COMPILE_FLAGS ${GAP9_SDK_COMPILE_FLAGS})
