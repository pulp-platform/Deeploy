# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
uint32_t compute_num = snrt_cluster_compute_core_num();

% if transB:
gemm_fp32_transB_opt(${M} / compute_num, ${O}, ${N}, ${A}, ${N} * compute_num, ${B}, ${N}, ${C}, ${O} * compute_num, ${data_out}, 1, 1 );
% else:
gemm_fp32_opt(${M} / compute_num, ${O}, ${N}, ${A}, ${N} * compute_num, ${B}, ${O}, ${C}, ${O} * compute_num, ${data_out}, 1, 1 );
%endif
""")
