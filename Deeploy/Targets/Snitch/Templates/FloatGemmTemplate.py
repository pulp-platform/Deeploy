from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
uint32_t compute_num = snrt_cluster_compute_core_num();
                
% if transB:
gemm_fp32_transB_opt(${M} / compute_num, ${O}, ${N}, ${A}, ${N} * compute_num, ${B}, ${N}, ${C}, ${O} * compute_num, ${data_out}, 1, 1 );
% else:                                 
gemm_fp32_opt(${M} / compute_num, ${O}, ${N}, ${A}, ${N} * compute_num, ${B}, ${O}, ${C}, ${O} * compute_num, ${data_out}, 1, 1 );
%endif
""")
