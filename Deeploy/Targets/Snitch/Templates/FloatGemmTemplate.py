from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class SnitchFloatGemmTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        if (operatorRepresentation['transB']):
            operatorRepresentation['kernelName'] = 'gemm_fp32_transB_opt'
        else:
            operatorRepresentation['kernelName'] = 'gemm_fp32_opt'
        return ctxt, operatorRepresentation, []


SnitchFloatGemmTemplateStr = r"""
    uint32_t compute_num = snrt_cluster_compute_core_num();
    uint32_t ldA = ${N} * compute_num;
    uint32_t ldB = ${O};
    uint32_t ldC = ${O} * compute_num;
    
    ${kernelName}( ${M} / compute_num, ${O}, ${N}, ${A}, ldA, ${B}, ldB, ${C}, ldC, ${data_out}, 1, 1);
"""
SnitchFloatGemm_Template = SnitchFloatGemmTemplate(SnitchFloatGemmTemplateStr)
