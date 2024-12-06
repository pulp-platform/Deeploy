from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class SnitchGemmTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        if isinstance(operatorRepresentation['alpha'], float):
            assert operatorRepresentation['alpha'].is_integer(
            ), f"Parameter alpha is not an integer: {operatorRepresentation['alpha']}"
            operatorRepresentation['alpha'] = int(operatorRepresentation['alpha'])
        if isinstance(operatorRepresentation['beta'], float):
            assert operatorRepresentation['beta'].is_integer(
            ), f"Parameter beta is not an integer: {operatorRepresentation['beta']}"
            operatorRepresentation['beta'] = int(operatorRepresentation['beta'])

        if operatorRepresentation['transB']:
            operatorRepresentation['kernelName'] = "Gemm_s8_transB_row_parallel"
        else:
            operatorRepresentation['kernelName'] = "Gemm_s8_row_parallel"

        return ctxt, operatorRepresentation, []


SnitchGemmTemplateStr = r"""
${kernelName}(${A}, ${B}, ${C}, ${data_out}, ${M}, ${N}, ${O}, ${alpha}, ${beta});
"""

SnitchGemm_Template = SnitchGemmTemplate(SnitchGemmTemplateStr)
