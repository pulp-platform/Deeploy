from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation


class SnitchRqGemmTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        if isinstance(operatorRepresentation['alpha'], float):
            assert operatorRepresentation['alpha'].is_integer()
            operatorRepresentation['alpha'] = int(operatorRepresentation['alpha'])
        if isinstance(operatorRepresentation['beta'], float):
            assert operatorRepresentation['beta'].is_integer()
            operatorRepresentation['beta'] = int(operatorRepresentation['beta'])

        #LMACAN: WARNING: Assumes rounding is expected
        add = ctxt.lookup(operatorRepresentation['add'])
        add.values += 2**(operatorRepresentation['log2D'] - 1)

        if operatorRepresentation['transB']:
            operatorRepresentation['kernelName'] = 'RQGemm_s8_transB_row_parallel_unrolled'
        else:
            operatorRepresentation['kernelName'] = 'RQGemm_s8_row_parallel_unrolled'

        return ctxt, operatorRepresentation, []


SnitchRqGemmTemplateStr = r"""
${kernelName}(${A}, ${B}, ${C}, ${data_out}, ${M}, ${N}, ${O}, ${alpha}, ${beta}, ${mul}, ${add}, ${log2D});
"""

SnitchRqGemm_Template = SnitchRqGemmTemplate(SnitchRqGemmTemplateStr)
