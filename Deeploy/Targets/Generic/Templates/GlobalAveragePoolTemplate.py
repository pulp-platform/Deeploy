from Deeploy.DeeployTypes import NodeTemplate, NetworkContext, OperatorRepresentation

class _GlobalAveragePoolTemplate(NodeTemplate):
    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation):
        return ctxt, operatorRepresentation, []

referenceTemplate = _GlobalAveragePoolTemplate("""
// GlobalAveragePool (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    ${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
    ${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};
    GlobalAveragePool_fp32_NCHW(
        ref_${data_out}_${data_in},
        ${N}, ${C}, ${H}, ${W}, ref_${data_out}_${data_out}
    );
END_SINGLE_CORE
""")
