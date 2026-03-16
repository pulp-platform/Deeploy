from Deeploy.DeeployTypes import NodeTemplate

# allocate
referenceAllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) snrt_l1alloc(${type.referencedType.typeWidth//8} * ${size});\n")
