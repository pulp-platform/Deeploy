from Deeploy.DeeployTypes import NodeTemplate

# allocate
referenceAllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) snrt_l1alloc(${type.referencedType.typeWidth//8} * ${size});\n")

spatzGenericAllocate = NodeTemplate("""
% if _memoryLevel == "L1":
${name} = (${type.typeName}) snrt_l1alloc(sizeof(${type.referencedType.typeName}) * ${size});\n
% elif _memoryLevel == "L3" or _memoryLevel is None:
${name} = (${type.typeName}) snrt_l3alloc(sizeof(${type.referencedType.typeName}) * ${size});\n
% else:
// COMPILER WARNING — unsupported memory level ${_memoryLevel}, defaulting to L3                                                                
${name} = (${type.typeName}) snrt_l3alloc(${type.referencedType.typeWidth//8} * ${size});                                                       
% endif 
""")
