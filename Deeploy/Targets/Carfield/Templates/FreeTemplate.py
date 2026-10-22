
from Deeploy.DeeployTypes import NodeTemplate

CarfieldGlobalFreeTemplate = NodeTemplate("")

# MemoryManagementGeneration already tracks liveness and inserts this right
# after the last consumer of the buffer, so filling it in is all that is needed
# to turn the bump allocator into a reusing one.
CarfieldVariableFreeTemplate = NodeTemplate("""
SINGLE_CORE deeploy_free(${name}, sizeof(${type.referencedType.typeName}) * ${size});
""")
