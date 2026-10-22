
from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
Carfield_Relu(${data_in}, ${data_out}, ${size}, core_id, numThreads);
""")
