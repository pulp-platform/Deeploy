# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

CarfieldGlobalInitTemplate = NodeTemplate("""
static ${type.referencedType.typeName} *${name} = (${type.referencedType.typeName} *)${address};
""")

CarfieldGlobalAllocTemplate = NodeTemplate("")

CarfieldVariableInitTemplate = NodeTemplate("""
${type.referencedType.typeName} *${name};
""")

CarfieldVariableAllocateTemplate = NodeTemplate("""
SINGLE_CORE ${name} = deeploy_malloc(sizeof(${type.referencedType.typeName}) * ${size});
""")
