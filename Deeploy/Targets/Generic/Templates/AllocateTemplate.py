# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceInitTemplate = NodeTemplate("${type.typeName} ${name};\n")
referenceAllocateTemplate = NodeTemplate(
    "${name} = (${type.typeName}) deeploy_malloc(${type.referencedType.typeWidth//8} * ${size});\n")

referenceGlobalInitTemplate = NodeTemplate("static ${type.referencedType.typeName} ${name}[${size}] = {${values}};\n")

referenceGlobalAllocateTemplate = NodeTemplate("")

referenceStructInitTemplate = NodeTemplate("""
static ${type.typeName} ${name};
""")
#static const ${type}* ${name} = &${name}_UL;

referenceStructAllocateTemplate = NodeTemplate("""
    ${name} = (${structDict.typeName}) ${str(structDict)};
""")
