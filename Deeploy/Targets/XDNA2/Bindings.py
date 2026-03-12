# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import bfloat16_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.Targets.XDNA2.Templates import AddTemplate
from Deeploy.Targets.XDNA2.TypeCheckers import XDNA2AddChecker

# XDNA2 does not use the standard C code transformation pipeline.
# The deployer generates a holistic MLIR module, not per-node C snippets.
# An empty CodeTransformation is used as a placeholder.
XDNA2Transformer = CodeTransformation([])

XDNA2AddBindings = [
    NodeBinding(
        XDNA2AddChecker([PointerClass(bfloat16_t), PointerClass(bfloat16_t)], [PointerClass(bfloat16_t)]),
        AddTemplate.referenceTemplate,
        XDNA2Transformer,
    )
]
