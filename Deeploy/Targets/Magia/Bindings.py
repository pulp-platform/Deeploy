# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0


from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t, int32_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.Targets.Generic.TypeCheckers import AddChecker
from Deeploy.Targets.Magia.Templates import AddTemplate

BasicTransformer = CodeTransformation([])

MagiaAddBindings = [
    NodeBinding(AddChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                AddTemplate.referenceTemplate, BasicTransformer),
]