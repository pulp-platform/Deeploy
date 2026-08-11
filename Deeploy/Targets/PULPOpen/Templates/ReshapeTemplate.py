# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.Targets.Generic.Templates.ReshapeTemplate import _ReshapeTemplate as _GenericReshapeTemplate


class _ReshapeTemplate(_GenericReshapeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = _ReshapeTemplate("""
// Reshape (Name: ${nodeName}, Op: ${nodeOp})
${data_out} = ${data_in};
""")
