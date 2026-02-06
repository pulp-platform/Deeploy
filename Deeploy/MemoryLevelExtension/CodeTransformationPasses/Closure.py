# ----------------------------------------------------------------------
#
# File: Closure.py
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Deeploy.AbstractDataTypes import PointerClass, StructClass, VoidType
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureGeneration, _closureCallTemplate
from Deeploy.DeeployTypes import ExecutionBlock, NetworkContext, NodeTemplate, StructBuffer, TransientBuffer


class MemoryAwareClosureGeneration(ClosureGeneration):
    """
    Memory-aware closure generation for multi-level memory hierarchies.

    This class extends ClosureGeneration to handle memory-aware closure
    generation where only certain memory levels are included in the closure
    arguments. It filters buffers based on their memory level, including
    only those that belong to specific memory regions in the hierarchy.

    Notes
    -----
    This class is useful for multi-level memory systems where different
    memory levels have different access patterns and only certain levels
    should be passed as closure arguments. Buffers are included if they:
    - Have no memory level annotation
    - Belong to the start region
    - Do not belong to the end region (are in higher levels)
    """

    def __init__(self,
                 closureCallTemplate: NodeTemplate = _closureCallTemplate,
                 closureSuffix = "_closure",
                 writeback: bool = True,
                 generateStruct: bool = True,
                 startRegion: str = "L2",
                 endRegion: str = "L1"):
        """
        Initialize the MemoryAwareClosureGeneration transformation pass.

        Parameters
        ----------
        closureCallTemplate : NodeTemplate, optional
            Template for generating closure function calls. Default is the
            global _closureCallTemplate.
        closureSuffix : str, optional
            Suffix to append to closure function names. Default is "_closure".
        writeback : bool, optional
            Whether to generate writeback code for closure arguments.
            Default is True.
        generateStruct : bool, optional
            Whether to generate argument structure definitions. Default is True.
        startRegion : str, optional
            The starting memory region to include in closures. Default is "L2".
        endRegion : str, optional
            The ending memory region to include in closures. Default is "L1".
        """
        super().__init__(closureCallTemplate, closureSuffix, writeback, generateStruct)
        self.startRegion = startRegion
        self.endRegion = endRegion

    # Don't override this
    def _generateClosureStruct(self, ctxt: NetworkContext, executionBlock: ExecutionBlock):
        """
        Generate memory-aware closure argument structure.

        Overrides the base class method to implement memory-level filtering.
        Only includes buffers that belong to appropriate memory levels based
        on the configured start and end regions.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context containing buffer information.
        executionBlock : ExecutionBlock
            The execution block to analyze for dynamic references.

        Notes
        -----
        This method filters dynamic references based on memory levels:
        - Includes buffers with no memory level annotation
        - Includes buffers from the start region
        - Includes buffers not from the end region (higher memory levels)

        The filtering logic ensures that only relevant buffers are passed
        as closure arguments, reducing memory transfer overhead in
        multi-level memory hierarchies.
        """

        # Add closure struct info to operatorRepresentation
        closureStructArgsType = {}
        closureStruct = {}
        makoDynamicReferences = self.extractDynamicReferences(ctxt, executionBlock, unrollStructs = True)

        filteredMakoDynamicReferences = []

        for ref in makoDynamicReferences:
            buf = ctxt.lookup(ref)
            if not hasattr(buf, "_memoryLevel") or buf._memoryLevel is None:
                filteredMakoDynamicReferences.append(ref)
                continue

            if buf._memoryLevel == self.startRegion or buf._memoryLevel != self.endRegion:
                filteredMakoDynamicReferences.append(ref)

        for arg in list(dict.fromkeys(filteredMakoDynamicReferences)):
            ref = ctxt.lookup(arg)
            if isinstance(ref, TransientBuffer):
                closureStructArgsType[ctxt._mangle(arg)] = PointerClass(VoidType)
            elif not isinstance(ref, StructBuffer):
                closureStructArgsType[ctxt._mangle(arg)] = ref._type

            if not isinstance(ref, StructBuffer):
                closureStruct[ctxt._mangle(arg)] = arg

        structClass = StructClass(self.closureName + "_args_t", closureStructArgsType)
        self.closureStructArgType = structClass
        self.closureStructArgs = self.closureStructArgType(closureStruct, ctxt)
