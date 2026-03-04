# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Dict, List, Tuple, Union

from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme
from Deeploy.Targets.PULPOpen.TileConstraints.SoftmaxCrossEntropyTileConstraint import \
    SoftmaxCrossEntropyTileConstraint


class SoftmaxCrossEntropyLossDualOutputTileConstraint(SoftmaxCrossEntropyTileConstraint):
    """TileConstraint for SoftmaxCrossEntropyLoss with 2 outputs:
      - log_prob  : [batch, num_classes]  (primary output — same as single-output version)
      - loss      : []  0-d scalar (scalar cross-entropy mean)

    Both batch and num_classes are pinned to their full size by the inherited
    addPolicyConstraint, so no actual tiling of SCE occurs.  The sole purpose of
    this subclass is to override wrapTilingSolution so that the base-class
    single-output assertion is bypassed, and the scalar loss buffer is included
    in the DMA output schedule.
    """

    # Key in operatorRepresentation for the scalar loss output buffer name.
    dataLossName = 'loss'

    @classmethod
    def wrapTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, List[TilingSchedule]]:

        logProbVar = operatorRepresentation[cls.dataOutName]   # e.g. "onnx::log_prob::3"
        lossVar    = operatorRepresentation.get(cls.dataLossName, '')

        # If loss is absent (empty string — single-output fallback) or not in the
        # memory constraint dict, delegate straight to the parent unchanged.
        if not lossVar or lossVar not in tilingSolution.outputTensorMemoryConstraints:
            return super().wrapTilingSolution(tilingSolution, targetMemLevel, ctxt, operatorRepresentation)

        # Build a single-output copy of tilingSolution (log_prob only) so that
        # the base-class assertion `len(outputTensorMemoryConstraints) == 1` passes.
        singleOutputSolution = copy.deepcopy(tilingSolution)
        singleOutputSolution.outputTensorMemoryConstraints = {
            logProbVar: tilingSolution.outputTensorMemoryConstraints[logProbVar]
        }

        # Call the base-class wrapTilingSolution, which runs cube computation and
        # calls serializeTilingSolution for log_prob.
        varReplacement, tilingSchedules = super().wrapTilingSolution(
            singleOutputSolution, targetMemLevel, ctxt, operatorRepresentation)

        # Extend each TilingSchedule to include the scalar loss output.
        # The loss tensor is always 1 element (0-d scalar represented as [1] for DMA).
        lossAddr = TileConstraint.getBaseAddr(tilingSolution, targetMemLevel, lossVar)

        # If the address is None (IO tensor with runtime-determined address, or tensor
        # not allocated at this memory level), skip — same logic as sanitizeTilingSchedule.
        if lossAddr == [None]:
            return varReplacement, tilingSchedules

        lossRect = HyperRectangle((0,), (1,))

        for schedule in tilingSchedules:
            schedule.outputBaseOffsets[cls.dataLossName] = lossAddr
            for step in schedule.outputLoadSchedule:
                step[cls.dataLossName] = lossRect

        return varReplacement, tilingSchedules
