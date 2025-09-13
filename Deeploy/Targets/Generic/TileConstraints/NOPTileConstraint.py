# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from Deeploy.DeeployTypes import NetworkContext
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel


class NOPTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        pointer: List[str] = []

        for key, value in parseDict.items():
            if not isinstance(value, str):
                continue

            if ctxt.is_global(value) or ctxt.is_local(value):
                pointer.append(value)

        #Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, outputBufferName]:

            _buffer = ctxt.lookup(bufferName)

            tilerModel.addTensorDimToModel(ctxt, bufferName)

            for idx, shapeDim in enumerate(_buffer.shape):
                tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = bufferName, dimIdx = idx) <= shapeDim)

        # Remove unused tensors from deployment
        for bufferName in pointer:
            if bufferName not in [inputBufferName, outputBufferName]:
                ctxt.lookup(bufferName)._deploy = False

        return tilerModel
