# ----------------------------------------------------------------------
#
# File: FlattenTileConstraint.py
#
# Last edited: 02.06.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
