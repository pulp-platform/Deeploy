# ----------------------------------------------------------------------
#
# File: codeGenerate.py
#
# Last edited: 23.05.2023
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

import os
from pprint import pprint
from typing import Dict, List, Optional, Tuple

import numpy as np

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentPlatform, NetworkDeployer, VariableBuffer
from Deeploy.Targets.MemPool.Platform import MemPoolPlatform

_TEXT_ALIGN = 30


def _shapeBroadcast(ctxt, value, name):
    if ctxt.is_global(f"{name}"):
        broadcastShape = ctxt.lookup(f"{name}").shape
        repeat = np.prod(broadcastShape) / np.prod(value.shape)
        # Raise error if repeat is not an integer
        if repeat % 1 != 0:
            raise ValueError(f"Input {name} has to be broadcastable to shape {broadcastShape}!")
        repeatNum = np.tile(value, int(repeat))
        broadcastNum = repeatNum.reshape(-1)
        ctxt.lookup(f"{name}").shape = broadcastNum.shape
    else:
        broadcastNum = value

    return broadcastNum


def generateTestInputsHeader(deployer: NetworkDeployer, test_inputs: List, inputTypes: Dict, inputOffsets: Dict) -> str:
    retStr = ""
    inputNames = [deployer.ctxt.lookup(buf.name) for buf in deployer.graph.inputs]
    inputTypes = {buf.name: buf._type for buf in inputNames}

    for index, num in enumerate(test_inputs):

        if f"input_{index}" not in inputTypes.keys():
            continue

        # WIESEP: Correctly handle empty arrays
        if np.prod(num.shape) == 0:
            continue

        test_inputs[index] -= inputOffsets[f"input_{index}"]

        broadcastNum = _shapeBroadcast(deployer.ctxt, num, f"input_{index}")

        data_type = inputTypes[f"input_{index}"]
        data_width = inputTypes[f"input_{index}"].referencedType.typeWidth

        retStr += f"{data_type.referencedType.typeName} testInputVector{index}[] ="
        retStr += "{"
        list_str = (", ").join([str(x) for x in broadcastNum])

        # WIESEP: Arrays have to be 4 byte alinged (at lest in banshee)
        bytes = len(broadcastNum) * (data_width // 8)
        if bytes % 4 != 0:
            bytes = 4 * int((bytes / 4 + 1))
            padding = (bytes * 8) // data_width - len(broadcastNum)
            list_str += ", "
            list_str += (", ").join([str(0) for x in range(padding)])

        retStr += list_str
        retStr += "};\n"

    retStr += f"void* testInputVector[{len(inputTypes)}] = " + "{"
    retStr += ", ".join([
        f"testInputVector{idx}" for idx, _ in enumerate(test_inputs)
        if np.prod(test_inputs[idx].shape) != 0 and f"input_{idx}" in inputTypes.keys()
    ])
    retStr += "};\n"

    return retStr


def generateTestOutputsHeader(deployer: NetworkDeployer,
                              test_outputs: List,
                              signProp: Optional[bool] = None,
                              verbose: Optional[bool] = None) -> str:

    output_signed = {}
    output_n_levels = {}
    output_data_type = {}

    if signProp is None:
        signProp = False

    if verbose is None:
        verbose = False

    retStr = ""

    for index, num in enumerate(test_outputs):
        output_data_type[f"output_{index}"] = deployer.ctxt.lookup(f'output_{index}')._type

        if signProp:
            output_n_levels[f"output_{index}"] = deployer.ctxt.lookup(f'output_{index}').nLevels
            output_signed[f"output_{index}"] = deployer.ctxt.lookup(f'output_{index}')._signed
            test_outputs[index] -= int(
                ((1 - output_signed[f"output_{index}"]) * (output_n_levels[f"output_{index}"] / 2)))

        data_type = output_data_type[f"output_{index}"]
        data_width = data_type.referencedType.typeWidth
        retStr += f"{data_type.referencedType.typeName} testOutputVector{index}[] ="
        retStr += "{"

        # WIESEP: Arrays have to be 4 byte alinged (at lest in banshee)
        list_str = (", ").join([str(x) for x in num])

        bytes = len(num) * (data_width // 8)
        if bytes % 4 != 0:
            bytes = 4 * int((bytes / 4 + 1))
            padding = (bytes * 8) // data_width - len(num)
            list_str += ", "
            list_str += (", ").join([str(0) for x in range(padding)])

        retStr += list_str
        retStr += "};\n"

    retStr += f"void* testOutputVector[{len(test_outputs)}] = " + "{"
    retStr += ", ".join([f"testOutputVector{idx}" for idx, _ in enumerate(test_outputs)])
    retStr += "};\n"

    if verbose:
        if signProp:
            print('Output N Levels:')
            pprint(output_n_levels, indent = 2, width = 120)
            print('Output Signed:')
            pprint(output_signed, indent = 2, width = 120)
        print('Output Data Type:')
        pprint(output_data_type, indent = 2, width = 120)

    return retStr


def generateTestNetworkHeader(deployer: NetworkDeployer, platform: DeploymentPlatform) -> str:

    retStr = ""

    retStr += """
    #ifndef __DEEPLOY_HEADER_
    #define __DEEPLOY_HEADER_
    #include <stdio.h>
    #include <stdint.h>
    #include <stdlib.h>
    """
    retStr += deployer.generateIncludeString()
    retStr += """
    void RunNetwork(uint32_t core_id, uint32_t numThreads);
    void InitNetwork(uint32_t core_id, uint32_t numThread);

    """

    retStr += deployer.generateIOBufferInitializationCode()
    retStr += """
    #endif
    """

    return retStr


def generateTestNetworkImplementation(deployer: NetworkDeployer,
                                      platform: DeploymentPlatform,
                                      verbose: Optional[bool] = None) -> str:

    if verbose is None:
        verbose = False

    retStr = ""

    retStr += """#include <stdio.h>
    #include <stdlib.h>
    """
    retStr += deployer.generateIncludeString()
    retStr += """

    #include "Network.h"

    """

    retStr += deployer.generateBufferInitializationCode()
    retStr += deployer.generateGlobalDefinitionCode()

    # WIESEP: Mempool assigns section attributes to intermediate buffers to allow .
    if isinstance(platform, MemPoolPlatform):
        retStr += deployer.generateInferenceInitializationCode()
        retStr += """
        void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
        """
    else:
        retStr += """
        void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
        """
        retStr += deployer.generateInferenceInitializationCode()

    retStr += deployer.generateFunction(verbose)
    retStr += """
    }

    void InitNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
    """
    retStr += deployer.generateEngineInitializationCode()
    retStr += deployer.generateBufferAllocationCode()
    retStr += """
    }
    """

    return retStr


def generateL3HexDump(deployer: NetworkDeployer, path: str, test_inputs: List, test_outputs: List):

    def type2TypeStr(dataType) -> Tuple[str, int]:
        width = dataType.referencedType.typeWidth
        signed = (dataType.referencedType.typeMin < 0)

        retStr = ""

        if signed:
            retStr += "int"
        else:
            retStr += "uint"

        retStr += str(width)

        return retStr, width

    def dumpBuffer(buf: VariableBuffer, path: str):

        if "input" in buf.name:
            idx = int(buf.name.split("_")[1])
            array = _shapeBroadcast(deployer.ctxt, test_inputs[idx], f"input_{idx}")

        elif "output" in buf.name:
            _list = buf.name.split("_")
            idx = int(_list[1])
            array = _shapeBroadcast(deployer.ctxt, test_outputs[idx], f"output_{idx}")

        elif isinstance(buf, ConstantBuffer):
            array = buf.values
        else:
            raise Exception(f"Unexpected buffer {buf}!")

        typeStr, width = type2TypeStr(buf._type)

        # Word alignment
        mod = (32 // width)
        paddingLength = (mod - (array.size % mod)) % mod
        paddedArray = np.pad(array.flatten(), (0, paddingLength), 'constant')

        paddedArray.astype(typeStr).tofile(path)

    # SCHEREMO: Dump all global const buffers as hex files
    globalConstBuffers = [
        buf for key, buf in deployer.ctxt.globalObjects.items() if isinstance(buf, VariableBuffer) and buf._deploy
    ]
    l3ConstBuffer = [buf for buf in globalConstBuffers if hasattr(buf, "_memoryLevel") and buf._memoryLevel == "L3"]

    os.makedirs(path, exist_ok = True)

    for idx, buf in enumerate(l3ConstBuffer):
        if hasattr(buf, "extName"):
            pathName = os.path.join(path, f"{buf.extName}.hex")
            dumpBuffer(buf, pathName)
