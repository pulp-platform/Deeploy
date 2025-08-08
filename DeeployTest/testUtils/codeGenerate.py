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
from typing import List, Tuple

import numpy as np

from Deeploy.DeeployTypes import CodeGenVerbosity, ConstantBuffer, NetworkDeployer, VariableBuffer
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


def generateTestInputsHeader(deployer: NetworkDeployer, test_inputs: List) -> str:
    vectors = []
    retStr = ""
    for index, values in enumerate(test_inputs):
        # WIESEP: Correctly handle empty arrays
        if np.prod(values.shape) == 0:
            continue

        values = _shapeBroadcast(deployer.ctxt, values, f"input_{index}")

        buffer = deployer.ctxt.lookup(f"input_{index}")
        typeName = buffer._type.referencedType.typeName
        typeWidth = buffer._type.referencedType.typeWidth

        vectorName = f"testInputVector{index}"
        vectors.append(vectorName)

        retStr += f"{typeName} {vectorName}[] ="
        retStr += "{"
        if typeName == 'float32_t':
            list_str = (", ").join([f'{x}f' if not (np.isinf(x) or np.isnan(x)) else str(x) for x in values])
        else:
            list_str = (", ").join([str(x) for x in values])

        # WIESEP: Arrays have to be 4 byte aligned (at least in banshee)
        bytes = (len(values) * typeWidth) // 8
        if bytes % 4 != 0:
            paddingBytes = bytes % 4
            paddingElements = paddingBytes * 8 // typeWidth
            list_str += ", "
            list_str += (", ").join([str(0) for _ in range(paddingElements)])

        retStr += list_str
        retStr += "};\n"

    retStr += f"void* testInputVector[{len(vectors)}] = {{"
    retStr += ", ".join(vectors)
    retStr += "};\n"
    return retStr


def generateTestOutputsHeader(deployer: NetworkDeployer, test_outputs: List[np.ndarray]) -> str:
    retStr = ""
    for index, values in enumerate(test_outputs):
        typeName = deployer.ctxt.lookup(f'output_{index}')._type.referencedType.typeName
        typeWidth = deployer.ctxt.lookup(f'output_{index}')._type.referencedType.typeWidth

        retStr += f"#define OUTPUTTYPE {typeName}\n"
        retStr += f"#define ISOUTPUTFLOAT {int(typeName == 'float32_t')}\n"
        retStr += f"{typeName} testOutputVector{index}[] ="
        retStr += "{"

        values = values.flatten()

        if typeName == "float32_t":
            list_str = (", ").join([f'{x}f' if not (np.isinf(x) or np.isnan(x)) else str(x) for x in values])
        else:
            list_str = (", ").join([str(x) for x in values])

        # WIESEP: Arrays have to be 4 byte aligned (at least in banshee)
        bytes = (len(values) * typeWidth) // 8
        if bytes % 4 != 0:
            paddingBytes = bytes % 4
            paddingElements = paddingBytes * 8 // typeWidth
            list_str += ", "
            list_str += (", ").join([str(0) for _ in range(paddingElements)])

        retStr += list_str
        retStr += "};\n"

    retStr += f"void* testOutputVector[{len(test_outputs)}] = " + "{"
    retStr += ", ".join([f"testOutputVector{idx}" for idx, _ in enumerate(test_outputs)])
    retStr += "};\n"
    return retStr


def generateTestNetworkHeader(deployer: NetworkDeployer) -> str:

    retStr = ""

    retStr += """
    #ifndef __DEEPLOY_HEADER__
    #define __DEEPLOY_HEADER__
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


def generateTestNetworkImplementation(deployer: NetworkDeployer, verbosityCfg: CodeGenVerbosity) -> str:
    retStr = ""

    retStr += """#include <stdio.h>
    #include <stdlib.h>
    #include <math.h>
    """
    retStr += deployer.generateIncludeString()
    retStr += """

    #include "Network.h"

    """

    retStr += deployer.generateBufferInitializationCode()
    retStr += deployer.generateGlobalDefinitionCode()

    # WIESEP: Mempool assigns section attributes to intermediate buffers to allow .
    if isinstance(deployer.Platform, MemPoolPlatform):
        retStr += deployer.generateInferenceInitializationCode()
        retStr += """
        void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
        """
    else:
        retStr += """
        void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
        """
        retStr += deployer.generateInferenceInitializationCode()

    retStr += deployer.generateFunction(verbosityCfg)
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
        if dataType.referencedType.typeName == "float32_t":
            retStr = "float32"
            width = 32
        else:
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

    # LMACAN: Dump all global buffers with the "extName" attribute
    os.makedirs(path, exist_ok = True)
    for buf in deployer.ctxt.globalObjects.values():
        if hasattr(buf, "extName"):
            pathName = os.path.join(path, f"{buf.extName}.hex")
            dumpBuffer(buf, pathName)


def generateTestNetwork(deployer: NetworkDeployer, test_inputs: List[np.ndarray], test_outputs: List[np.ndarray],
                        dumpdir: str, verbosityCfg: CodeGenVerbosity) -> None:
    assert deployer.prepared, "An unprepared deployer was given"

    # Create input and output vectors
    os.makedirs(dumpdir, exist_ok = True)

    testInputStr = generateTestInputsHeader(deployer, test_inputs)
    with open(f'{dumpdir}/testinputs.h', "w") as f:
        f.write(testInputStr)

    testOutputStr = generateTestOutputsHeader(deployer, test_outputs)
    with open(f'{dumpdir}/testoutputs.h', "w") as f:
        f.write(testOutputStr)

    # Generate code for Network
    testNetworkHeaderStr = generateTestNetworkHeader(deployer)
    with open(f'{dumpdir}/Network.h', "w") as f:
        f.write(testNetworkHeaderStr)

    testNetworkImplementationStr = generateTestNetworkImplementation(deployer, verbosityCfg)
    with open(f'{dumpdir}/Network.c', "w") as f:
        f.write(testNetworkImplementationStr)

    generateL3HexDump(deployer, os.path.join(f'{dumpdir}', 'hex'), test_inputs, test_outputs)

    clang_format = "{BasedOnStyle: llvm, IndentWidth: 2, ColumnLimit: 160}"
    os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/Network.c')
    os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/Network.h')
    os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/testoutputs.h')
    os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/testinputs.h')
