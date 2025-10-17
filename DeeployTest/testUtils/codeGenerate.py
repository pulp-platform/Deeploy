# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import math
import os
from typing import List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import FloatImmediate, IntegerImmediate
from Deeploy.DeeployTypes import CodeGenVerbosity, ConstantBuffer, NetworkDeployer, VariableBuffer
from Deeploy.Targets.MemPool.Platform import MemPoolPlatform
from Deeploy.Targets.PULPOpen.Platform import MemoryPULPPlatform, MemoryPULPPlatformWrapper, PULPPlatform

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


def generateArray(name: str, buffer: VariableBuffer, values: np.ndarray) -> str:
    assert math.prod(buffer.shape) == math.prod(values.shape), \
        f"Buffer size ({math.prod(buffer.shape)}) and values size ({math.prod(values.shape)}) are not equal."
    refTy = buffer._type.referencedType

    values = values.flatten()

    if issubclass(refTy, FloatImmediate):
        if refTy.typeWidth == 32:
            suffix = "f"
        elif refTy.typeWidth == 64:
            suffix = ""
        else:
            raise RuntimeError(
                f"Unimplemented floating-poing literal suffix for type {refTy.typeName} of typeWidth {refTy.typeWidth}")

        def formatFloat(x: float, suffix: str = "") -> str:
            if np.isinf(x) or np.isnan(x):
                return str(x)
            else:
                return str(x) + suffix

        list_str = ",".join(formatFloat(x) for x in values)
    elif issubclass(refTy, IntegerImmediate):
        suffix = "u" if refTy.typeMin >= 0 else ""
        suffix += "l" if refTy.typeWidth >= 64 else ""
        list_str = ",".join(str(int(x)) + suffix for x in values)
    else:
        list_str = ",".join(str(x) for x in values)

    # WIESEP: Arrays have to be 4 byte aligned (at least in banshee)
    total_bytes = (values.size * refTy.typeWidth) // 8
    pad_bytes = (-total_bytes) % 4
    if pad_bytes:
        paddingElements = (pad_bytes * 8 + refTy.typeWidth - 1) // refTy.typeWidth
        list_str += ", " + (", ").join("0" for _ in range(paddingElements))

    return f"{refTy.typeName} {name}[] = {{ {list_str} }};\n"


def generateTestInputsHeader(deployer: NetworkDeployer, test_inputs: List) -> str:
    vectors = []
    retStr = ""
    for index, values in enumerate(test_inputs):
        # WIESEP: Correctly handle empty arrays
        if np.prod(values.shape) == 0:
            continue

        bufferName = f"input_{index}"

        #LMACAN: We have some tests which have extra inputs and this is a hack to circumvent that
        if not deployer.ctxt.is_buffer(bufferName):
            continue

        buffer = deployer.ctxt.lookup(bufferName)
        assert isinstance(buffer, VariableBuffer)

        bufferSize = math.prod(buffer.shape)
        valuesSize = math.prod(values.shape)
        assert bufferSize % valuesSize == 0, \
        f"Values shape {values.shape} of size {valuesSize} cannot be repeated into buffer of shape {buffer.shape} and size {bufferSize}."
        repeat = bufferSize // valuesSize
        values = np.tile(values, repeat)

        vectorName = f"testInputVector{index}"
        retStr += generateArray(vectorName, buffer, values)
        vectors.append(vectorName)

    retStr += f"void* testInputVector[{len(vectors)}] = {{"
    retStr += ",".join(vectors)
    retStr += "};\n"

    return retStr


def generateTestOutputsHeader(deployer: NetworkDeployer, test_outputs: List[np.ndarray]) -> str:
    vectors = []
    retStr = ""
    for index, values in enumerate(test_outputs):
        buffer = deployer.ctxt.lookup(f"output_{index}")
        assert isinstance(buffer, VariableBuffer)
        refTy = buffer._type.referencedType

        retStr += f"#define OUTPUTTYPE {refTy.typeName}\n"
        retStr += f"#define ISOUTPUTFLOAT {int(refTy.typeName == 'float32_t')}\n"

        vectorName = f"testOutputVector{index}"
        retStr += generateArray(vectorName, buffer, values)
        vectors.append(vectorName)

    retStr += f"void* testOutputVector[{len(vectors)}] = {{"
    retStr += ",".join(vectors)
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
    if isinstance(deployer.Platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):
        retStr += """
        void RunNetwork();
        void InitNetwork();

        """
    else:
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
    elif isinstance(deployer.Platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):
        retStr += """
        void RunNetwork(){
        """
        retStr += deployer.generateInferenceInitializationCode()
    else:
        retStr += """
        void RunNetwork(__attribute__((unused)) uint32_t core_id, __attribute__((unused)) uint32_t numThreads){
        """
        retStr += deployer.generateInferenceInitializationCode()

    retStr += deployer.generateFunction(verbosityCfg)
    if isinstance(deployer.Platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):
        retStr += """
        }

        void InitNetwork(){
        """
    else:
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
