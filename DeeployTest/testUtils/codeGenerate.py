# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import List, Tuple

import numpy as np

from Deeploy.DeeployTypes import CodeGenVerbosity, ConstantBuffer, NetworkDeployer, VariableBuffer
from Deeploy.Targets.Carfield.Platform import CarfieldPlatform
from Deeploy.Targets.MemPool.Platform import MemPoolPlatform
from Deeploy.Targets.PULPOpen.Platform import MemoryPULPPlatform, MemoryPULPPlatformWrapper, PULPPlatform
from Deeploy.Targets.Snitch.Platform import SnitchPlatform

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

        bufferName = f"input_{index}"

        #LMACAN: We have some tests which have extra inputs and this is a hack to circumvent that
        if not deployer.ctxt.is_buffer(bufferName):
            continue

        values = _shapeBroadcast(deployer.ctxt, values, bufferName)

        buffer = deployer.ctxt.lookup(bufferName)

        # When the input lives in L3, its data is delivered at runtime from
        # the readfs hex file (load_file_to_ram) and every L3-capable harness
        # skips the testInputVector memcpy for external (L3) addresses. Emitting
        # the data here would just duplicate the whole input tensor inside the
        # binary, so keep a NULL placeholder to preserve testInputVector[]
        # indexing without storing the data twice.
        if getattr(buffer, "_memoryLevel", None) == "L3":
            vectors.append("NULL")
            continue

        typeName = buffer._type.referencedType.typeName
        typeWidth = buffer._type.referencedType.typeWidth

        if isinstance(deployer.Platform, CarfieldPlatform):
            vectorName = f"testInputVector{index}"
            vectors.append(vectorName)
            retStr += f"#define {vectorName} (({typeName} *){buffer.address})\n"
            continue

        vectorName = f"testInputVector{index}"
        vectors.append(vectorName)

        retStr += f"{typeName} {vectorName}[] ="
        retStr += "{"
        if typeName == 'float32_t':
            list_str = (", ").join([f'{x}f' if not (np.isinf(x) or np.isnan(x)) else str(x) for x in values])
        else:
            list_str = (", ").join([str(x) for x in values])

        # WIESEP: Arrays have to be 4 byte aligned (at least in banshee)
        total_bytes = (values.size * typeWidth) // 8
        pad_bytes = (-total_bytes) % 4
        if pad_bytes:
            paddingElements = (pad_bytes * 8 + typeWidth - 1) // typeWidth
            list_str += ", " + (", ").join("0" for _ in range(paddingElements))

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

        if isinstance(deployer.Platform, CarfieldPlatform):
            buffer = deployer.ctxt.lookup(f'output_{index}')
            retStr += f"#define testOutputVector{index} (({typeName} *){buffer.address})\n"
            continue

        retStr += f"{typeName} testOutputVector{index}[] ="
        retStr += "{"

        values = values.flatten()

        if typeName == "float32_t":
            list_str = (", ").join([f'{x}f' if not (np.isinf(x) or np.isnan(x)) else str(x) for x in values])
        else:
            list_str = (", ").join([str(x) for x in values])

        # WIESEP: Arrays have to be 4 byte aligned (at least in banshee)
        total_bytes = (len(values) * typeWidth) // 8
        pad_bytes = (-total_bytes) % 4
        if pad_bytes:
            paddingElements = (pad_bytes * 8 + typeWidth - 1) // typeWidth
            list_str += ", " + (", ").join("0" for _ in range(paddingElements))

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

    if isinstance(deployer.Platform, CarfieldPlatform):
        retStr += """#include <stdio.h>
        #include <stdlib.h>
        """
    else:
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

    # MemPool and Snitch declare intermediate buffers at file scope (before RunNetwork) so they are shared across cores.
    if isinstance(deployer.Platform, (MemPoolPlatform, SnitchPlatform, CarfieldPlatform)):
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

        # Check if buffer name matches exactly "input_N" or "output_N" pattern
        parts = buf.name.split("_")
        if len(parts) == 2 and parts[0] == "input" and parts[1].isdigit():
            idx = int(parts[1])
            array = _shapeBroadcast(deployer.ctxt, test_inputs[idx], f"input_{idx}")

        elif len(parts) == 2 and parts[0] == "output" and parts[1].isdigit():
            idx = int(parts[1])
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


def _carfieldNumpyDtype(dataType) -> str:
    # Numpy dtype string matching the C element type, little-endian explicit.
    referenced = dataType.referencedType
    width = referenced.typeWidth

    if referenced.typeName == "float32_t":
        return "<f4"

    if width == 8:
        return "i1" if referenced.typeMin < 0 else "u1"

    order = "<"
    kind = "i" if referenced.typeMin < 0 else "u"
    return f"{order}{kind}{width // 8}"


def generateCarfieldDDRBlobs(deployer: NetworkDeployer, path: str, test_inputs: List[np.ndarray],
                             test_outputs: List[np.ndarray]) -> None:
    """Write the raw DDR images that `gdb restore` pushes onto the board.

    Three binary files (blobs), because they change at different times: the weights only when
    the network changes, the input and golden every time the test vectors are
    regenerated.

        weights.bin   every ConstantBuffer, at the offsets CarfieldDeployer assigned
        input.bin     the network inputs
        golden.bin    the expected outputs, for the on-device comparison
        layout.txt    the addresses and the restore commands, to read while in gdb

    The addresses are never recomputed here: they are read back from
    `buf.address`, which `CarfieldDeployer.bind()` stamped on every buffer."""
    os.makedirs(path, exist_ok = True)

    def dataFor(buf: VariableBuffer) -> np.ndarray:
        """The bytes this buffer stands for, in its C element type."""
        parts = buf.name.split("_")
        if len(parts) == 2 and parts[0] == "input" and parts[1].isdigit():
            array = _shapeBroadcast(deployer.ctxt, test_inputs[int(parts[1])], buf.name)
        elif len(parts) == 2 and parts[0] == "output" and parts[1].isdigit():
            array = _shapeBroadcast(deployer.ctxt, test_outputs[int(parts[1])], buf.name)
        elif isinstance(buf, ConstantBuffer):
            array = buf.values
        else:
            raise Exception(f"Unexpected buffer {buf}!")

        # test_inputs/test_outputs arrive as float64 (generateNetwork widens them
        # before type inference); the cast brings them back to the C element type.
        return np.ascontiguousarray(array).reshape(-1).astype(_carfieldNumpyDtype(buf._type))

    def writeBlob(fileName: str, buffers: List[VariableBuffer], origin: int) -> int:
        """Write buffers into one file, each at its offset from "origin".

        "origin" is the address the file will be restored at, so offset 0 in the
        file is byte 0 of that region, using a single global base here instead
        would prepend the distance from the weights and shift every tensor.

        Positioning with seek() rather than appending in order is what makes the
        blob independent of iteration order: a tensor lands where its address
        says, full stop. The gaps left by the alignment padding are the holes
        seek() leaves behind, which read back as zeros.
        """
        filePath = os.path.join(path, fileName)
        with open(filePath, "wb") as f:
            for buf in buffers:
                f.seek(int(buf.address, 16) - origin)
                f.write(dataFor(buf).tobytes())

        size = os.path.getsize(filePath)

        # Independent of the seek arithmetic above: the file must span exactly
        # from the region start to the end of its last tensor. A wrong origin
        # inflates the file, and this is what notices.
        expected = max(int(buf.address, 16) - origin + dataFor(buf).nbytes for buf in buffers)
        assert size == expected, \
            f"{fileName} is {size} bytes but the layout spans {expected}; origin 0x{origin:08X} is wrong"

        return size

    constants = [buf for buf in deployer.ctxt.globalObjects.values() if isinstance(buf, ConstantBuffer) and buf._deploy]

    blobs = []
    if constants:
        blobs.append(("weights.bin", constants, deployer.weightsRegion[0]))
    if deployer.inputs():
        blobs.append(("input.bin", list(deployer.inputs()), deployer.inputBlobAddress))
    if deployer.outputs():
        blobs.append(("golden.bin", list(deployer.outputs()), deployer.goldenBlobAddress))

    written = []
    for fileName, buffers, startAddress in blobs:
        size = writeBlob(fileName, buffers, startAddress)
        written.append((fileName, startAddress, size, len(buffers)))

    # Read the blobs back and compare tensor by tensor. This catches the failure
    # mode that produces no error at all: a wrong offset or a wrong dtype leaves
    # every pointer valid and every value wrong.
    for fileName, buffers, startAddress in blobs:
        raw = np.fromfile(os.path.join(path, fileName), dtype = np.uint8)
        for buf in buffers:
            expected = dataFor(buf)
            offset = int(buf.address, 16) - startAddress
            actual = raw[offset:offset + expected.nbytes].view(expected.dtype)
            assert np.array_equal(actual, expected), \
                f"blob {fileName} does not match buffer {buf.name} at {buf.address}"

    with open(os.path.join(path, "layout.txt"), "w") as f:
        f.write("Carfield DDR layout\n")
        f.write("=" * 78 + "\n\n")
        f.write("Restore commands (run these in gdb, once per session):\n\n")
        for fileName, startAddress, size, _ in written:
            f.write(f"  restore {fileName} binary 0x{startAddress:08X}\n")
        f.write(f"\n  arena base (deeploy_malloc starts here): 0x{deployer.arenaBaseAddress:08X}\n")
        f.write(f"\n{'Blob':<14} {'Address':>12} {'Bytes':>14} {'Tensors':>9}\n")
        f.write("-" * 78 + "\n")
        for fileName, startAddress, size, count in written:
            f.write(f"{fileName:<14} 0x{startAddress:08X} {size:>14,d} {count:>9d}\n")
        f.write(f"\n{'Tensor':<44} {'Address':>12} {'Bytes':>12}\n")
        f.write("-" * 78 + "\n")
        for _, buffers, _ in blobs:
            for buf in buffers:
                f.write(f"{buf.name:<44} {buf.address:>12} {dataFor(buf).nbytes:>12,d}\n")


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

    # Carfield keeps weights, input and golden out of the binary entirely: they
    # are restored into DDR with gdb and reached through the fixed-address
    # pointers CarfieldDeployer assigned.
    if isinstance(deployer.Platform, CarfieldPlatform):
        generateCarfieldDDRBlobs(deployer, os.path.join(f'{dumpdir}', 'ddr'), test_inputs, test_outputs)

    clang_format = "{BasedOnStyle: llvm, IndentWidth: 2, ColumnLimit: 160}"
    os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/Network.c')
    os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/Network.h')
    os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/testoutputs.h')
    os.system(f'clang-format -i --style="{clang_format}" {dumpdir}/testinputs.h')
