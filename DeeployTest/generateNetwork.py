# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from testUtils.codeGenerate import generateTestNetwork
from testUtils.graphDebug import generateDebugConfig
from testUtils.platformMapping import mapDeployer, mapPlatform
from testUtils.testRunner import TestGeneratorArgumentParser
from testUtils.typeMapping import inferTypeAndOffset, parseDataType

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import IntegerDataTypes
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.DebugPasses import EmulateCMSISRequantPass
from Deeploy.DeeployTypes import _NoVerbosity
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.Targets.CortexM.Platform import CMSISPlatform
from Deeploy.Targets.PULPOpen.Platform import PULPClusterEngine, PULPPlatform


def generateNetwork(args):
    log.debug("Arguments: %s", args)

    onnx_graph = onnx.load_model(f'{args.dir}/network.onnx')
    graph = gs.import_onnx(onnx_graph)

    inputs = np.load(f'{args.dir}/inputs.npz')
    outputs = np.load(f'{args.dir}/outputs.npz')
    if os.path.isfile(f'{args.dir}/activations.npz'):
        activations = np.load(f'{args.dir}/activations.npz')
    else:
        activations = None

    # build {name, type} and {name, offset} maps
    manual_types = {}
    manual_offsets = {}
    for kv in args.input_type_map:
        try:
            name, tstr = kv.split('=', 1)
        except ValueError as exc:
            raise ValueError(f"Invalid --input-type-map entry '{kv}'. Expected NAME=TYPE.") from exc
        name, tstr = name.strip(), tstr.strip()
        try:
            manual_types[name] = parseDataType(tstr)
        except ValueError as exc:
            raise ValueError(f"Invalid --input-type-map entry '{kv}': {exc}") from exc
    for kv in args.input_offset_map:
        try:
            name, ostr = kv.split('=', 1)
        except ValueError as exc:
            raise ValueError(f"Invalid --input-offset-map entry '{kv}'. Expected NAME=OFFSET.") from exc
        name, ostr = name.strip(), ostr.strip()
        try:
            manual_offsets[name] = int(ostr)
        except ValueError as exc:
            raise ValueError(f"Invalid --input-offset-map entry '{kv}': OFFSET must be an integer.") from exc

    # Sanity check for unknown input names
    manual_keys = set(manual_types)
    assert manual_keys == set(
        manual_offsets
    ), f"Override inputs should have both type and offset specified. Inputs without both specified: {manual_keys ^ set(manual_types)}"
    assert manual_keys <= set(
        inputs.files
    ), f"Unknown input names in overrides: {manual_keys - set(inputs.files)} (Valid names are: {set(inputs.files)})"

    if args.debug:
        test_inputs, test_outputs, graph = generateDebugConfig(inputs, outputs, activations, graph)

    else:
        # Load as float64 for uniform handling, but preserve original dtypes for type inference
        test_input_original_dtypes = [inputs[x].dtype for x in inputs.files]
        test_inputs = [inputs[x].reshape(-1).astype(np.float64) for x in inputs.files]
        test_outputs = [outputs[x].reshape(-1).astype(np.float64) for x in outputs.files]

        # WIESEP: Hack to get CI running because only one specific array is used
        if "WaveFormer" in args.dir:
            test_inputs = [test_inputs[0]]
            test_outputs = [test_outputs[-2]]

    platform, signProp = mapPlatform(args.platform)

    clusters = [engine for engine in platform.engines if isinstance(engine, PULPClusterEngine)]
    for cluster in clusters:
        cluster.n_cores = args.cores

    inputTypes = {}
    inputOffsets = {}

    log.debug(f"Platform: {platform} (sign: {signProp})")

    log.debug("Platform Engines:")
    for engine in platform.engines:
        log.debug(f" - {engine.name}: {engine}")

    for index, (name, values) in enumerate(zip(inputs.files, test_inputs)):
        if np.prod(values.shape) == 0:
            continue

        if name in manual_keys:
            _type = manual_types[name]
            offset = manual_offsets[name]

            # Check if the provided values fit into the dereferenced type
            vals = values.astype(np.int64) - offset
            if not _type.checkPromotion(vals):
                lo, hi = _type.typeMin, _type.typeMax
                raise RuntimeError(f"Provided type '{_type.typeName}' with offset {offset} "
                                   f"does not match input values in range [{vals.min()}, {vals.max()}] "
                                   f"(expected range [{lo}, {hi}])")

            # Suggest a smaller fitting type if possible
            fitting_types = [t for t in sorted(IntegerDataTypes, key = lambda x: x.typeWidth) if t.checkPromotion(vals)]
            if fitting_types and fitting_types[0] is not _type:
                log.warning(f"Data spans [{int(vals.min())}, {int(vals.max())}], "
                            f"which would fit in '{fitting_types[0].typeName}', "
                            f"but user forced '{_type.typeName}'.")

            _type = PointerClass(_type)
        else:
            original_dtype = test_input_original_dtypes[index] if index < len(test_input_original_dtypes) else None
            _type, offset = inferTypeAndOffset(values, signProp, original_dtype = original_dtype)

        inputTypes[f"input_{index}"] = _type
        inputOffsets[f"input_{index}"] = offset

    _DEEPLOYSTATEDIR = os.path.join(args.dumpdir, "deeployStates")

    deployer = mapDeployer(platform, graph, inputTypes, deeployStateDir = _DEEPLOYSTATEDIR, inputOffsets = inputOffsets)

    log.debug(f"Deployer: {deployer}")

    if not isinstance(
            platform, CMSISPlatform
    ) and not "CNN_Linear1" in args.dir and not "GEMM/Regular_RQPerRow" in args.dir and not "MatMul/Regular_RQ" in args.dir:
        deployer.loweringOptimizer.passes.insert(0, EmulateCMSISRequantPass())

    verbosityCfg = _NoVerbosity
    if isinstance(platform, PULPPlatform):
        verbosityCfg.untiledProfiling = args.profileUntiled

    # Parse graph and infer output levels and signedness
    _ = deployer.prepare(verbosityCfg)

    # Offset the input and output values if signprop
    if signProp:
        test_inputs = [value - inputOffsets[f"input_{i}"] for i, value in enumerate(test_inputs)]

        for i, values in enumerate(test_outputs):
            buffer = deployer.ctxt.lookup(f"output_{i}")
            if buffer._type.referencedType.typeName == "float32_t":
                continue
            if not buffer._signed:
                values -= buffer.nLevels // 2

    generateTestNetwork(deployer, test_inputs, test_outputs, args.dumpdir, verbosityCfg)


if __name__ == '__main__':

    parser = TestGeneratorArgumentParser(description = "Deeploy Code Generation Utility.")
    parser.add_argument('--debug',
                        dest = 'debug',
                        action = 'store_true',
                        default = False,
                        help = 'Enable debugging mode\n')
    parser.add_argument('--profileUntiled',
                        action = 'store_true',
                        dest = 'profileUntiled',
                        default = False,
                        help = 'Profile Untiled for L2\n')
    parser.add_argument('--input-type-map',
                        nargs = '*',
                        default = [],
                        type = str,
                        help = '(Optional) mapping of input names to data types. '
                        'If not specified, types are inferred from the input data. '
                        'Example: --input-type-map input_0=int8_t input_1=float32_t ...')
    parser.add_argument('--input-offset-map',
                        nargs = '*',
                        default = [],
                        type = str,
                        help = '(Optional) mapping of input names to offsets. '
                        'If not specified, offsets are set to 0. '
                        'Example: --input-offset-map input_0=0 input_1=128 ...')
    parser.add_argument('--shouldFail', action = 'store_true')
    parser.add_argument(
        "--cores",
        type = int,
        default = 1,
        help =
        "Number of cores on which the network is run. Currently, required for im2col buffer sizing on Siracusa. Default: 1.",
    )
    parser.set_defaults(shouldFail = False)

    args = parser.parse_args()

    try:
        generateNetwork(args)
    except Exception as e:
        if args.shouldFail:
            print("\033[92mNetwork generation ended, failed as expected!\033[0m")
            sys.exit(0)
        else:
            raise e

    if args.shouldFail:
        raise RuntimeError("Expected to fail!")
