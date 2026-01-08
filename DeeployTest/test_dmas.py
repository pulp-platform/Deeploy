# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""DMA test suite for Siracusa and Snitch platforms.

Tests three DMA implementations across various tensor shapes and configurations:
- MchanDma: Siracusa L2→L1 DMA transfers
- L3Dma: Siracusa L3→L2 DMA transfers
- SnitchDma: Snitch L2→L1 DMA transfers

Total test matrix: 3 DMAs × 10 shapes × 2 buffering modes = 60 tests
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from testUtils.codeGenerate import generateTestNetwork
from testUtils.dmaUtils import (MemcpyLayer, MemcpyParser, MemcpyTileConstraint, MemcpyTypeChecker, generate_graph,
                                 memcpyTemplate, prepare_deployer_with_custom_tiling, setup_pulp_deployer,
                                 setup_snitch_deployer)
from testUtils.pytestRunner import build_binary, configure_cmake, get_test_paths, get_worker_id
from testUtils.typeMapping import baseTypeFromName, dtypeFromDeeployType

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import (ArgumentStructGeneration,
                                                                                 MemoryManagementGeneration)
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding, NodeMapper, _NoVerbosity
from Deeploy.Targets.PULPOpen.Bindings import L3MemoryAwareFunctionCallClosure
from Deeploy.Targets.PULPOpen.Bindings import MemoryAwareFunctionCallClosure as PULPMemoryAwareFunctionCallClosure
from Deeploy.Targets.PULPOpen.Bindings import TilingCallClosure as PULPTilingCallClosure
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPClusterTiling import PULPClusterTiling
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPL3Tiling import PULPL3Tiling
from Deeploy.Targets.PULPOpen.DMA.L3Dma import L3Dma, l3DmaHack
from Deeploy.Targets.PULPOpen.DMA.MchanDma import MchanDma
from Deeploy.Targets.Snitch.Bindings import MemoryAwareFunctionCallClosure, TilingCallClosure
from Deeploy.Targets.Snitch.CodeTransformationPasses import SnitchClusterTiling
from Deeploy.Targets.Snitch.CodeTransformationPasses.SnitchClusterSynch import SnitchSynchCoresPass
from Deeploy.Targets.Snitch.CodeTransformationPasses.SnitchCoreFilter import SnitchCoreFilterPass
from Deeploy.Targets.Snitch.CodeTransformationPasses.SnitchProfileExecutionBlock import SnitchProfileExecutionBlockPass
from Deeploy.Targets.Snitch.DMA.SnitchDma import SnitchDma
from Deeploy.TilingExtension.CodeTransformationPasses.TilingVariableReplacement import (
    TilingVariableReplacement, TilingVariableReplacementUpdate)
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings


@pytest.fixture(autouse=True)
def clear_deeploy_state():
    """Clear dynamically generated struct classes from AbstractDataTypes before each test.
    
    This prevents state pollution between DMA tests where dynamically generated
    struct classes (like _memcpy_0_tiling_closure_args_t) persist and cause
    conflicts when tests with different configurations try to create new versions.
    """
    import Deeploy.AbstractDataTypes as ADT
    
    # Get list of all attributes before test
    attrs_to_remove = []
    for attr_name in dir(ADT):
        # Remove dynamically generated struct classes (closure args, etc.)
        if attr_name.startswith('_') and ('closure_args' in attr_name or 'memcpy' in attr_name.lower()):
            attr = getattr(ADT, attr_name, None)
            if isinstance(attr, type):
                attrs_to_remove.append(attr_name)
    
    # Remove stale struct classes
    for attr_name in attrs_to_remove:
        delattr(ADT, attr_name)
    
    yield  # Run the test
    
    # Clean up after test as well
    for attr_name in dir(ADT):
        if attr_name.startswith('_') and ('closure_args' in attr_name or 'memcpy' in attr_name.lower()):
            attr = getattr(ADT, attr_name, None)
            if isinstance(attr, type):
                try:
                    delattr(ADT, attr_name)
                except AttributeError:
                    pass


# Test shape configurations: (input_shape, tile_shape, node_count, data_type)
DMA_TEST_SHAPES = [
    ((10, 10), (10, 10), 1, "uint8_t"),
    ((10, 10), (10, 4), 1, "uint8_t"),
    ((10, 10), (10, 4), 1, "uint16_t"),
    ((10, 10), (10, 4), 1, "uint32_t"),
    ((10, 10), (3, 4), 1, "uint32_t"),
    ((10, 10), (3, 4), 2, "uint32_t"),
    ((10, 10, 10), (2, 3, 4), 1, "uint8_t"),
    ((10, 10, 10, 10), (2, 3, 5, 4), 1, "uint8_t"),
    ((10, 10, 10, 10), (2, 3, 5, 4), 1, "uint32_t"),
    ((10, 10, 10, 10, 10), (2, 3, 5, 7, 4), 1, "uint8_t"),
]


def param_id_dma(val):
    """Generate readable test IDs for DMA parametrized tests."""
    if isinstance(val, tuple) and len(val) == 4:
        input_shape, tile_shape, node_count, data_type = val
        shape_str = "x".join(map(str, input_shape))
        tile_str = "x".join(map(str, tile_shape))
        return f"{shape_str}_tile{tile_str}_n{node_count}_{data_type}"
    elif isinstance(val, bool):
        return "doublebuffer" if val else "singlebuffer"
    return str(val)


def setup_dma_deployer(dma_type: str, input_shape: tuple, tile_shape: tuple, node_count: int, data_type: str,
                        doublebuffer: bool, gen_dir: str):
    """
    Set up deployer for DMA testing with custom tiling.
    
    Args:
        dma_type: DMA implementation ("MchanDma", "L3Dma", "SnitchDma")
        input_shape: Tensor shape to copy
        tile_shape: Tiling dimensions
        node_count: Number of memcpy nodes
        data_type: Data type (uint8_t, uint16_t, uint32_t)
        doublebuffer: Enable double buffering
        gen_dir: Generation directory
        
    Returns:
        tuple: (deployer, test_inputs, test_outputs)
    """
    _type = baseTypeFromName(data_type)
    dtype = dtypeFromDeeployType(_type)

    # Validate shapes
    assert len(input_shape) == len(tile_shape), \
        f'Input and tile shape must have same dimensionality: {len(input_shape)}D vs {len(tile_shape)}D'
    assert all(tileDim <= inDim for inDim, tileDim in zip(input_shape, tile_shape)), \
        f'Tile shape {tile_shape} must be <= input shape {input_shape}'

    # DMA-specific configuration
    if dma_type == "MchanDma":
        defaultMemory = "L2"
        targetMemory = "L1"
        dma_obj = MchanDma()
    elif dma_type == "L3Dma":
        defaultMemory = "L3"
        targetMemory = "L2"
        dma_obj = L3Dma()
    elif dma_type == "SnitchDma":
        defaultMemory = "L2"
        targetMemory = "L1"
        dma_obj = SnitchDma()
    else:
        raise ValueError(f"Unknown DMA type: {dma_type}")

    # Generate graph and setup deployer
    graph = generate_graph(node_count, input_shape, dtype)
    inputTypes = {"input_0": PointerClass(_type)}
    _DEEPLOYSTATEDIR = os.path.join(gen_dir, "deeployStates")
    
    if dma_type == "SnitchDma":
        deployer = setup_snitch_deployer(defaultMemory, targetMemory, graph, inputTypes, doublebuffer, _DEEPLOYSTATEDIR)
    else:
        deployer = setup_pulp_deployer(defaultMemory, targetMemory, graph, inputTypes, doublebuffer, _DEEPLOYSTATEDIR)

    # Create transformer with DMA-specific passes
    if dma_type == "SnitchDma":
        transformer = CodeTransformation([
            SnitchCoreFilterPass("compute"),
            SnitchProfileExecutionBlockPass(),
            TilingVariableReplacement(targetMemory),
            TilingCallClosure(writeback = False),
            SnitchSynchCoresPass(),
            TilingVariableReplacementUpdate(targetMemory),
            SnitchClusterTiling(defaultMemory, targetMemory, dma_obj),
            ArgumentStructGeneration(),
            MemoryManagementGeneration(targetMemory),
            MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
            MemoryManagementGeneration(defaultMemory),
            MemoryManagementGeneration(),
        ])
    elif dma_type == "L3Dma":
        # L3Dma uses PULPL3Tiling and L3MemoryAwareFunctionCallClosure
        transformer = CodeTransformation([
            TilingVariableReplacement(targetMemory),
            PULPTilingCallClosure(writeback = False, generateStruct = True),
            TilingVariableReplacementUpdate(targetMemory),
            PULPL3Tiling("L3", "L2", l3DmaHack),
            ArgumentStructGeneration(),
            L3MemoryAwareFunctionCallClosure(writeback = False),
            MemoryManagementGeneration("L2"),
            MemoryManagementGeneration("L3.*"),
            MemoryManagementGeneration(),
        ])
    else:  # MchanDma
        transformer = CodeTransformation([
            TilingVariableReplacement(targetMemory),
            PULPTilingCallClosure(writeback = False, generateStruct = True),
            TilingVariableReplacementUpdate(targetMemory),
            PULPClusterTiling(defaultMemory, targetMemory, dma_obj),
            ArgumentStructGeneration(),
            MemoryManagementGeneration(targetMemory),
            TilingVariableReplacement(defaultMemory),
            PULPMemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
            MemoryManagementGeneration(defaultMemory),
            MemoryManagementGeneration(),
        ])

    # Set up bindings
    binding = NodeBinding(MemcpyTypeChecker(), memcpyTemplate, transformer)
    tilingReadyBindings = TilingReadyNodeBindings([binding], MemcpyTileConstraint())
    memcpyMapper = NodeMapper(MemcpyParser(), tilingReadyBindings)
    memcpyMapping = {"Memcpy": MemcpyLayer([memcpyMapper])}
    deployer.Platform.engines[0].Mapping.update(memcpyMapping)

    # Prepare custom tiling
    prepare_deployer_with_custom_tiling(deployer, defaultMemory, targetMemory, tile_shape, doublebuffer)

    # Generate test inputs/outputs
    if dtype == np.float32:
        test_inputs = np.random.rand(*input_shape)
    else:
        test_inputs = np.arange(stop = np.prod(input_shape), dtype = dtype).reshape(input_shape)
    test_outputs = test_inputs

    return deployer, test_inputs, test_outputs


@pytest.mark.dma
@pytest.mark.parametrize("test_shape", DMA_TEST_SHAPES, ids = param_id_dma)
@pytest.mark.parametrize("doublebuffer", [True, False], ids = param_id_dma)
def test_mchan_dma(test_shape, doublebuffer, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen,
                   skipsim) -> None:
    """Test MchanDma (Siracusa L2→L1 DMA transfers)."""
    input_shape, tile_shape, node_count, data_type = test_shape

    # Setup paths
    test_name = f"testMchanDma_{param_id_dma(test_shape)}_{param_id_dma(doublebuffer)}"
    platform = "Siracusa"
    gen_dir, _, test_name_clean = get_test_paths(f"test_dma_gen/{test_name}", platform, base_dir = deeploy_test_dir)

    # Generate network
    if not skipgen:
        deployer, test_inputs, test_outputs = setup_dma_deployer("MchanDma", input_shape, tile_shape, node_count,
                                                                   data_type, doublebuffer, gen_dir)
        generateTestNetwork(deployer, [test_inputs], [test_outputs], gen_dir, _NoVerbosity)

    # Build and run
    worker_id = get_worker_id()
    if worker_id == "master":
        build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / "build_master")
    else:
        build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / f"build_{worker_id}")

    from testUtils.pytestRunner import DeeployTestConfig
    config = DeeployTestConfig(
        test_name = test_name_clean,
        test_dir = gen_dir,
        platform = platform,
        simulator = 'gvsoc',
        tiling = True,
        gen_dir = gen_dir,
        build_dir = build_dir,
        toolchain = toolchain,
        toolchain_install_dir = toolchain_dir,
        cmake_args = list(cmake_args) + ["NUM_CORES=8"],
    )

    configure_cmake(config)
    build_binary(config)

    if not skipsim:
        from testUtils.pytestRunner import run_simulation
        result = run_simulation(config)
        assert result.success, f"MchanDma test failed with {result.error_count} errors"
        assert result.error_count == 0, f"Found {result.error_count} errors"


@pytest.mark.dma
@pytest.mark.parametrize("test_shape", DMA_TEST_SHAPES, ids = param_id_dma)
@pytest.mark.parametrize("doublebuffer", [True, False], ids = param_id_dma)
def test_l3_dma(test_shape, doublebuffer, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen,
                skipsim) -> None:
    """Test L3Dma (Siracusa L3→L2 DMA transfers)."""
    input_shape, tile_shape, node_count, data_type = test_shape

    # Setup paths
    test_name = f"testL3Dma_{param_id_dma(test_shape)}_{param_id_dma(doublebuffer)}"
    platform = "Siracusa"
    gen_dir, _, test_name_clean = get_test_paths(f"test_dma_gen/{test_name}", platform, base_dir = deeploy_test_dir)

    # Generate network
    if not skipgen:
        deployer, test_inputs, test_outputs = setup_dma_deployer("L3Dma", input_shape, tile_shape, node_count,
                                                                   data_type, doublebuffer, gen_dir)
        generateTestNetwork(deployer, [test_inputs], [test_outputs], gen_dir, _NoVerbosity)

    # Build and run
    worker_id = get_worker_id()
    if worker_id == "master":
        build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / "build_master")
    else:
        build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / f"build_{worker_id}")

    from testUtils.pytestRunner import DeeployTestConfig
    config = DeeployTestConfig(
        test_name = test_name_clean,
        test_dir = gen_dir,
        platform = platform,
        simulator = 'gvsoc',
        tiling = True,
        gen_dir = gen_dir,
        build_dir = build_dir,
        toolchain = toolchain,
        toolchain_install_dir = toolchain_dir,
        cmake_args = list(cmake_args) + ["NUM_CORES=8"],
    )

    configure_cmake(config)
    build_binary(config)

    if not skipsim:
        from testUtils.pytestRunner import run_simulation
        result = run_simulation(config)
        assert result.success, f"L3Dma test failed with {result.error_count} errors"
        assert result.error_count == 0, f"Found {result.error_count} errors"


@pytest.mark.dma
@pytest.mark.parametrize("test_shape", DMA_TEST_SHAPES, ids = param_id_dma)
@pytest.mark.parametrize("doublebuffer", [True, False], ids = param_id_dma)
def test_snitch_dma(test_shape, doublebuffer, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen,
                    skipsim) -> None:
    """Test SnitchDma (Snitch L2→L1 DMA transfers)."""
    input_shape, tile_shape, node_count, data_type = test_shape

    # Setup paths
    test_name = f"testSnitchDma_{param_id_dma(test_shape)}_{param_id_dma(doublebuffer)}"
    platform = "Snitch"
    gen_dir, _, test_name_clean = get_test_paths(f"test_dma_gen/{test_name}", platform, base_dir = deeploy_test_dir)

    # Generate network
    if not skipgen:
        deployer, test_inputs, test_outputs = setup_dma_deployer("SnitchDma", input_shape, tile_shape, node_count,
                                                                   data_type, doublebuffer, gen_dir)
        generateTestNetwork(deployer, [test_inputs], [test_outputs], gen_dir, _NoVerbosity)

    # Build and run
    worker_id = get_worker_id()
    if worker_id == "master":
        build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / "build_master")
    else:
        build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / f"build_{worker_id}")

    from testUtils.pytestRunner import DeeployTestConfig
    config = DeeployTestConfig(
        test_name = test_name_clean,
        test_dir = gen_dir,
        platform = platform,
        simulator = 'gvsoc',
        tiling = True,
        gen_dir = gen_dir,
        build_dir = build_dir,
        toolchain = toolchain,
        toolchain_install_dir = toolchain_dir,
        cmake_args = list(cmake_args) + ["NUM_CORES=9"],
    )

    configure_cmake(config)
    build_binary(config)

    if not skipsim:
        from testUtils.pytestRunner import run_simulation
        result = run_simulation(config)
        assert result.success, f"SnitchDma test failed with {result.error_count} errors"
        assert result.error_count == 0, f"Found {result.error_count} errors"
