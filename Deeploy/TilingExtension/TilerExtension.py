# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Create Monad that take a Deployer and make it TilerAware
# Define Tiler Obj centralize all tilling related functionalities for a given deployer.
# Like Template-T-Obj mapping, propagate cst, graph edition, etc

import copy
import csv
import os
import subprocess
from typing import Dict, List, Literal, Optional, OrderedDict, Tuple, Type, Union

import numpy as np
import onnx_graphsurgeon as gs
import plotly.graph_objects as go
import plotly.io as pio
from ortools.constraint_solver.pywrapcp import IntVar, SolutionCollector

import Deeploy.CommonExtensions.DataTypes as BasicDataTypes
from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.NetworkDeployers.NetworkDeployerWrapper import NetworkDeployerWrapper
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeBinding, NodeTemplate, ONNXLayer, Schedule, \
    SubGraph, TransientBuffer
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.Logging import SUCCESS_MARK
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryDeployerWrapper, \
    MemoryLevelAwareDeployer, MemoryPlatform, MemoryPlatformWrapper, TargetMemoryLevelMapping
from Deeploy.TilingExtension.GenericFlow import GenericFlowState
from Deeploy.TilingExtension.MemoryConstraintFlows import GraphMemoryConstraintFlow, TensorMemLevelTuple, \
    convertFlowState2NodeMemoryConstraint
from Deeploy.TilingExtension.MemoryConstraints import MemoryConstraint, NodeMemoryConstraint, \
    PatternMemoryConstraints, TensorMemoryConstraint
from Deeploy.TilingExtension.MemoryScheduler import MemoryBlock, MemoryScheduler
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel

TilingSolution = List[PatternMemoryConstraints]
MemoryMap = Dict[str, List[List[MemoryBlock]]]

_deallocTemplate = NodeTemplate("")


class Tiler():
    """Tiler for a computation graphs with memory-aware optimization.

    The Tiler class provides functionality for tiling operations to fit within
    memory constraints of target hardware platforms. It performs memory allocation, constraint
    solving, and scheduling to optimize execution within hierarchical memory systems.

    Parameters
    ----------
    memoryHierarchy : MemoryHierarchy
        The memory hierarchy specification defining available memory levels and their capacities.

    Attributes
    ----------
    arenaName : str
        Name prefix for memory arena buffers.
    memorySchedulerClass : Type[MemoryScheduler]
        Class type for memory scheduler instances.
    memoryHierarchy : MemoryHierarchy
        The memory hierarchy configuration.
    tilerModel : Optional[TilerModel]
        The constraint solver model for tiling optimization.
    innerMemoryScheduler : MemoryScheduler
        Scheduler for inner memory level allocation.
    outerMemoryScheduler : MemoryScheduler
        Scheduler for outer memory level allocation.
    symbolicMemoryConstraints : Optional[List[PatternMemoryConstraints]]
        Symbolic memory constraints for the tiling problem.
    visualizeMemoryAlloc : bool
        Flag to enable memory allocation visualization.
    memoryAllocStrategy : {"TetrisRandom", "TetrisCo-Opt", "MiniMalloc"}
        Strategy for memory allocation.
    searchStrategy : {"min", "max", "random-max"}
        Search strategy for constraint solving.

    Examples
    --------
    >>> L3 = MemoryLevel(name = "L3", neighbourNames = ["L2"], size = 1024000)
    >>> L2 = MemoryLevel(name = "L2", neighbourNames = ["L3", "L1"], size = 512000)
    >>> L1 = MemoryLevel(name = "L1", neighbourNames = ["L2"], size = 128000)
    >>> memoryHierarchy = MemoryHierarchy([L3, L2, L1])
    >>> memoryHierarchy.setDefaultMemoryLevel("L3")
    >>> tiler = Tiler(hierarchy)
    >>> tiler.memoryAllocStrategy = "MiniMalloc"
    >>> solution = tiler.computeTilingSchedule(context)
    """

    arenaName = "MEMORYARENA"
    memorySchedulerClass: Type[MemoryScheduler] = MemoryScheduler

    _MINIMALLOC_INPUT_FILENAME = "input_minimalloc"
    _MINIMALLOC_OUTPUT_FILENAME = "output_minimalloc"

    # Initialize with the list of TemplateTCFbinding
    def __init__(self, memoryHierarchy: MemoryHierarchy, testName: Optional[str] = None, workDir: Optional[str] = None):
        """Initialize the Tiler with a memory hierarchy.

        Parameters
        ----------
        memoryHierarchy : MemoryHierarchy
            The memory hierarchy specification defining available memory levels.
        testName : Optional[str], optional
            Optional name for the test case, used for file naming. Defaults to None.
        workDir : Optional[str], optional
            Optional working directory for temporary files. Defaults to None.
        """

        self.memoryHierarchy = memoryHierarchy
        self.tilerModel: Optional[TilerModel] = None
        self.innerMemoryScheduler = self.memorySchedulerClass("_inner", tileScheduler = True)
        self.outerMemoryScheduler = self.memorySchedulerClass("_outer", tileScheduler = False)
        self.symbolicMemoryConstraints: Optional[List[PatternMemoryConstraints]] = None

        self._worstCaseBufferSize: Dict[str, int] = {}

        self.visualizeMemoryAlloc: bool = False
        self.memoryAllocStrategy: Literal["TetrisRandom", "TetrisCo-Opt", "MiniMalloc"] = "TetrisRandom"
        self.searchStrategy: Literal["min", "max", "random-max"] = "random-max"

        if workDir is not None:
            os.makedirs(workDir, exist_ok = True)
            minimalloc_base = os.path.join(workDir, self._MINIMALLOC_INPUT_FILENAME)
            minimalloc_output_base = os.path.join(workDir, self._MINIMALLOC_OUTPUT_FILENAME)
        else:
            minimalloc_base = self._MINIMALLOC_INPUT_FILENAME
            minimalloc_output_base = self._MINIMALLOC_OUTPUT_FILENAME

        if testName is not None:
            # VJUNG: Sanitize path
            safe_test_name = testName.replace("/", "_").replace("\\", "_")
            self._minimalloc_input = f"{minimalloc_base}_{safe_test_name}"
            self._minimalloc_output = f"{minimalloc_output_base}_{safe_test_name}"
        else:
            self._minimalloc_input = minimalloc_base
            self._minimalloc_output = minimalloc_output_base

    @property
    def worstCaseBufferSize(self):
        """Get the worst-case buffer sizes for each memory level.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping memory level names to their worst-case buffer sizes in bytes.
        """
        return self._worstCaseBufferSize

    def plotMemoryAlloc(self, memoryMap: Dict[str, List[List[MemoryBlock]]], ctxt: NetworkContext, deeployStateDir: str,
                        memoryHierarchy: MemoryHierarchy):
        """Generate interactive visualization of memory allocation patterns.

        Creates an HTML file with Plotly visualizations showing memory allocation
        over time for each memory level in the hierarchy.

        Parameters
        ----------
        memoryMap : Dict[str, List[List[MemoryBlock]]]
            Memory allocation map containing blocks for each memory level and time step.
        ctxt : NetworkContext
            Network context containing buffer information.
        deeployStateDir : str
            Directory path where the visualization HTML file will be saved.
        memoryHierarchy : MemoryHierarchy
            Memory hierarchy configuration for the visualization.

        Notes
        -----
        Generates a file named 'memory_alloc.html' in the specified directory.
        Each memory level is visualized as a separate subplot showing buffer
        lifetimes and address space usage.
        """

        os.makedirs(os.path.abspath(deeployStateDir), exist_ok = True)
        memoryAllocPlotPath = os.path.abspath(os.path.join(deeployStateDir, f"memory_alloc.html"))

        addTraceConfig = {"fill": "toself", "hoverinfo": "text", "mode": "lines", "line": dict(width = 2)}

        def plotSingleMemoryLevel(memoryLevel: MemoryLevel):
            """ Generates a single Plotly subplot for a memory level. """
            fig = go.Figure()
            constantBuffersOffset = 0

            infiniteLifetimeBuffers = [
                buffer for buffer in ctxt.globalObjects.values()
                if not self.arenaName in buffer.name and isinstance(buffer, ConstantBuffer)
            ]

            constantBuffersOffset = 0
            for ioBuffer in infiniteLifetimeBuffers:
                if not ioBuffer._memoryLevel == memoryLevel.name:
                    continue
                _ioSize = np.prod(ioBuffer.shape) * ioBuffer._type.referencedType.typeWidth // 8
                _maxLifetime = len(memoryMap[memoryLevel.name])
                fig.add_trace(
                    go.Scatter(x = [-0.5, -0.5, _maxLifetime + 0.5, _maxLifetime + 0.5],
                               y = [
                                   constantBuffersOffset, constantBuffersOffset + _ioSize,
                                   constantBuffersOffset + _ioSize, constantBuffersOffset
                               ],
                               name = ioBuffer.name,
                               text = ioBuffer.name,
                               **addTraceConfig))
                constantBuffersOffset += _ioSize

            for memoryMapStep in memoryMap[memoryLevel.name]:
                for buffer in memoryMapStep:
                    if not hasattr(buffer, "_addrSpace") or buffer._addrSpace is None:
                        log.warning(
                            f"Buffer {buffer.name} has no address space assigned, skipping it in the memory allocation plot."
                        )
                        continue

                    fig.add_trace(
                        go.Scatter(x = [
                            buffer._lifetime[0] - 0.5, buffer._lifetime[0] - 0.5, buffer._lifetime[1] + 0.5,
                            buffer._lifetime[1] + 0.5
                        ],
                                   y = [
                                       constantBuffersOffset + buffer._addrSpace[0],
                                       constantBuffersOffset + buffer._addrSpace[1],
                                       constantBuffersOffset + buffer._addrSpace[1],
                                       constantBuffersOffset + buffer._addrSpace[0]
                                   ],
                                   name = buffer.name,
                                   text = buffer.name,
                                   **addTraceConfig))

            fig.update_xaxes(title_text = "Lifetime")
            fig.update_yaxes(title_text = "Address Space (Bytes)")
            fig.update_layout(title = f"Memory Allocation - {memoryLevel.name}", showlegend = False)

            fig.add_trace(
                go.Scatter(
                    x = [-0.5, len(memoryMap[memoryLevel.name]) - 1.5],
                    y = [memoryLevel.size, memoryLevel.size],
                    name = f"{memoryLevel.name} Memory Size",
                    text = f"{memoryLevel.name} Memory Size",
                    line = dict(color = "red", width = 2, dash = "dash"),
                    fill = "toself",
                    hoverinfo = "text",
                    mode = "lines",
                ))

            return fig

        from Deeploy.TilingExtension.HtmlTemplates import getHtmlMemoryAllocationVisualisation, getSubplotHtml

        subplotHtml = ""
        for memoryLevelName in memoryMap.keys():
            figJson = pio.to_json(plotSingleMemoryLevel(memoryHierarchy.memoryLevels[memoryLevelName]))
            subplotHtml += getSubplotHtml(figJson, memoryLevelName)

        outputHtml = getHtmlMemoryAllocationVisualisation(subplotHtml)

        with open(memoryAllocPlotPath, "w", encoding = "utf-8") as f:
            f.write(outputHtml)

    def _convertCtxtToStaticSchedule(self, ctxt: NetworkContext,
                                     memoryMap: Dict[str, List[List[MemoryBlock]]]) -> NetworkContext:
        """Convert network context to use static memory allocation.

        Transforms the network context to use statically allocated memory arenas
        based on the computed memory map. Updates buffer allocation templates
        to reference specific offsets within memory arenas.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context to be updated.
        memoryMap : Dict[str, List[List[MemoryBlock]]]
            Memory allocation map containing blocks for each memory level.

        Returns
        -------
        NetworkContext
            Updated network context with static memory allocation.

        Notes
        -----
        Creates memory arena buffers for each memory level and updates
        individual buffer allocation templates to use offsets within these arenas.
        """

        maxAddr: Dict[str, int] = {}

        for memoryLevel, patternList in memoryMap.items():
            currentMax = 0
            for nodeList in patternList:
                blockNames = [block.name for block in nodeList]
                for node in nodeList:

                    _buffer = ctxt.lookup(node.name)
                    # SCHEREMO: If alias buffers have zero cost, they don't contribute to the currentMax and their addrSpace is None
                    if hasattr(_buffer, "_alias") and (ctxt.is_global(_buffer._alias) or _buffer._alias in blockNames):
                        continue

                    currentMax = max(currentMax, node._addrSpace[1])

            maxAddr[memoryLevel] = currentMax
            self._worstCaseBufferSize[memoryLevel] = currentMax

        for level, addrSpace in maxAddr.items():
            if addrSpace == 0:
                continue

            arenaName = f"{self.arenaName}_{level}"

            scratchBuffer = ctxt.VariableBuffer(arenaName, [addrSpace])
            scratchBuffer._type = PointerClass(BasicDataTypes.int8_t)
            ctxt.add(scratchBuffer, "global")
            scratchBuffer._instance = scratchBuffer._type(arenaName, ctxt)
            scratchBuffer._memoryLevel = level

            # JUNGVI: Memory Arena buffers should be allocated first since other variable global buffers may belong to a memory arena
            ctxt.globalObjects.move_to_end(scratchBuffer.name, last = False)

        # SCHEREMO: Adapt homelevel tensors to their respective arena
        for memoryLevel, patternList in memoryMap.items():
            if not ctxt.is_global(f"{self.arenaName}_{memoryLevel}"):
                continue
            staticBuf = ctxt.lookup(f"{self.arenaName}_{memoryLevel}")
            for nodeList in patternList:
                blockNames = [block.name for block in nodeList]
                for node in nodeList:
                    tensorName = node.name
                    _buffer = ctxt.lookup(tensorName)

                    if _buffer._memoryLevel != memoryLevel:
                        continue

                    if hasattr(_buffer, "_alias") and ctxt.is_global(_buffer._alias):
                        continue

                    if hasattr(_buffer, "_alias") and _buffer._alias in blockNames:

                        alias = ctxt.dealiasBuffer(tensorName)
                        aliasNodes = [node for node in nodeList if node.name == alias]

                        assert len(aliasNodes) == 1, f"alias {alias} references more than one node!"

                        aliasNode = aliasNodes[0]

                        _buffer.allocTemplate = NodeTemplate(
                            " \
                        ${name} = (${type.typeName}) " +
                            f"((char*){str(staticBuf._instance)} + {aliasNode.addrSpace[0]});")
                        _buffer.deallocTemplate = _deallocTemplate

                        continue

                    offset = node.addrSpace[0]

                    _buffer.allocTemplate = NodeTemplate(" \
                    ${name} = (${type.typeName}) " + f"((char*){str(staticBuf._instance)} + {offset});")
                    _buffer.deallocTemplate = _deallocTemplate

        return ctxt

    def minimalloc(self, memoryMap, ctxt, nodeMemoryConstraint, capacity: int, memoryLevel: str):
        """Perform memory allocation using the MiniMalloc external tool.

        Interfaces with the external MiniMalloc memory allocator to compute
        optimal memory allocation for the given memory blocks and constraints.

        Parameters
        ----------
        memoryMap : List[MemoryBlock]
            List of memory blocks to be allocated.
        ctxt : NetworkContext
            Network context containing buffer information.
        nodeMemoryConstraint : Optional[NodeMemoryConstraint]
            Memory constraints for the current node, if available.
        capacity : int
            Total memory capacity available for allocation.
        memoryLevel : str
            Name of the memory level being allocated.

        Returns
        -------
        List[MemoryBlock]
            Updated memory blocks with assigned address spaces.

        Raises
        ------
        KeyError
            If MINIMALLOC_INSTALL_DIR environment variable is not set.
        subprocess.CalledProcessError
            If the MiniMalloc tool fails to execute successfully.

        Notes
        -----
        Requires the MiniMalloc tool to be installed and the MINIMALLOC_INSTALL_DIR
        environment variable to be set to the installation directory.
        """

        with open(f"{self._minimalloc_input}.csv", mode = "w", newline = "") as file:
            writer = csv.writer(file, lineterminator = "\n")
            writer.writerow(["id", "lower", "upper", "size"])
            for memoryBlock in memoryMap:

                _buffer = ctxt.lookup(memoryBlock.name)
                if nodeMemoryConstraint is None:
                    _bufferSize = _buffer.size if isinstance(
                        _buffer,
                        TransientBuffer) else np.prod(_buffer.shape) * (_buffer._type.referencedType.typeWidth / 8)
                else:
                    if isinstance(_buffer, TransientBuffer):
                        _bufferSize = nodeMemoryConstraint.tensorMemoryConstraints[
                            memoryBlock.name].memoryConstraints[memoryLevel].size
                    else:
                        _bufferSize = nodeMemoryConstraint.tensorMemoryConstraints[
                            memoryBlock.name].memoryConstraints[memoryLevel].size * (
                                _buffer._type.referencedType.typeWidth /
                                8) * nodeMemoryConstraint.tensorMemoryConstraints[
                                    memoryBlock.name].memoryConstraints[memoryLevel].multiBufferCoefficient

                writer.writerow([
                    memoryBlock.name,
                    str(memoryBlock.lifetime[0]),
                    str(memoryBlock.lifetime[1] + 1),
                    str(int(_bufferSize))
                ])

        try:
            minimallocInstallDir = os.environ["MINIMALLOC_INSTALL_DIR"]
        except KeyError:
            raise KeyError("MINIMALLOC_INSTALL_DIR symbol not found!")

        minimallocOutput = subprocess.run([
            f"{minimallocInstallDir}/minimalloc", f"--capacity={capacity}", f"--input={self._minimalloc_input}.csv",
            f"--output={self._minimalloc_output}.csv"
        ],
                                          capture_output = True,
                                          text = True)

        if minimallocOutput.returncode != 0:
            log.error(
                f"Memory allocator failed with return code {minimallocOutput.returncode} at memory level {memoryLevel} with capacity of {capacity} bytes!"
            )
            raise subprocess.CalledProcessError(minimallocOutput.returncode, " ".join(minimallocOutput.args))

        with open(f"{self._minimalloc_output}.csv", mode = "r", newline = "") as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                for memoryBlock in memoryMap:
                    if memoryBlock.name == row[0]:
                        memoryBlock._addrSpace = (int(row[-1]), int(row[-1]) + int(row[-2]))

        return memoryMap

    def computeTilingSchedule(self, ctxt: NetworkContext) -> TilingSolution:
        """Compute the optimal tiling schedule for the network.

        Solves the constraint optimization problem to find the best tiling
        solution that satisfies memory and computational constraints.

        Parameters
        ----------
        ctxt : NetworkContext
            Network context containing the computational graph and constraints.

        Returns
        -------
        TilingSolution
            The computed tiling solution with memory constraints for each pattern.

        Raises
        ------
        AssertionError
            If the tiler model or symbolic memory constraints are not initialized.

        Notes
        -----
        This method requires that setupModel() has been called previously to
        initialize the constraint model and symbolic memory constraints.
        """
        assert self.tilerModel is not None and self.symbolicMemoryConstraints is not None, "Set up the model before trying to compute a schedule!"
        collector = self.tilerModel.trySolveModel()
        tilingSolution = self._getTilingSolution(self.tilerModel, ctxt, collector, self.symbolicMemoryConstraints)
        if not self.memoryAllocStrategy == "MiniMalloc":
            assert self.tilerModel is not None
            log.debug(" - Extract Memory Allocation")
            self.innerMemoryScheduler.annotateSolution(ctxt, self.tilerModel)
            self.outerMemoryScheduler.annotateSolution(ctxt, self.tilerModel)
        return tilingSolution

    def computeMemoryMap(self, ctxt: NetworkContext, tilingSolution: TilingSolution) -> MemoryMap:
        """Compute memory allocation map from the tiling solution.

        Generates a concrete memory allocation map that assigns specific
        memory addresses to each buffer based on the tiling solution.

        Parameters
        ----------
        ctxt : NetworkContext
            Network context containing buffer information.
        tilingSolution : TilingSolution
            The computed tiling solution.

        Returns
        -------
        MemoryMap
            Dictionary mapping memory level names to lists of memory blocks
            for each time step.

        Notes
        -----
        The memory allocation strategy (TetrisRandom, TetrisCo-Opt, or MiniMalloc)
        determines how the actual memory addresses are assigned.
        """
        memoryMap = {}

        for key in self.innerMemoryScheduler.memoryMap.keys():
            memoryMap[key] = [*self.innerMemoryScheduler.memoryMap[key], *self.outerMemoryScheduler.memoryMap[key]]

        if self.memoryAllocStrategy == "MiniMalloc":
            log.debug(" - Solve Memory Allocation with MiniMalloc")
            for memoryLevel in memoryMap.keys():
                constantTensorOffset = self.outerMemoryScheduler.getConstantTensorOffset(ctxt, memoryLevel)
                if memoryLevel == self.memoryHierarchy._defaultMemoryLevel.name:
                    memoryMap[memoryLevel][-1] = self.minimalloc(
                        memoryMap[memoryLevel][-1], ctxt, None,
                        self.memoryHierarchy.memoryLevels[memoryLevel].size - constantTensorOffset, memoryLevel)
                else:
                    for idx, memMap in enumerate(memoryMap[memoryLevel]):
                        if len(memoryMap[memoryLevel][idx]) != 0:
                            memoryMap[memoryLevel][idx] = self.minimalloc(
                                memMap, ctxt, tilingSolution[idx].nodeConstraints[0],
                                self.memoryHierarchy.memoryLevels[memoryLevel].size - constantTensorOffset, memoryLevel)
            log.info(f" {SUCCESS_MARK} Memory allocation successful!")

        return memoryMap

    def annotateMemoryLevel(self, ctxt: NetworkContext, tilingSolution: TilingSolution,
                            memoryMap: Dict) -> NetworkContext:
        """Annotate memory constraints with actual address space allocations.

        Updates the memory constraints in the tiling solution with the actual
        address spaces computed during memory allocation.

        Parameters
        ----------
        ctxt : NetworkContext
            Network context containing buffer information.
        tilingSolution : TilingSolution
            The tiling solution to be annotated.
        memoryMap : Dict[str, List[List[MemoryBlock]]]
            Memory allocation map with assigned address spaces.

        Returns
        -------
        NetworkContext
            Updated network context (returned for consistency).

        Notes
        -----
        This method modifies the tiling solution in-place by adding address
        space information to memory constraints.
        """
        for idx, pattern in enumerate(tilingSolution):
            for nodeIdx, nodeConstraint in enumerate(pattern.nodeConstraints):
                for tensorConstraint in nodeConstraint.tensorMemoryConstraints.values():
                    for memoryConstraint in tensorConstraint.memoryConstraints.values():
                        patternList = memoryMap[memoryConstraint.memoryLevel]
                        blockPattern = patternList[idx]

                        # SCHEREMO: Don't try to annotate home base of tensor
                        if ctxt.lookup(tensorConstraint.tensorName
                                      )._memoryLevel == memoryConstraint.memoryLevel and not isinstance(
                                          ctxt.lookup(tensorConstraint.tensorName), TransientBuffer):
                            continue

                        _block = [memBlock for memBlock in blockPattern if memBlock.name == tensorConstraint.tensorName]

                        assert len(
                            _block
                        ) == 1, f"Missing or superfluous memory block {tensorConstraint.tensorName} allocation found in {_block}!"

                        block = _block[0]
                        memoryConstraint.addrSpace = block.addrSpace
        return ctxt

    def setupModel(self, ctxt: NetworkContext, schedule: Schedule, layerBinding: OrderedDict[str, ONNXLayer],
                   targetMemoryLevelMapping: TargetMemoryLevelMapping) -> NetworkContext:
        """Set up the constraint optimization model for tiling.

        Initializes the tiler model with geometric constraints, memory constraints,
        and optimization objectives based on the network schedule and layer bindings.

        Parameters
        ----------
        ctxt : NetworkContext
            Network context containing the computational graph.
        schedule : Schedule
            Execution schedule defining the order of operations.
        layerBinding : OrderedDict[str, ONNXLayer]
            Mapping from node names to their layer implementations.
        targetMemoryLevelMapping : TargetMemoryLevelMapping
            Mapping defining which memory levels to use for each tensor.

        Returns
        -------
        NetworkContext
            The network context (returned for consistency).

        Notes
        -----
        This method must be called before computeTilingSchedule() to initialize
        the constraint model and symbolic memory constraints.
        """

        wrapSchedule: List[SubGraph] = []
        for entry in schedule:
            if isinstance(entry, gs.Node):
                wrapSchedule.append([entry])
            else:
                wrapSchedule.append(entry)

        tilerModel = TilerModel(searchStrategy = self.searchStrategy)
        tilerModel = self._setupGeometricConstraints(tilerModel, ctxt, wrapSchedule, layerBinding)
        tilerModel = self._setupTensorDimensionProducts(tilerModel, ctxt, wrapSchedule)
        tilerModel = self._setupHeuristics(tilerModel, ctxt, wrapSchedule)
        tilerModel, allSymbolicMemoryConstraints = self._setupMemoryConstraints(tilerModel, ctxt, wrapSchedule,
                                                                                layerBinding, targetMemoryLevelMapping)

        self.tilerModel = tilerModel
        self.symbolicMemoryConstraints = allSymbolicMemoryConstraints

        return ctxt

    # SCHEREMO: Return a integer factor or IntVar variable for the multi Buffer coefficient given the tiling path, hop and tensorName.
    def multiBufferStrategy(self, tilerModel: TilerModel, ctxt: NetworkContext, pattern: SubGraph, path: List[str],
                            hop: str, tensorName: str) -> Union[int, IntVar]:
        """Determine multi-buffering coefficient for a tensor in the tiling strategy.

        Computes the buffering factor (e.g., double buffering = 2) for a given tensor
        based on its type and usage pattern in the computation graph. This coefficient
        is used to determine how many copies of the tensor should be kept in memory.

        Parameters
        ----------
        tilerModel : TilerModel, (unused)
            The constraint solver model.
        ctxt : NetworkContext
            Network context containing buffer information.
        pattern : SubGraph, (unused)
            The computation pattern being analyzed.
        path : List[str], (unused)
            Memory hierarchy path for the tensor.
        hop : str, (unused)
            Current memory level in the path.
        tensorName : str
            Name of the tensor to analyze.

        Returns
        -------
        Union[int, IntVar]
            Buffering coefficient (typically 1 for transient buffers, 2 for others).

        Notes
        -----
        The multi-buffering strategy helps overlap computation with data movement
        by maintaining multiple copies of buffers at different memory levels.
        """

        varBuffer = ctxt.lookup(tensorName)

        generalCoeff = 2

        if isinstance(varBuffer, TransientBuffer):
            coefficient = 1
        elif isinstance(varBuffer, ConstantBuffer):
            coefficient = generalCoeff
        else:
            coefficient = generalCoeff

        # if tensorName == pattern[-1].outputs[0].name:
        #     maxVal = (np.prod(varBuffer.shape) // (coefficient)).item()
        #     numElt = tilerModel.getTensorNumberOfEltVar(tensorName)
        #     constr = numElt <= maxVal

        #     if (constr != True):
        #         tilerModel.addConstraint(constr)

        return coefficient

    # SCHEREMO: Given a PatternMemoryConstraints object, propagate the IOBuffer freeing strategy.
    # Input: Single-buffered Liveness analysis of the input/output IO buffers that should be tiled
    # Output: Buffering-strategy aware liveness analysis of the input/output IO buffers

    # This version implements "static n-ple buffering"

    def propagateIOBufferStrategy(self, tileConstraintPattern: PatternMemoryConstraints, pattern: SubGraph,
                                  ctxt: NetworkContext) -> PatternMemoryConstraints:
        """Propagate I/O buffer strategy across the tiling pattern.

        Implements static n-tuple buffering strategy by propagating border tensor
        constraints across all steps in the tiling pattern.

        Parameters
        ----------
        tileConstraintPattern : PatternMemoryConstraints
            Memory constraints for the tiling pattern.
        pattern : SubGraph
            The computation subgraph being tiled.
        ctxt : NetworkContext
            Network context containing buffer information.

        Returns
        -------
        PatternMemoryConstraints
            Updated pattern memory constraints with propagated I/O buffer strategy.

        Notes
        -----
        This method ensures that border tensors (inputs/outputs of the pattern)
        maintain consistent memory allocation across all computation steps.
        """

        borderTensorStep = NodeMemoryConstraint()
        for patternStep in tileConstraintPattern.nodeConstraints:
            borderTensorStep += patternStep

        for idx in range(len(tileConstraintPattern.nodeConstraints)):
            tileConstraintPattern.nodeConstraints[idx] += borderTensorStep

        return tileConstraintPattern

    def _resolveTensorMemoryConstraint(self, tilerModel: TilerModel, ctxt: NetworkContext, collector: SolutionCollector,
                                       tensorConstraint: TensorMemoryConstraint) -> TensorMemoryConstraint:
        """Resolve symbolic tensor memory constraints to concrete values.

        Converts symbolic variables in tensor memory constraints to their
        concrete values from the solver solution.

        Parameters
        ----------
        tilerModel : TilerModel
            The constraint solver model with the solution.
        ctxt : NetworkContext
            Network context containing buffer information.
        collector : SolutionCollector
            Solution collector from the constraint solver.
        tensorConstraint : TensorMemoryConstraint
            Symbolic tensor memory constraint to resolve.

        Returns
        -------
        TensorMemoryConstraint
            Tensor memory constraint with resolved concrete values.

        Raises
        ------
        AssertionError
            If the tiler model is not initialized.

        Notes
        -----
        This method extracts the actual buffer sizes and shapes from the
        solved constraint model and creates concrete memory constraints.
        """
        assert self.tilerModel is not None, "Can't resolve tensor memory constraints, tilerModel is None!"

        tensorName = tensorConstraint.tensorName
        solvedTensorConstraint = TensorMemoryConstraint(tensorName, {}, ctxt)

        for memoryLevel, memoryConstraint in tensorConstraint.memoryConstraints.items():
            size = self.tilerModel._resolveVariable(memoryConstraint.size)

            newMemoryConstraint: MemoryConstraint = MemoryConstraint(memoryLevel, size)
            multiBufferCoefficient = self.tilerModel._resolveVariable(memoryConstraint.multiBufferCoefficient)
            newMemoryConstraint.multiBufferCoefficient = multiBufferCoefficient

            if not isinstance(ctxt.lookup(tensorName), TransientBuffer):

                tensorShapeLen = 1 if isinstance(ctxt.lookup(tensorName).shape, int) else len(
                    ctxt.lookup(tensorName).shape)
                newShape: List[int] = []

                if isinstance(memoryConstraint.size, int):
                    newShape = ctxt.lookup(tensorName).shape
                else:
                    _, copyIdx = tilerModel.getNameCopyIdx(memoryConstraint.size.Name())
                    for i in range(tensorShapeLen):
                        newShape.append(
                            self.tilerModel._resolveVariable(tilerModel.getTensorDimVar(tensorName, i, copyIdx)))

                newMemoryConstraint.shape = (newShape,) if isinstance(newShape, int) else tuple(newShape)

            solvedTensorConstraint.addMemoryConstraint(newMemoryConstraint)

        return solvedTensorConstraint

    def _getTilingSolution(self, tilerModel: TilerModel, ctxt: NetworkContext, collector: SolutionCollector,
                           allConstraints: List[PatternMemoryConstraints]) -> List[PatternMemoryConstraints]:
        """Extract tiling solution from the solved constraint model.

        Processes all pattern memory constraints and resolves symbolic variables
        to create a concrete tiling solution.

        Parameters
        ----------
        tilerModel : TilerModel
            The solved constraint model.
        ctxt : NetworkContext
            Network context containing buffer information.
        collector : SolutionCollector
            Solution collector from the constraint solver.
        allConstraints : List[PatternMemoryConstraints]
            List of all symbolic pattern memory constraints.

        Returns
        -------
        List[PatternMemoryConstraints]
            Resolved tiling solution with concrete memory constraints.

        Notes
        -----
        Only constraints that require resolution (multi-level or transient buffers)
        are processed. Global single-level buffers are skipped.
        """

        retList = []

        def _checkResolve(ctxt, tensorName, tensorConstraint):

            if ctxt.is_global(tensorName) and len(tensorConstraint.memoryConstraints.values()) <= 1:
                return False
            if len(tensorConstraint.memoryConstraints.values()) <= 1 and not isinstance(
                    ctxt.lookup(tensorName), TransientBuffer):
                return False
            return True

        for patternConstraints in allConstraints:
            newMemoryConstraint = PatternMemoryConstraints()
            for stepConstraints in patternConstraints.nodeConstraints:
                newStepMemoryConstraint = NodeMemoryConstraint()
                for tensorName, tensorConstraint in stepConstraints.tensorMemoryConstraints.items():
                    if _checkResolve(ctxt, tensorName, tensorConstraint):
                        solvedTensorConstraint = self._resolveTensorMemoryConstraint(
                            tilerModel, ctxt, collector, tensorConstraint)
                        ioDir = stepConstraints.getIO(tensorName)
                        newStepMemoryConstraint.addTensorConstraint(solvedTensorConstraint, ioDir)

                newMemoryConstraint.addConstraint(newStepMemoryConstraint)
            retList.append(newMemoryConstraint)

        return retList

    def _setupTensorDimensionProducts(self, tilerModel: TilerModel, ctxt: NetworkContext,
                                      schedule: List[SubGraph]) -> TilerModel:
        """Set up tensor dimension product variables in the tiler model.

        Adds variables representing the number of elements in each tensor
        to the constraint model for each pattern in the schedule.

        Parameters
        ----------
        tilerModel : TilerModel
            The constraint model to update.
        ctxt : NetworkContext
            Network context containing buffer information.
        schedule : List[SubGraph]
            List of computation patterns in the schedule.

        Returns
        -------
        TilerModel
            Updated tiler model with tensor dimension variables.

        Notes
        -----
        Only processes tensors that are marked for deployment in the context.
        """

        for idx, pattern in enumerate(schedule):
            subGraph = gs.Graph(nodes = pattern)
            subgraphTensors: OrderedDict[str, gs.Tensor] = subGraph.tensors(check_duplicates = True)

            for _, tensor in subgraphTensors.items():
                if not ctxt.lookup(tensor.name)._deploy:
                    continue

                tilerModel.addTensorNumOfEltToModel(ctxt, tensor.name, idx)

        return tilerModel

    def _setupGeometricConstraints(self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
                                   layerBinding: OrderedDict[str, ONNXLayer]) -> TilerModel:
        """Set up geometric constraints for each layer in the schedule.

        Adds geometric and policy constraints from each layer's tile constraint
        specification to the tiler model.

        Parameters
        ----------
        tilerModel : TilerModel
            The constraint model to update.
        ctxt : NetworkContext
            Network context containing buffer information.
        schedule : List[SubGraph]
            List of computation patterns in the schedule.
        layerBinding : OrderedDict[str, ONNXLayer]
            Mapping from node names to their layer implementations.

        Returns
        -------
        TilerModel
            Updated tiler model with geometric constraints.

        Notes
        -----
        Each pattern is treated as a decoupled sub-problem with respect to
        geometric constraints. Dimension variables are regenerated for each
        tensor using the copyIdx mechanism.
        """

        # SCHEREMO: Each pattern is a decoupled sub-problem w.r.t the geometric constraints.
        # We need to regenerate dimension variables for each tensor
        # This is done by setting the copyIdx in the tilerModel

        for idx, pattern in enumerate(schedule):
            tilerModel.copyIdx = idx

            for node in pattern:

                if node.name not in layerBinding.keys():
                    continue

                parseDict = layerBinding[node.name].mapper.parser.operatorRepresentation
                template = layerBinding[node.name].mapper.binder.template

                tilerModel = template.tileConstraint.addGeometricalConstraint(tilerModel,
                                                                              parseDict = parseDict,
                                                                              ctxt = ctxt)

                tilerModel = template.tileConstraint.addPolicyConstraint(tilerModel, parseDict = parseDict, ctxt = ctxt)

        return tilerModel

    def _setupHeuristics(self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph]) -> TilerModel:
        """Set up optimization heuristics for the tiler model.

        Adds optimization objectives to maximize memory usage efficiency
        for each pattern in the schedule.

        Parameters
        ----------
        tilerModel : TilerModel
            The constraint model to update.
        ctxt : NetworkContext
            Network context containing buffer information.
        schedule : List[SubGraph]
            List of computation patterns in the schedule.

        Returns
        -------
        TilerModel
            Updated tiler model with optimization objectives.

        Notes
        -----
        Creates pattern-level memory size variables and adds maximization
        objectives to encourage efficient memory utilization.
        """

        for idx, pattern in enumerate(schedule):

            patternTensorList = []
            seenTensorNameList = []
            for node in pattern:
                for gsTensor in node.inputs + node.outputs:
                    ctxtTensor = ctxt.lookup(gsTensor.name)
                    if ctxtTensor.name not in seenTensorNameList:
                        seenTensorNameList.append(ctxtTensor.name)
                        patternTensorList.append(ctxtTensor)

            patternMemSizeExpr: IntVar = 0
            for tensor in patternTensorList:
                if not ctxt.lookup(tensor.name)._deploy:
                    continue

                patternMemSizeExpr += tilerModel.getTensorNumberOfEltVar(
                    tensorName = tensor.name, copyIdx = idx) * (tensor._type.referencedType.typeWidth // 8)

            if isinstance(patternMemSizeExpr, int):
                _max = patternMemSizeExpr
            else:
                _max = patternMemSizeExpr.Max()

            patternVariable = tilerModel.addVariable(name = "DEEPLOY_PATTERN_MEM",
                                                     lowerBound = 1,
                                                     upperBound = _max,
                                                     copyIdx = idx)
            tilerModel.addConstraint(patternVariable == patternMemSizeExpr)

            tilerModel.addObjective(patternVariable, 'maximize')

        return tilerModel

    def _setupMemoryConstraints(
            self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
            layerBinding: OrderedDict[str, ONNXLayer],
            targetMemoryLevelMapping: TargetMemoryLevelMapping) -> Tuple[TilerModel, List[PatternMemoryConstraints]]:
        """Set up memory constraints for the tiling optimization.

        Generates memory constraints for both inner and outer memory levels,
        considering the memory hierarchy and scheduling requirements.

        Parameters
        ----------
        tilerModel : TilerModel
            The constraint model to update.
        ctxt : NetworkContext
            Network context containing buffer information.
        schedule : List[SubGraph]
            List of computation patterns in the schedule.
        layerBinding : OrderedDict[str, ONNXLayer]
            Mapping from node names to their layer implementations.
        targetMemoryLevelMapping : TargetMemoryLevelMapping
            Mapping defining which memory levels to use for each tensor.

        Returns
        -------
        Tuple[TilerModel, List[PatternMemoryConstraints]]
            Updated tiler model and list of all memory constraints.

        Notes
        -----
        Sets up both outer (inter-pattern) and inner (intra-pattern) memory
        constraints, considering the chosen memory allocation strategy.
        """

        allMemoryConstraints = self._generateAllMemoryConstraints(tilerModel, ctxt, schedule, layerBinding,
                                                                  targetMemoryLevelMapping)

        outerMemoryConstraints = PatternMemoryConstraints()
        for constraint in allMemoryConstraints:
            for nodeConstraint in constraint.nodeConstraints:
                outerMemoryConstraints.addConstraint(nodeConstraint)

        if self.memoryAllocStrategy == "MiniMalloc":
            # JUNGVI: This method adds the memory constraints in case of decoupled tiling and memory allocation.
            self.outerMemoryScheduler.constraintTileBuffersWithOverlappingLifetime(tilerModel, ctxt,
                                                                                   outerMemoryConstraints,
                                                                                   self.memoryHierarchy)

        for level in self.memoryHierarchy.memoryLevels.keys():
            self.outerMemoryScheduler.scheduleMemoryConstraints(tilerModel, ctxt, [outerMemoryConstraints],
                                                                self.memoryHierarchy, self.memoryAllocStrategy, level)

        # Update inner memoryHierarchy with outer constraints
        innerMemoryHierarchy = MemoryHierarchy([])
        for level, memLevel in self.memoryHierarchy.memoryLevels.items():
            newMemLevel = copy.copy(memLevel)

            if not self.memoryAllocStrategy == "MiniMalloc":
                outerConstraint = tilerModel.getVariable(self.outerMemoryScheduler.getSymbolicCostName(0, level), 0)
                newMemLevel.size = newMemLevel.size - outerConstraint

            innerMemoryHierarchy._add(newMemLevel)

        for level in innerMemoryHierarchy.memoryLevels.keys():
            self.innerMemoryScheduler.scheduleMemoryConstraints(tilerModel, ctxt, allMemoryConstraints,
                                                                innerMemoryHierarchy, self.memoryAllocStrategy, level)

        return tilerModel, allMemoryConstraints

    def _generateAllMemoryConstraints(
            self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
            layerBinding: OrderedDict[str, ONNXLayer],
            targetMemoryLevelMapping: TargetMemoryLevelMapping) -> List[PatternMemoryConstraints]:
        """Generate all memory constraints combining dynamic and constant tensors.

        Creates comprehensive memory constraints by combining dynamic tensor
        constraints with constant tensor constraints for each pattern.

        Parameters
        ----------
        tilerModel : TilerModel
            The constraint model.
        ctxt : NetworkContext
            Network context containing buffer information.
        schedule : List[SubGraph]
            List of computation patterns in the schedule.
        layerBinding : OrderedDict[str, ONNXLayer]
            Mapping from node names to their layer implementations.
        targetMemoryLevelMapping : TargetMemoryLevelMapping
            Mapping defining which memory levels to use for each tensor.

        Returns
        -------
        List[PatternMemoryConstraints]
            Complete list of memory constraints for all patterns.

        Notes
        -----
        Combines results from _generateMemoryConstraints to create the complete
        constraint set including both variable and constant buffers.
        """

        dynamicTensorConstraints, constantTensorConstraints = self._generateMemoryConstraints(
            tilerModel, ctxt, schedule, layerBinding, targetMemoryLevelMapping)

        allConstraints: List[PatternMemoryConstraints] = []
        # Initialize structures

        for pattern in dynamicTensorConstraints:
            allPattern = PatternMemoryConstraints()
            for step in pattern.nodeConstraints:
                allStep = step + constantTensorConstraints
                allPattern.addConstraint(allStep)
            allConstraints.append(allPattern)

        return allConstraints

    def _generateMemoryConstraints(
        self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
        layerBinding: OrderedDict[str, ONNXLayer], targetMemoryLevelMapping: TargetMemoryLevelMapping
    ) -> Tuple[List[PatternMemoryConstraints], NodeMemoryConstraint]:
        """Generate memory constraints for variable and constant buffers.

        Creates detailed memory constraints including outer/inner variable
        buffer constraints, tiled tensor constraints, and constant buffer
        constraints.

        Parameters
        ----------
        tilerModel : TilerModel
            The constraint model.
        ctxt : NetworkContext
            Network context containing buffer information.
        schedule : List[SubGraph]
            List of computation patterns in the schedule.
        layerBinding : OrderedDict[str, ONNXLayer]
            Mapping from node names to their layer implementations.
        targetMemoryLevelMapping : TargetMemoryLevelMapping
            Mapping defining which memory levels to use for each tensor.

        Returns
        -------
        Tuple[List[PatternMemoryConstraints], NodeMemoryConstraint]
            Tuple containing:
            - List of pattern memory constraints for dynamic tensors
            - Node memory constraint for constant buffers

        Notes
        -----
        Generates three levels of constraints:
        1. First-level: global buffers + higher-level tensors
        2. Tiled tensor constraints with double buffering
        3. In-place tensor constraints for unkilled tensors
        """

        # SCHEREMO: Construct non-double-buffered constraints of local variable buffers

        outerVariableConstraints, innerVariableConstraints = self._generateVariableBufferConstraints(
            tilerModel, ctxt, schedule, layerBinding, targetMemoryLevelMapping)

        # SCHEREMO: Construct global buffer constraints

        constantBufferConstraint = self._generateBufferConstraints(ctxt)

        # SCHEREMO: Construct first-level constraint set (all global buffers + tensors stored in higher level)

        firstLevelConstraints: List[PatternMemoryConstraints] = copy.copy(outerVariableConstraints)
        for patternConstraint in firstLevelConstraints:
            for idx in range(len(patternConstraint.nodeConstraints)):
                patternConstraint.nodeConstraints[idx] += constantBufferConstraint

        # SCHEREMO: Construct constraint set for tiled tensors (including double buffering, excluding static global constraints)
        tiledTensorConstraints: List[PatternMemoryConstraints] = self._generateTilePathConstraints(
            tilerModel, ctxt, firstLevelConstraints, innerVariableConstraints, schedule)

        # SCHEREMO: Construct constraint set for tiled tensors + local-only tensors (dynamic tensor set)
        dynamicTensorConstraints: List[PatternMemoryConstraints] = []
        for tilingConstraints, innerConstraints in zip(tiledTensorConstraints, innerVariableConstraints):

            dynamicTensorPattern = PatternMemoryConstraints()
            for tilingPatternStep, innerPatternStep in zip(tilingConstraints.nodeConstraints,
                                                           innerConstraints.nodeConstraints):
                dynamicTensorPatternStep = copy.copy(tilingPatternStep)

                # Pick all constraints that are purely internal
                for innerTensorName, innerTensor in innerPatternStep.tensorMemoryConstraints.items():
                    if not any([
                            innerTensorName == dynamicTensorName
                            for dynamicTensorName, tensor in dynamicTensorPatternStep.tensorMemoryConstraints.items()
                    ]):
                        ioDir = tilingPatternStep.getIO(innerTensorName)
                        dynamicTensorPatternStep.addTensorConstraint(innerTensor, ioDir)
                dynamicTensorPattern.addConstraint(dynamicTensorPatternStep)
            dynamicTensorConstraints.append(dynamicTensorPattern)

        # SCHEREMO: Construct unkilled tensor set
        inplaceTensorConstraints: List[PatternMemoryConstraints] = []
        for tilingConstraints, outerConstraints in zip(dynamicTensorConstraints, firstLevelConstraints):
            dynamicTensorPattern = PatternMemoryConstraints()
            for tilingPatternStep, outerPatternStep in zip(tilingConstraints.nodeConstraints,
                                                           outerConstraints.nodeConstraints):
                dynamicTensorPatternStep = copy.copy(tilingPatternStep)

                # Pick all constraints that are purely internal
                for outerTensorName, outerTensor in outerPatternStep.tensorMemoryConstraints.items():
                    if not any(
                        [(outerTensorName == dynamicTensorName) or (ctxt.is_global(outerTensorName))
                         for dynamicTensorName, tensor in dynamicTensorPatternStep.tensorMemoryConstraints.items()]):
                        dynamicTensorPatternStep.addTensorConstraint(outerTensor, "intermediate")
                dynamicTensorPattern.addConstraint(dynamicTensorPatternStep)
            inplaceTensorConstraints.append(dynamicTensorPattern)

        return inplaceTensorConstraints, constantBufferConstraint

    def _generateTilePath(self, tilerModel: TilerModel, ctxt: NetworkContext,
                          tensorMemoryConstraint: TensorMemoryConstraint, pattern: SubGraph) -> TensorMemoryConstraint:
        """Generate tiling path for a tensor across memory hierarchy levels.

        Creates memory constraints for a tensor that needs to move between
        different levels of the memory hierarchy, including multi-buffering.

        Parameters
        ----------
        tilerModel : TilerModel
            The constraint model.
        ctxt : NetworkContext
            Network context containing buffer information.
        tensorMemoryConstraint : TensorMemoryConstraint
            Original tensor memory constraint with multiple levels.
        pattern : SubGraph
            The computation pattern using this tensor.

        Returns
        -------
        TensorMemoryConstraint
            Updated tensor memory constraint with complete tiling path.

        Raises
        ------
        AssertionError
            If the tensor constraint doesn't have exactly 2 memory levels,
            or if the multi-buffer factor is invalid.

        Notes
        -----
        Uses breadth-first search to find the path between memory levels
        and applies multi-buffering strategy at each intermediate level.
        """

        assert len(tensorMemoryConstraint.memoryConstraints.keys()
                  ) == 2, "Can't generate a tile path for more than one hierarchy level!"

        tensorName = tensorMemoryConstraint.tensorName

        valList = list(tensorMemoryConstraint.memoryConstraints.values())
        constraintA = valList[0]
        constraintB = valList[1]

        # SCHEREMO : Base is whichever constraint is constant
        base = constraintA if isinstance(constraintA.size, int) else constraintB
        end = constraintA if base == constraintB else constraintB

        path = self.memoryHierarchy.bfs(base.memoryLevel, end.memoryLevel)
        requiredHops = path[1:]

        returnTensorConstraint = TensorMemoryConstraint(tensorName, {}, ctxt)
        returnTensorConstraint.addMemoryConstraint(base)

        for hop in requiredHops:
            factor = self.multiBufferStrategy(tilerModel, ctxt, pattern, path, hop, tensorName)
            assert factor >= 1 and isinstance(factor, int), "Invalid factor!"

            memConstraint = MemoryConstraint(hop, end.size)
            memConstraint.multiBufferCoefficient = factor
            returnTensorConstraint.addMemoryConstraint(memConstraint)

        return returnTensorConstraint

    def _generateIntermediateTilingSteps(self, tilerModel: TilerModel, ctxt: NetworkContext,
                                         sourceStep: NodeMemoryConstraint, destinationStep: NodeMemoryConstraint,
                                         pattern: SubGraph) -> NodeMemoryConstraint:
        """Generate intermediate tiling steps between source and destination constraints.

        Creates tiling constraints for tensors that need to move between different
        memory levels within a computation pattern.

        Parameters
        ----------
        tilerModel : TilerModel
            The constraint model.
        ctxt : NetworkContext
            Network context containing buffer information.
        sourceStep : NodeMemoryConstraint
            Memory constraints for the source step.
        destinationStep : NodeMemoryConstraint
            Memory constraints for the destination step.
        pattern : SubGraph
            The computation pattern being analyzed.

        Returns
        -------
        NodeMemoryConstraint
            Memory constraints for intermediate tiling steps.

        Notes
        -----
        Identifies tensors that require tiling (those with multiple memory
        constraints) and generates appropriate tiling paths for them.
        """
        tileConstraintStep = NodeMemoryConstraint()

        mergedStep = sourceStep + destinationStep
        tileTensorConstraints = [
            tensor for tensor in mergedStep.tensorMemoryConstraints.values()
            if len(tensor.memoryConstraints.values()) > 1
        ]

        for tileTensor in tileTensorConstraints:
            tiledTensor = self._generateTilePath(tilerModel, ctxt, tileTensor, pattern)
            ioDir = mergedStep.getIO(tileTensor.tensorName)
            tileConstraintStep.addTensorConstraint(tiledTensor, ioDir)

        return tileConstraintStep

    def _generateTilePathConstraints(self, tilerModel: TilerModel, ctxt: NetworkContext,
                                     sourceConstraints: List[PatternMemoryConstraints],
                                     destinationConstraints: List[PatternMemoryConstraints],
                                     schedule: List[SubGraph]) -> List[PatternMemoryConstraints]:
        """Generate tiling path constraints for all patterns in the schedule.

        Creates comprehensive tiling constraints by combining source and destination
        constraints for each pattern and applying I/O buffer strategies.

        Parameters
        ----------
        tilerModel : TilerModel
            The constraint model.
        ctxt : NetworkContext
            Network context containing buffer information.
        sourceConstraints : List[PatternMemoryConstraints]
            Source memory constraints for each pattern.
        destinationConstraints : List[PatternMemoryConstraints]
            Destination memory constraints for each pattern.
        schedule : List[SubGraph]
            List of computation patterns in the schedule.

        Returns
        -------
        List[PatternMemoryConstraints]
            Complete tiling path constraints for all patterns.

        Raises
        ------
        AssertionError
            If source pattern constraints are not single-step.

        Notes
        -----
        Assumes source patterns are constant and single-step since they
        represent tensors that are live throughout the pattern execution.
        """

        tileConstraints = []

        for idx, (sourceConstraint, destinationConstraint) in enumerate(zip(sourceConstraints, destinationConstraints)):

            tilerModel.copyIdx = idx

            assert (len(sourceConstraint.nodeConstraints) == 1
                   ), "source pattern must be constant and single step, since it's live throughout the pattern!"
            sourcePatternStep = sourceConstraint.nodeConstraints[0]

            tileConstraint = PatternMemoryConstraints()

            for destinationConstraintStep in destinationConstraint.nodeConstraints:
                tileConstraintStep = self._generateIntermediateTilingSteps(tilerModel, ctxt, sourcePatternStep,
                                                                           destinationConstraintStep, schedule[idx])
                tileConstraint.addConstraint(tileConstraintStep)

            propagatedTileConstraint = self.propagateIOBufferStrategy(tileConstraint, schedule[idx], ctxt)
            assert len(propagatedTileConstraint.nodeConstraints) == len(tileConstraint.nodeConstraints)

            tileConstraints.append(propagatedTileConstraint)

        return tileConstraints

    def _generateBufferConstraints(self, ctxt: NetworkContext) -> NodeMemoryConstraint:
        """Generate memory constraints for constant global buffers.

        Creates memory constraints for all constant buffers that are marked
        for deployment in the network context.

        Parameters
        ----------
        ctxt : NetworkContext
            Network context containing buffer information.

        Returns
        -------
        NodeMemoryConstraint
            Memory constraints for all constant global buffers.

        Notes
        -----
        Only processes constant buffers with _deploy flag set to True.
        Each buffer is treated as an input tensor in the constraints.
        """

        constantGlobalConstraint: NodeMemoryConstraint = NodeMemoryConstraint()
        constantGlobalBuffers = [
            node for node in ctxt.globalObjects.values() if isinstance(node, ConstantBuffer) and node._deploy == True
        ]

        for constantBuffer in constantGlobalBuffers:

            tensorName = constantBuffer.name

            memorySize = int(np.prod(ctxt.lookup(tensorName).shape))

            elementMemorySize = memorySize
            memoryConstraint = MemoryConstraint(constantBuffer._memoryLevel, elementMemorySize)
            tensorConstraint = TensorMemoryConstraint(constantBuffer.name,
                                                      {memoryConstraint.memoryLevel: memoryConstraint}, ctxt)
            constantGlobalConstraint.addTensorConstraint(tensorConstraint, "input")

        return constantGlobalConstraint

    def _generateVariableBufferConstraints(
        self, tilerModel: TilerModel, ctxt: NetworkContext, schedule: List[SubGraph],
        layerBinding: OrderedDict[str, ONNXLayer], targetMemoryLevelMapping: TargetMemoryLevelMapping
    ) -> Tuple[List[PatternMemoryConstraints], List[PatternMemoryConstraints]]:
        """Generate memory constraints for variable buffers using flow analysis.

        Performs liveness analysis on the computation graph to determine
        memory requirements for variable buffers at different points in execution.

        Parameters
        ----------
        tilerModel : TilerModel
            The constraint model.
        ctxt : NetworkContext
            Network context containing buffer information.
        schedule : List[SubGraph]
            List of computation patterns in the schedule.
        layerBinding : OrderedDict[str, ONNXLayer]
            Mapping from node names to their layer implementations.
        targetMemoryLevelMapping : TargetMemoryLevelMapping
            Mapping defining which memory levels to use for each tensor.

        Returns
        -------
        Tuple[List[PatternMemoryConstraints], List[PatternMemoryConstraints]]
            Tuple containing:
            - Outer memory constraints (inter-pattern)
            - Inner memory constraints (intra-pattern)

        Notes
        -----
        Uses graph flow analysis to compute liveness information and generates
        both outer (pattern-level) and inner (step-level) memory constraints.
        Includes transient buffer constraints for each computation step.
        """

        def deltaFlow(
                patternFlow: List[GenericFlowState[TensorMemLevelTuple]]) -> GenericFlowState[TensorMemLevelTuple]:

            initialFlow = patternFlow[0]
            endFlow = patternFlow[1]

            # SCHEREMO: The genset and killset of the innerflow are correct; however, since we now pass the initialliveset of the pattern to the constraint flow. we need to remove bypassed tensors
            mergedLiveSet = initialFlow.liveSet - endFlow.liveSet
            mergedGenSet = initialFlow.genSet
            mergedKillSet = initialFlow.killSet

            mergedFlow = GenericFlowState[TensorMemLevelTuple](mergedLiveSet, mergedKillSet, mergedGenSet)

            return mergedFlow

        initialLiveBuffers = {
            value.name
            for value in ctxt.globalObjects.values()
            if (isinstance(value, ctxt.VariableBuffer) and value._users != [])
        }

        producedBuffers = {layer.node.outputs[0].name for layer in layerBinding.values()}
        inputBufferNames = initialLiveBuffers - producedBuffers
        inputBuffers = [ctxt.lookup(name) for name in inputBufferNames]

        initialLiveTensors = {TensorMemLevelTuple(buf.name, buf._memoryLevel) for buf in inputBuffers}

        constraintFlow = GraphMemoryConstraintFlow(ctxt, targetMemoryLevelMapping)
        graphFlowStates = constraintFlow.flow(schedule, initialLiveTensors)

        innerMemConstraints: List[PatternMemoryConstraints] = []
        outerMemConstraints: List[PatternMemoryConstraints] = []

        for idx, pattern in enumerate(schedule):

            tilerModel.copyIdx = idx

            innerPatternMemoryConstraints = PatternMemoryConstraints()
            outerPatternMemoryConstraints = PatternMemoryConstraints()

            outerFlowState = graphFlowStates[idx]
            patternFlow = constraintFlow._patternFlowStates[idx]

            dynamicOuterBufferConstraints = convertFlowState2NodeMemoryConstraint(tilerModel,
                                                                                  ctxt,
                                                                                  outerFlowState,
                                                                                  useMax = True)

            outerPatternMemoryConstraints.addConstraint(dynamicOuterBufferConstraints)
            outerMemConstraints.append(outerPatternMemoryConstraints)

            mergedFlow = [deltaFlow(patternFlow)]

            for step, innerFlowState in zip(pattern, mergedFlow):
                transientBufferConstraints = self._generatePatternStepTransientBufferConstraints(
                    tilerModel, ctxt, layerBinding, step, targetMemoryLevelMapping)

                dynamicInnerBufferConstraints = convertFlowState2NodeMemoryConstraint(tilerModel,
                                                                                      ctxt,
                                                                                      innerFlowState,
                                                                                      useMax = False)

                innerPatternMemoryConstraints.addConstraint(transientBufferConstraints + dynamicInnerBufferConstraints)

            innerMemConstraints.append(innerPatternMemoryConstraints)

        return outerMemConstraints, innerMemConstraints

    def _generatePatternStepTransientBufferConstraints(
            self, tilerModel: TilerModel, ctxt: NetworkContext, layerBinding: OrderedDict[str, ONNXLayer],
            step: gs.Node, targetMemoryLevelMapping: TargetMemoryLevelMapping) -> NodeMemoryConstraint:
        """Generate memory constraints for transient buffers in a pattern step.

        Computes memory requirements for temporary buffers needed during
        the execution of a single computation step.

        Parameters
        ----------
        tilerModel : TilerModel
            The constraint model.
        ctxt : NetworkContext
            Network context containing buffer information.
        layerBinding : OrderedDict[str, ONNXLayer]
            Mapping from node names to their layer implementations.
        step : gs.Node
            The computation node being analyzed.
        targetMemoryLevelMapping : TargetMemoryLevelMapping
            Mapping defining which memory levels to use for each tensor.

        Returns
        -------
        NodeMemoryConstraint
            Memory constraints for transient buffers in this step.

        Notes
        -----
        Transient buffers are assumed to be allocated in the same memory
        level as the main input of the computation step. Buffer sizes are
        computed using the layer template's transient buffer size calculation.
        """

        patternStepTransientBufferSizes = NodeMemoryConstraint()

        template = layerBinding[step.name].mapper.binder.template

        symbolicNodeRep = template.tileConstraint.constructSymbolicNodeRep(
            tilerModel, parseDict = layerBinding[step.name].mapper.parser.operatorRepresentation, ctxt = ctxt)

        transientBufferList: List[Tuple[str,
                                        Union[int,
                                              IntVar]]] = template.computeTransientBuffersSize(ctxt, symbolicNodeRep)

        for tensorName, memorySize in transientBufferList:

            # SCHEREMO: Assume transientbuffers end up in the same level as their user's main input
            memoryLevelName = targetMemoryLevelMapping.lookup(step.name, step.inputs[0].name)
            ctxt.lookup(tensorName)._memoryLevel = memoryLevelName

            transientSize = tilerModel.addTransientBufferSizeToModel(tensorName, memorySize)

            #memoryLevelName = self.memoryHierarchy.getDefaultMemoryLevel().name

            transientMemoryConstraint = MemoryConstraint(memoryLevelName, transientSize)
            transientBufferConstraint = TensorMemoryConstraint(tensorName, {memoryLevelName: transientMemoryConstraint},
                                                               ctxt)
            patternStepTransientBufferSizes.addTensorConstraint(transientBufferConstraint, "intermediate")

        return patternStepTransientBufferSizes

    def assertLayerWiseTiling(self, schedule: List[List[gs.Node]]) -> bool:
        """Assert that the schedule uses layer-wise tiling (one node per pattern).

        Verifies that each pattern in the schedule contains exactly one node,
        which is required for certain memory allocation strategies.

        Parameters
        ----------
        schedule : List[List[gs.Node]]
            The execution schedule to validate.

        Returns
        -------
        bool
            True if all patterns contain exactly one node, False otherwise.

        Notes
        -----
        Layer-wise tiling is required when using the MiniMalloc memory
        allocation strategy.
        """
        for pattern in schedule:
            if len(pattern) > 1:
                return False

        return True

    def assertUniformMemoryLevelAllocation(self, ctxt: NetworkContext, defaultMemoryLevel: str) -> bool:
        """Assert that all local buffers are allocated to the default memory level.

        Verifies that all local buffers in the network context are assigned
        to the specified default memory level.

        Parameters
        ----------
        ctxt : NetworkContext
            Network context containing buffer information.
        defaultMemoryLevel : str
            Name of the default memory level to check against.

        Returns
        -------
        bool
            True if all local buffers use the default memory level, False otherwise.

        Notes
        -----
        Uniform memory level allocation is required when using the MiniMalloc
        memory allocation strategy.
        """
        for buffer in ctxt.localObjects.values():
            if buffer._memoryLevel != defaultMemoryLevel:
                return False
        return True

    def testTilingSolutionCorrectness(self, tilingSolution: TilingSolution) -> None:
        """Test the correctness of a computed tiling solution.

        Validates that buffer sizes in the tiling solution are properly
        aligned according to memory alignment requirements.

        Parameters
        ----------
        tilingSolution : TilingSolution
            The tiling solution to validate.

        Raises
        ------
        AssertionError
            If any buffer is not properly aligned or if multi-buffer
            coefficients are not integers.

        Notes
        -----
        Checks that all allocated buffers meet the byte alignment requirements
        specified in MemoryScheduler.byteAlignment.
        """
        # LMACAN: Assert buffer sizes are word aligned as per comment in MemoryScheduler.py:MemoryScheduler._buildCostVector()
        byteAlignment = MemoryScheduler.byteAlignment
        for patternMemoryConstraint in tilingSolution:
            for nodeMemoryConstraint in patternMemoryConstraint.nodeConstraints:
                for tensorMemoryConstraint in nodeMemoryConstraint.tensorMemoryConstraints.values():
                    for memoryConstraint in tensorMemoryConstraint.memoryConstraints.values():
                        if memoryConstraint.addrSpace is not None:
                            assert isinstance(memoryConstraint.multiBufferCoefficient, int)
                            bufferSize = (memoryConstraint.addrSpace[1] -
                                          memoryConstraint.addrSpace[0]) // memoryConstraint.multiBufferCoefficient
                            assert bufferSize % byteAlignment == 0, f"Buffer in {memoryConstraint} is not {byteAlignment} byte aligned"

    def testMemoryMapCorrectness(self, memoryMap: Dict[str, List[List[MemoryBlock]]], graph: gs.Graph,
                                 schedule: Schedule) -> None:
        """Test the correctness of a computed memory map.

        Validates that the memory map correctly represents buffer lifetimes
        and ensures all required buffers are alive when needed.

        Parameters
        ----------
        memoryMap : Dict[str, List[List[MemoryBlock]]]
            The memory map to validate.
        graph : gs.Graph
            The computation graph.
        schedule : Schedule
            The execution schedule.

        Raises
        ------
        AssertionError
            If output buffers are not alive until the end, input buffers
            are not alive at the beginning, or required buffers are not
            alive during computation steps.

        Notes
        -----
        Performs comprehensive validation of buffer lifetimes to ensure
        the memory map is consistent with the computation requirements.
        """

        memoryBlockMap = {
            memoryBlock.name: memoryBlock for levelMemoryMap in memoryMap.values() for memoryBlock in levelMemoryMap[-1]
        }

        # JUNGVI: Assert output buffers are alive until the end
        for tensor in graph.outputs:
            assert memoryBlockMap[tensor.name]._lifetime[-1] == len(
                schedule), "Invalid memory map! Output buffer is not alive at the last step!"

        # JUNGVI: Assert input buffers are alive at the beginning
        for inputBuffer in graph.inputs:
            assert memoryBlockMap[
                inputBuffer.name]._lifetime[0] == 0, "Invalid memory map! Input buffer is not alive at step 0!"

        # JUNGVI: Assert that at every computation step, the required buffers are alive somewhere in memory
        for stepIdx, pattern in enumerate(schedule):
            node = pattern[0]
            nodeIO = [node for node in node.inputs + node.outputs if not isinstance(node, gs.Constant)]
            for tensor in nodeIO:
                lifetime = memoryBlockMap[tensor.name]._lifetime
                assert stepIdx in range(lifetime[0], lifetime[-1] +
                                        1), f"Invalid memory map! Buffer {tensor.name} is not alive at step {stepIdx}!"


class TilerDeployerWrapper(NetworkDeployerWrapper):
    """Wrapper for network deployers that adds tiling capabilities.

    Extends NetworkDeployerWrapper to provide automatic tiling and memory
    management for neural network deployment on memory-constrained hardware.

    Parameters
    ----------
    deployer : Union[MemoryLevelAwareDeployer, MemoryDeployerWrapper]
        The base deployer to wrap with tiling capabilities.
    tilerCls : Type[Tiler], optional
        The tiler class to use, by default Tiler.

    Attributes
    ----------
    tiler : Tiler
        The tiler instance used for memory optimization.

    Raises
    ------
    AssertionError
        If the platform is not a MemoryPlatform or MemoryPlatformWrapper.

    Notes
    -----
    The wrapper automatically handles tiling setup, constraint solving,
    and memory allocation during the binding process.
    """

    def __init__(self,
                 deployer: Union[MemoryLevelAwareDeployer, MemoryDeployerWrapper],
                 tilerCls: Type[Tiler] = Tiler,
                 testName: Optional[str] = None,
                 workDir: Optional[str] = None):
        """Initialize the tiler deployer wrapper.

        Parameters
        ----------
        deployer : Union[MemoryLevelAwareDeployer, MemoryDeployerWrapper]
            The base deployer to wrap.
        tilerCls : Type[Tiler], optional
            The tiler class to instantiate, by default Tiler.
        testName : Optional[str], optional
            Optional name for the test case, used for file naming. Defaults to None.
        workDir : Optional[str], optional
            Optional working directory for temporary files. Defaults to None.
        """
        super().__init__(deployer)
        assert isinstance(self.Platform, (MemoryPlatform, MemoryPlatformWrapper)), \
            f"Platform should be a MemoryPlatform or MemoryPlatformWrapper! Got {type(self.Platform).__name__}"
        self.tiler = tilerCls(self.Platform.memoryHierarchy, testName = testName, workDir = workDir)

    @property
    def worstCaseBufferSize(self):
        """Get the worst-case buffer sizes including inputs and outputs.

        Computes the total worst-case memory requirements including
        both tiled buffers and input/output buffers.

        Returns
        -------
        Dict[str, int]
            Dictionary mapping memory level names to their total worst-case
            buffer sizes in bytes.

        Notes
        -----
        Extends the tiler's worst-case buffer size calculation by adding
        the memory requirements of input and output buffers.
        """
        return self.tiler.worstCaseBufferSize

    def tile(self, tilingSolution: Optional[TilingSolution] = None, memoryMap: Optional[MemoryMap] = None):
        """Perform tiling and memory allocation for the network.

        Executes the complete tiling process including constraint setup,
        optimization, memory allocation, and code generation updates.

        Parameters
        ----------
        tilingSolution : Optional[TilingSolution], optional
            Pre-computed tiling solution to use instead of computing one.
            If None, the solution will be computed automatically.
        memoryMap : Optional[MemoryMap], optional
            Pre-computed memory map to use instead of computing one.
            If None, the memory map will be computed automatically.

        Raises
        ------
        AssertionError
            If only one of tilingSolution or memoryMap is provided,
            if MiniMalloc is used with non-layer-wise tiling,
            or if tensors are not uniformly allocated when using MiniMalloc.

        Notes
        -----
        When using MiniMalloc memory allocation strategy, additional
        constraints apply:
        - Only layer-wise execution is supported
        - All tensors must be in the default memory level

        The method performs validation of the computed solutions and
        updates the execution blocks with tiling information.
        """
        assert (tilingSolution is None and memoryMap is None) or (tilingSolution is not None and memoryMap is not None), \
            "You need to provide both the manual tilingSolution and the memoryMap to override tiling."

        schedule = self.scheduler(self.graph)

        if tilingSolution is None and memoryMap is None:
            # JUNGVI: Currently using MiniMalloc is only supported for layer-wise execution and all tensors in the default memory level.
            if self.tiler.memoryAllocStrategy == "MiniMalloc":
                assert self.tiler.assertLayerWiseTiling(schedule), "Using MiniMalloc and DFT is not supported!"
                assert self.tiler.assertUniformMemoryLevelAllocation(
                    self.ctxt, self.Platform.memoryHierarchy._defaultMemoryLevel.name
                ), "All tensors have to be in the default memory level when using MiniMalloc!"

            log.debug(" - Setup Constraint Model")
            self.tiler.setupModel(ctxt = self.ctxt,
                                  schedule = schedule,
                                  layerBinding = self.layerBinding,
                                  targetMemoryLevelMapping = self.getTargetMemoryLevelMapping())
            tilingSolution = self.tiler.computeTilingSchedule(self.ctxt)

            memoryMap = self.tiler.computeMemoryMap(self.ctxt, tilingSolution)

        assert tilingSolution is not None and memoryMap is not None

        log.debug(" - Test Tiling Solution Correctness")
        self.tiler.testTilingSolutionCorrectness(tilingSolution)

        log.debug(" - Annotate Memory Levels")
        self.tiler.annotateMemoryLevel(self.ctxt, tilingSolution, memoryMap)

        self.ctxt = self.tiler._convertCtxtToStaticSchedule(self.ctxt, memoryMap)

        if self.tiler.visualizeMemoryAlloc:
            log.info(f" > Export Memory Allocation Visualization to {self.deeployStateDir}")
            self.tiler.plotMemoryAlloc(memoryMap, self.ctxt, self.deeployStateDir, self.Platform.memoryHierarchy)

        log.debug(" - Test Memory Map Correctness")
        self.tiler.testMemoryMapCorrectness(memoryMap, self.graph, schedule)

        # SCHEREMO: Annotate execution block with solution
        for layer, pattern in zip(self.layerBinding.values(), tilingSolution):
            layer.mapper.binder.executionBlock.patternMemoryConstraint = pattern

        # SCHEREMO: Code generation STUB

    def bind(self):
        """Bind the network with automatic tiling.

        Performs the complete binding process including layer binding
        and automatic tiling optimization.

        Returns
        -------
        bool
            True if binding was successful, False otherwise.

        Notes
        -----
        Calls the parent bind() method first, then performs tiling
        if the initial binding was successful.
        """
        if not super().bind():
            return False

        log.info("- Performing Tiling and Memory Allocation")
        self.tile()
        return True

    def _printMemorySummary(self):
        log.info("")
        log.info("Memory Usage Report:")
        log.info(f"  {'Level':<14} {'Capacity (bytes)':>10} {'Total':>10} (    Static + Dynamic   ) (Usage )")
        log.info("  " + "-" * 78)

        for level, dynamicSize in self.worstCaseBufferSize.items():
            staticSize = self.tiler.outerMemoryScheduler.getConstantTensorOffset(self.ctxt, level)
            capacity = self.tiler.memoryHierarchy.memoryLevels[level].size
            total = staticSize + dynamicSize

            log.info(f"  {level:<20} {capacity:10,} {total:10,d} "
                     f"({staticSize:10,d} + {dynamicSize:10,d}) "
                     f"({total / capacity * 100:5.1f}%)")


def TilingReadyNodeBindings(nodeBindings: List[NodeBinding], tileConstraint: TileConstraint) -> List[NodeBinding]:
    """Apply tiling constraints to a list of node bindings.

    Creates deep copies of the provided node bindings and attaches the
    specified tile constraint to each binding's template.

    Parameters
    ----------
    nodeBindings : List[NodeBinding]
        List of node bindings to make tiling-ready.
    tileConstraint : TileConstraint
        The tile constraint to attach to each binding.

    Returns
    -------
    List[NodeBinding]
        List of node bindings with tiling constraints attached.

    Notes
    -----
    The function creates deep copies to avoid modifying the original
    node bindings. Each template in the copied bindings gets the
    tileConstraint attribute set.

    Examples
    --------
    >>> bindings = [binding1, binding2, binding3]
    >>> constraint = MyTileConstraint()
    >>> tiling_bindings = TilingReadyNodeBindings(bindings, constraint)
    """
    nodeBindingsCopy = copy.deepcopy(nodeBindings)  #.copy()
    for binding in nodeBindingsCopy:
        binding.template.tileConstraint = tileConstraint

    return nodeBindingsCopy
