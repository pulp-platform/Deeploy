# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Optional, Tuple, Type, Union

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.DeeployTypes import DeploymentPlatform, NetworkDeployer, TopologyOptimizer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryPlatform, MemoryPlatformWrapper
from Deeploy.Targets.Chimera.Deployer import ChimeraDeployer
from Deeploy.Targets.Chimera.Platform import ChimeraOptimizer, ChimeraPlatform
from Deeploy.Targets.CortexM.Deployer import CMSISDeployer
from Deeploy.Targets.CortexM.Platform import CMSISOptimizer, CMSISPlatform
from Deeploy.Targets.GAP9.Deployer import GAP9Deployer
from Deeploy.Targets.GAP9.Platform import GAP9Platform, MemoryGAP9Platform, MemoryGAP9PlatformWrapper
from Deeploy.Targets.Generic.Deployer import GenericDeployer
from Deeploy.Targets.Generic.Platform import GenericOptimizer, GenericPlatform
from Deeploy.Targets.MemPool.Deployer import MemPoolDeployer
from Deeploy.Targets.MemPool.Platform import MemPoolOptimizer, MemPoolPlatform
from Deeploy.Targets.Neureka.Deployer import NeurekaDeployer
from Deeploy.Targets.Neureka.Platform import MemoryNeurekaPlatform, MemoryNeurekaPlatformWrapper, NeurekaOptimizer, \
    NeurekaPlatform
from Deeploy.Targets.PULPOpen.Deployer import PULPDeployer
from Deeploy.Targets.PULPOpen.Platform import MemoryPULPPlatform, MemoryPULPPlatformWrapper, PULPOptimizer, PULPPlatform
from Deeploy.Targets.Snitch.Deployer import SnitchDeployer
from Deeploy.Targets.Snitch.Platform import SnitchOptimizer, SnitchPlatform
from Deeploy.Targets.SoftHier.Deployer import SoftHierDeployer
from Deeploy.Targets.SoftHier.Platform import SoftHierOptimizer, SoftHierPlatform

_SIGNPROP_PLATFORMS = ["Apollo3", "Apollo4", "QEMU-ARM", "Generic", "MemPool", "SoftHier"]
_NONSIGNPROP_PLATFORMS = ["Siracusa", "Siracusa_w_neureka", "PULPOpen", "Snitch", "Chimera", "GAP9"]
_PLATFORMS = _SIGNPROP_PLATFORMS + _NONSIGNPROP_PLATFORMS


def defaultScheduler(graph: gs.Graph):
    return graph.nodes


def mapPlatform(platformName: str) -> Tuple[DeploymentPlatform, bool]:

    assert platformName in _PLATFORMS,\
        "Platform's signprop preference is unknown! Add it in platformMapping.py."

    if platformName in _SIGNPROP_PLATFORMS:
        signProp = True
    else:
        signProp = False

    if platformName == "Apollo3" or platformName == "Apollo4" or platformName == "QEMU-ARM":
        Platform = CMSISPlatform()

    elif platformName == "MemPool":
        Platform = MemPoolPlatform()

    elif platformName == "Generic":
        Platform = GenericPlatform()

    elif platformName == "Siracusa" or platformName == "PULPOpen":
        Platform = PULPPlatform()

    elif platformName == "GAP9":
        Platform = GAP9Platform()

    elif platformName == "Siracusa_w_neureka":
        Platform = NeurekaPlatform()

    elif platformName == "Snitch":
        Platform = SnitchPlatform()

    elif platformName == "SoftHier":
        Platform = SoftHierPlatform()

    elif platformName == "Chimera":
        Platform = ChimeraPlatform()

    else:
        raise RuntimeError(f"Deployment platform {platformName} is not implemented")

    return Platform, signProp


def setupMemoryPlatform(platform: DeploymentPlatform, memoryHierarchy: MemoryHierarchy,
                        defaultTargetMemoryLevel: MemoryLevel) -> Union[MemoryPlatform, MemoryPlatformWrapper]:
    if isinstance(platform, PULPPlatform):
        return MemoryPULPPlatformWrapper(platform, memoryHierarchy, defaultTargetMemoryLevel)
    elif isinstance(platform, NeurekaPlatform):
        weightMemoryLevel = memoryHierarchy.memoryLevels["WeightMemory_SRAM"] \
            if "WeightMemory_SRAM" in memoryHierarchy.memoryLevels else None
        return MemoryNeurekaPlatformWrapper(platform, memoryHierarchy, defaultTargetMemoryLevel, weightMemoryLevel)
    if isinstance(platform, GAP9Platform):
        return MemoryGAP9PlatformWrapper(platform, memoryHierarchy, defaultTargetMemoryLevel)
    else:
        return MemoryPlatformWrapper(platform, memoryHierarchy, defaultTargetMemoryLevel)


def mapDeployer(platform: DeploymentPlatform,
                graph: gs.Graph,
                inputTypes: Dict[str, Type[Pointer]],
                loweringOptimizer: Optional[TopologyOptimizer] = None,
                scheduler: Optional[Callable] = None,
                name: Optional[str] = None,
                default_channels_first: Optional[bool] = None,
                deeployStateDir: Optional[str] = None,
                inputOffsets: Optional[Dict[str, int]] = None) -> NetworkDeployer:

    if scheduler is None:
        scheduler = defaultScheduler

    if deeployStateDir is None:
        deeployStateDir = "deeployStates"

    if name is None:
        name = "DeeployNetwork"

    if isinstance(platform, CMSISPlatform):

        if loweringOptimizer is None:
            loweringOptimizer = CMSISOptimizer

        if default_channels_first is None:
            default_channels_first = False

        deployer = CMSISDeployer(graph,
                                 platform,
                                 inputTypes,
                                 loweringOptimizer,
                                 scheduler,
                                 name = name,
                                 default_channels_first = default_channels_first,
                                 deeployStateDir = deeployStateDir,
                                 inputOffsets = inputOffsets)

    elif isinstance(platform, MemPoolPlatform):

        if loweringOptimizer is None:
            loweringOptimizer = MemPoolOptimizer

        if default_channels_first is None:
            default_channels_first = True

        deployer = MemPoolDeployer(graph,
                                   platform,
                                   inputTypes,
                                   loweringOptimizer,
                                   scheduler,
                                   name = name,
                                   default_channels_first = default_channels_first,
                                   deeployStateDir = deeployStateDir,
                                   inputOffsets = inputOffsets)

    elif isinstance(platform, SoftHierPlatform):

        if loweringOptimizer is None:
            loweringOptimizer = SoftHierOptimizer

        if default_channels_first is None:
            default_channels_first = True

        deployer = SoftHierDeployer(graph,
                                    platform,
                                    inputTypes,
                                    loweringOptimizer,
                                    scheduler,
                                    name = name,
                                    default_channels_first = default_channels_first,
                                    deeployStateDir = deeployStateDir,
                                    inputOffsets = inputOffsets)

    elif isinstance(platform, GenericPlatform):
        # WIESEP: CMSIS performs add-multiply-divide and we normally do multiply-add-divide
        #         Because these deployer were fine-tuned with a add-multiply-divide aware deployer can emulate this
        #         behavior with the EmulateCMSISRequantPass

        if loweringOptimizer is None:
            loweringOptimizer = GenericOptimizer

        if default_channels_first is None:
            default_channels_first = True

        deployer = GenericDeployer(graph,
                                   platform,
                                   inputTypes,
                                   loweringOptimizer,
                                   scheduler,
                                   name = name,
                                   default_channels_first = default_channels_first,
                                   deeployStateDir = deeployStateDir,
                                   inputOffsets = inputOffsets)

    elif isinstance(platform, (NeurekaPlatform, MemoryNeurekaPlatform, MemoryNeurekaPlatformWrapper)):

        if loweringOptimizer is None:
            loweringOptimizer = NeurekaOptimizer

        if default_channels_first is None:
            default_channels_first = False

        deployer = NeurekaDeployer(graph,
                                   platform,
                                   inputTypes,
                                   loweringOptimizer,
                                   scheduler,
                                   name = name,
                                   default_channels_first = default_channels_first,
                                   deeployStateDir = deeployStateDir)

    elif isinstance(platform, (GAP9Platform, MemoryGAP9Platform, MemoryGAP9PlatformWrapper)):

        if loweringOptimizer is None:
            loweringOptimizer = PULPOptimizer

        if default_channels_first is None:
            default_channels_first = False

        deployer = GAP9Deployer(graph,
                                platform,
                                inputTypes,
                                loweringOptimizer,
                                scheduler,
                                name = name,
                                default_channels_first = default_channels_first,
                                deeployStateDir = deeployStateDir)

    elif isinstance(platform, (PULPPlatform, MemoryPULPPlatform, MemoryPULPPlatformWrapper)):

        if loweringOptimizer is None:
            loweringOptimizer = PULPOptimizer

        if default_channels_first is None:
            default_channels_first = False

        deployer = PULPDeployer(graph,
                                platform,
                                inputTypes,
                                loweringOptimizer,
                                scheduler,
                                name = name,
                                default_channels_first = default_channels_first,
                                deeployStateDir = deeployStateDir)

    elif isinstance(platform, (SnitchPlatform)):
        if loweringOptimizer is None:
            loweringOptimizer = SnitchOptimizer

        if default_channels_first is None:
            default_channels_first = False

        deployer = SnitchDeployer(graph,
                                  platform,
                                  inputTypes,
                                  loweringOptimizer,
                                  scheduler,
                                  name = name,
                                  default_channels_first = default_channels_first,
                                  deeployStateDir = deeployStateDir)

    elif isinstance(platform, (ChimeraPlatform)):
        if loweringOptimizer is None:
            loweringOptimizer = ChimeraOptimizer

        if default_channels_first is None:
            default_channels_first = False

        deployer = ChimeraDeployer(graph,
                                   platform,
                                   inputTypes,
                                   loweringOptimizer,
                                   scheduler,
                                   name = name,
                                   default_channels_first = default_channels_first,
                                   deeployStateDir = deeployStateDir)

    else:
        raise RuntimeError(f"Deployer for platform {platform} is not implemented")

    return deployer
