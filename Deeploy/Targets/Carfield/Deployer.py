# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.DeeployTypes import ConstantBuffer, DeploymentPlatform, TopologyOptimizer, VariableBuffer
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.Targets.Generic.Deployer import GenericDeployer


def _align(value: int, alignment: int) -> int:
    """Round `value` up to the next multiple of `alignment`."""
    return (value + alignment - 1) // alignment * alignment


class CarfieldDeployer(GenericDeployer):
    """Deployer for the Carfield PULP cluster.

    Adds the static DDR layout on top of the Generic flow: every ConstantBuffer
    (the weights and biases of the network) is assigned a fixed DDR address at
    compile time, so that `CarfieldConstantBuffer.initTemplate` can emit a plain
    pointer instead of a statically initialised array. The tensor data itself
    never enters the binary: tensor data is written to a single binary file and pushed into DDR
    once per session with `gdb restore`.

    This class is the single source of truth for the layout. Whoever writes the
    binary files must read the addresses assigned here (`buf.address`) rather than
    recomputing the order, otherwise the pointers and the bytes can silently
    disagree.
    """

    #: Base address of the weights region in DDR. The binary file is restored here.
    ddrBaseAddress: int = 0x80000000

    #: 64 keeps each tensor on its own cache line and costs a few KiB over the whole program. The binary file writer MUST use
    #: the same value.
    ddrAlignment: int = 64

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first = True,
                 deeployStateDir: str = "DeeployStateDir",
                 inputOffsets: Dict[str, int] = {}):

        super().__init__(graph,
                         deploymentPlatform,
                         inputTypes,
                         loweringOptimizer,
                         scheduler,
                         name,
                         default_channels_first = default_channels_first,
                         deeployStateDir = deeployStateDir,
                         inputOffsets = inputOffsets)

        self.weightsRegion: Tuple[int, int] = (self.ddrBaseAddress, self.ddrBaseAddress)
        self.inputBlobAddress: Optional[int] = None
        self.goldenBlobAddress: Optional[int] = None
        self.arenaBaseAddress: Optional[int] = None

    def _constantBuffers(self) -> List[ConstantBuffer]:
        """Constants to place in DDR, in the order they will be written to the binary file."""
        return [
            buf for buf in self.ctxt.globalObjects.values()
            if isinstance(buf, ConstantBuffer) and buf._deploy
        ]

    @staticmethod
    def _bufferSizeInBytes(buf: VariableBuffer) -> int:

        if isinstance(buf, ConstantBuffer):
            nElements = int(buf.values.size)
        else:
            nElements = int(np.prod(buf.shape))

        return nElements * (buf._type.referencedType.typeWidth // 8)

    def _assignDDRAddresses(self) -> None:
        """Lay out the static DDR regions and stamp the addresses on the buffers.

        Layout, all regions 64-byte aligned:

            0x80000000  weights.bin   every ConstantBuffer, back to back
                        input.bin     the network input tensor
                        golden.bin    the expected output, for the on-device check
                        arena         everything deeploy_malloc hands out
        """
        address = self.ddrBaseAddress

        for buf in self._constantBuffers():
            # Hex string, not int: Mako would render an int in decimal.
            buf.address = f"0x{address:08X}"
            address = _align(address + self._bufferSizeInBytes(buf), self.ddrAlignment)

        self.weightsRegion = (self.ddrBaseAddress, address)

        # The input and golden .bin files are not ConstantBuffers: they are test
        # vectors, restored next to the weights. They get an "address" too, even
        # though no template reads it.
        self.inputBlobAddress = address
        for buf in self.inputs():
            buf.address = f"0x{address:08X}"
            address = _align(address + self._bufferSizeInBytes(buf), self.ddrAlignment)

        self.goldenBlobAddress = address
        for buf in self.outputs():
            buf.address = f"0x{address:08X}"
            address = _align(address + self._bufferSizeInBytes(buf), self.ddrAlignment)

        # The arena starts on a 1 MiB boundary purely so the address stays easy
        # to read and to type into gdb.
        self.arenaBaseAddress = _align(address, 1 << 20)

    def _logDDRLayout(self) -> None:
        """Print the layout: the gdb restore commands are typed by hand."""
        constants = self._constantBuffers()
        weightsBytes = self.weightsRegion[1] - self.weightsRegion[0]

        log.info("")
        log.info("Carfield DDR layout:")
        log.info(f"  {'Region':<14} {'Address':>12} {'Bytes':>14}  Contents")
        log.info("  " + "-" * 72)
        log.info(f"  {'weights.bin':<14} 0x{self.weightsRegion[0]:08X} {weightsBytes:>14,d}  "
                 f"{len(constants)} constant tensor(s)")
        log.info(f"  {'input.bin':<14} 0x{self.inputBlobAddress:08X} "
                 f"{sum(self._bufferSizeInBytes(b) for b in self.inputs()):>14,d}  "
                 f"{len(self.inputs())} network input(s)")
        log.info(f"  {'golden.bin':<14} 0x{self.goldenBlobAddress:08X} "
                 f"{sum(self._bufferSizeInBytes(b) for b in self.outputs()):>14,d}  "
                 f"{len(self.outputs())} expected output(s)")
        log.info(f"  {'arena':<14} 0x{self.arenaBaseAddress:08X} {'-':>14}  deeploy_malloc")
        log.info("")

    def bind(self) -> bool:
        """Bind the network, then freeze the static DDR layout."""
        if not super().bind():
            return False

        self._assignDDRAddresses()
        self._logDDRLayout()

        return True
