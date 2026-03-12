# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate


class XDNA2NodeTemplate(NodeTemplate):
    """Base class for XDNA2 templates.

    Temporary Feature:
    Unlike Mako-based templates for C code, XDNA2 templates do not produce
    code snippets. Instead they store AIE kernel metadata that the
    XDNA2Deployer reads when generating the holistic MLIR module.
    """

    def __init__(self, kernel_fn_name: str, kernel_obj: str, kernel_src: str, tile_size: int = 1024):
        """Initialize an XDNA2NodeTemplate.

        Parameters
        ----------
        kernel_fn_name : str
            Name of the AIE C++ kernel function (e.g. "eltwise_add_bf16_vector").
        kernel_obj : str
            Compiled kernel object file name (e.g. "add.o").
        kernel_src : str
            Kernel source file name relative to TargetLibraries/XDNA2/kernels/
            (e.g. "add.cc").
        tile_size : int
            Number of elements per tile (default 1024, max 4096).
        """
        # Empty Mako template — no C code is generated per node.
        super().__init__("")
        self.kernel_fn_name = kernel_fn_name
        self.kernel_obj = kernel_obj
        self.kernel_src = kernel_src
        self.tile_size = tile_size

    def getAIEParams(self, operatorRepresentation: dict) -> dict:
        """Return the aie.iron parameters for this node.

        Parameters
        ----------
        operatorRepresentation : dict
            The operator representation dict produced by the parser.

        Returns
        -------
        dict
            Parameters to pass to the corresponding aie.iron design function.
        """
        raise NotImplementedError


class XDNA2AddTemplate(XDNA2NodeTemplate):
    """XDNA2 template for BF16 elementwise Add."""

    def __init__(self):
        super().__init__(
            kernel_fn_name = "eltwise_add_bf16_vector",
            kernel_obj = "add.o",
            kernel_src = "add.cc",
            tile_size = 1024,
        )

    def getAIEParams(self, operatorRepresentation: dict) -> dict:
        num_elements = int(operatorRepresentation['size'])
        tile_size = min(num_elements, self.tile_size)
        # Ensure num_elements is divisible by tile_size
        if num_elements % tile_size != 0:
            tile_size = 1
        return {
            'num_elements': num_elements,
            'n_cols': 1,
            'n_channels': 1,
            'tile_size': tile_size,
            'trace_size': 0,
        }


referenceTemplate = XDNA2AddTemplate()
