# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import tempfile
from typing import Callable, Dict, Optional, Type

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.DeeployTypes import DeploymentPlatform, TopologyOptimizer
from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.Targets.XDNA2.Templates.AddTemplate import XDNA2NodeTemplate

# JUNGVI: Will be removed once Deeploy generates it's own MLIR

# Default path to the mlir-aie Python environment.
# Can be overridden via the MLIR_AIE_PYTHON env variable.
_DEFAULT_IRON_PYTHON = os.environ.get(
    "MLIR_AIE_PYTHON",
    "/scratch/jungvi/micromamba/envs/iron/bin/python",
)

# Path to the IRON design scripts shipped with mlir-aie examples.
# Can be overridden via the IRON_OPERATORS_DIR env variable.
_DEFAULT_IRON_OPERATORS_DIR = os.environ.get(
    "IRON_OPERATORS_DIR",
    "/scratch/jungvi/IRON/iron/operators",
)


class XDNA2Deployer(SignPropDeployer):
    """Deployer for the XDNA2 (AIE2p) platform.

    Unlike other Deeploy deployers that generate C code, this deployer
    generates an mlir-aie MLIR module.  The MLIR is produced by invoking the
    IRON operator ``design.py`` scripts as subprocesses (using the mlir-aie
    Python environment) so that the main Deeploy environment does not need to
    have ``aie.iron`` installed.

    It also writes ``testinputs.h`` and ``testoutputs.h`` via the XDNA2
    generation script so the XRT C++ testbench can be compiled against
    known-good golden values.
    """

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = False,
                 deeployStateDir: str = "DeeployStateDir",
                 inputOffsets: Optional[Dict[str, int]] = None,
                 iron_python: Optional[str] = None,
                 iron_operators_dir: Optional[str] = None):
        """
        Parameters
        ----------
        iron_python : str, optional
            Path to the Python interpreter in the mlir-aie (IRON) environment.
            Defaults to ``MLIR_AIE_PYTHON`` env variable or
            ``/scratch/jungvi/micromamba/envs/iron/bin/python``.
        iron_operators_dir : str, optional
            Path to the IRON operators directory containing per-operator
            ``design.py`` scripts.
            Defaults to ``IRON_OPERATORS_DIR`` env variable or
            ``/scratch/jungvi/IRON/iron/operators``.
        """
        super().__init__(
            graph,
            deploymentPlatform,
            inputTypes,
            loweringOptimizer,
            scheduler,
            name,
            default_channels_first = default_channels_first,
            deeployStateDir = deeployStateDir,
            inputOffsets = inputOffsets if inputOffsets is not None else {},
        )
        self._iron_python = iron_python or _DEFAULT_IRON_PYTHON
        self._iron_operators_dir = iron_operators_dir or _DEFAULT_IRON_OPERATORS_DIR

    # ------------------------------------------------------------------
    # MLIR generation
    # ------------------------------------------------------------------

    def generateMLIR(self) -> str:
        """Generate an mlir-aie MLIR module for the prepared graph.

        Iterates over ``self.layerBinding``, extracts AIE parameters from each
        bound template, and calls the corresponding IRON ``design.py`` script
        as a subprocess.  Currently only a single BF16 Add node is supported.

        Returns
        -------
        str
            MLIR module string (ready to be written to ``network.mlir``).

        Raises
        ------
        RuntimeError
            If the graph contains unsupported operators or if the IRON
            subprocess fails.
        """
        assert self.prepared, "XDNA2Deployer.generateMLIR() called before prepare()"

        mlir_parts = []

        for node_name, layer in self.layerBinding.items():
            mapper = layer.mapper
            template = mapper.binder.template
            op_repr = mapper.parser.operatorRepresentation

            if not isinstance(template, XDNA2NodeTemplate):
                raise RuntimeError(
                    f"Node '{node_name}' has no XDNA2NodeTemplate — "
                    f"only BF16 Add is supported in this release.")

            aie_params = template.getAIEParams(op_repr)
            log.info(f"[XDNA2] Generating MLIR for node '{node_name}' "
                     f"with params: {aie_params}")

            mlir_str = self._generate_add_mlir(aie_params)
            mlir_parts.append(mlir_str)

        if not mlir_parts:
            raise RuntimeError("No bound layers found in graph — cannot generate MLIR.")

        # For a single-node graph the MLIR is just the one module.
        # Multi-node support would require merging modules.
        return mlir_parts[0]

    def _generate_add_mlir(self, aie_params: dict) -> str:
        """Call the IRON elementwise_add design.py to produce MLIR.

        Parameters
        ----------
        aie_params : dict
            Dict with keys: num_elements, n_cols, n_channels, tile_size, trace_size.

        Returns
        -------
        str
            MLIR module string.
        """
        design_script = os.path.join(
            self._iron_operators_dir, "elementwise_add", "design.py"
        )

        if not os.path.isfile(design_script):
            raise RuntimeError(
                f"IRON design script not found: {design_script}\n"
                f"Set IRON_OPERATORS_DIR to point to the IRON operators directory.")

        if not os.path.isfile(self._iron_python):
            raise RuntimeError(
                f"IRON Python interpreter not found: {self._iron_python}\n"
                f"Set MLIR_AIE_PYTHON to the mlir-aie Python interpreter.")

        with tempfile.NamedTemporaryFile(suffix=".mlir", delete=False) as tmp:
            output_path = tmp.name

        try:
            cmd = [
                self._iron_python,
                design_script,
                "--dev", "npu2",
                "--length", str(aie_params['num_elements']),
                "--columns", str(aie_params['n_cols']),
                "--channels", str(aie_params['n_channels']),
                "--tile-size", str(aie_params['tile_size']),
                "--trace-size", str(aie_params['trace_size']),
                "--output-file-path", output_path,
            ]

            log.debug(f"[XDNA2] Running: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"IRON design.py failed (exit {result.returncode}):\n"
                    f"  cmd: {' '.join(cmd)}\n"
                    f"  stdout: {result.stdout}\n"
                    f"  stderr: {result.stderr}")

            with open(output_path, 'r') as f:
                mlir_str = f.read()

        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

        return mlir_str
