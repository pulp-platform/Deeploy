# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate
from Deeploy.Targets.Generic.Templates.SliceTemplate import _SliceTemplate
from Deeploy.Targets.PULPOpen.TileConstraints.SliceConstraint import SliceTileConstraint


class _TiledSliceTemplate(_SliceTemplate):
    """Copy a Slice using scalar geometry that can be replaced for each tile."""

    def alignToContext(self, ctxt, operatorRepresentation):
        ctxt, rep, buffers = super().alignToContext(ctxt, operatorRepresentation)
        starts, steps = SliceTileConstraint.normalizedParameters(rep, ctxt)
        rank = len(starts)
        code = ["// Slice with tile-local coordinates", "BEGIN_SINGLE_CORE", "{", "uint32_t slice_out_idx = 0;"]
        for axis in range(rank):
            rep[f'slice_in_dim_{axis}'] = rep['data_in_shape'][axis]
            rep[f'slice_out_dim_{axis}'] = rep['data_out_shape'][axis]
            rep[f'slice_start_{axis}'] = starts[axis]
            # Explicit scalar placeholders let TilingVariableReplacement substitute
            # each dimension/start with its per-tile runtime value.
            code.append(f"for (int64_t i_{axis} = 0; i_{axis} < ${{slice_out_dim_{axis}}}; ++i_{axis}) {{")
        code.append("int64_t slice_in_idx = 0;")
        for axis, step in enumerate(steps):
            code.append(f"slice_in_idx = slice_in_idx * ${{slice_in_dim_{axis}}}"
                        f" + ${{slice_start_{axis}}} + i_{axis} * ({step});")
        code.append("${data_out}[slice_out_idx++] = ${data_in}[slice_in_idx];")
        code.extend(["}"] * rank)
        code.extend(["}", "END_SINGLE_CORE"])
        self.template = NodeTemplate("\n".join(code)).template
        return ctxt, rep, buffers


referenceTemplate = _TiledSliceTemplate("")
