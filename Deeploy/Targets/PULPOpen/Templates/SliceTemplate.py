# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate
from Deeploy.Targets.Generic.Templates.SliceTemplate import _SliceTemplate
from Deeploy.Targets.PULPOpen.TileConstraints.SliceConstraint import SliceTileConstraint


class _TiledSliceTemplate(_SliceTemplate):
    """Multi-core Slice whose geometry is given by scalar placeholders that can be replaced for each tile.

    For every axis ``k`` the kernel uses ``slice_in_dim_k`` (extent of the input buffer it reads),
    ``slice_out_dim_k`` (extent of the output tile) and ``slice_start_k`` (index of the first selected input
    element inside that buffer). Untiled, ``alignToContext`` sets them to the full shapes and the normalized
    starts; tiled, ``SliceTileConstraint`` replaces them per tile. Steps are constants and emitted as literals.

    The output elements are enumerated in row-major order with one linear index, which is split into
    contiguous chunks across the cluster cores. Each core decodes its first index once and then walks
    innermost-axis row segments; a segment is copied with ``memcpy`` when the innermost step is 1 and with a
    strided scalar loop otherwise. All per-axis input indices ``start_k + i_k * step_k`` lie in
    ``[0, slice_in_dim_k - 1]`` for both step signs, so the flattened input index is never negative.
    """

    def alignToContext(self, ctxt, operatorRepresentation):
        ctxt, rep, buffers = super().alignToContext(ctxt, operatorRepresentation)
        starts, steps = SliceTileConstraint.normalizedParameters(rep, ctxt)
        rank = len(starts)
        for axis in range(rank):
            rep[f'slice_in_dim_{axis}'] = rep['data_in_shape'][axis]
            rep[f'slice_out_dim_{axis}'] = rep['data_out_shape'][axis]
            rep[f'slice_start_{axis}'] = starts[axis]

        last = rank - 1
        lastStep = steps[last]
        typeWidth = rep['data_out_type'].referencedType.typeWidth // 8

        def inIndex(axis):
            return f"((int32_t) ${{slice_start_{axis}}} + (int32_t) i_{axis} * ({steps[axis]}))"

        code = [
            "// Slice (Name: ${nodeName}, Op: ${nodeOp})",
            "{",
            "const uint32_t slice_core_id = pi_core_id();",
            "const uint32_t slice_size = " + " * ".join(f"${{slice_out_dim_{axis}}}" for axis in range(rank)) + ";",
            # Same balanced split as the PULP Transpose kernel: the first `leftover` cores get one more element.
            "const uint32_t slice_base_chunk = slice_size / NUM_CORES;",
            "const uint32_t slice_leftover = slice_size - slice_base_chunk * NUM_CORES;",
            "uint32_t slice_n = slice_base_chunk * slice_core_id"
            " + (slice_core_id < slice_leftover ? slice_core_id : slice_leftover);",
            "const uint32_t slice_end = slice_n + slice_base_chunk + (slice_core_id < slice_leftover ? 1 : 0);",
            # Decode the first output element of this core into per-axis indices.
            "uint32_t slice_rem = slice_n;",
        ]
        for axis in reversed(range(rank)):
            if axis == 0:
                code.append("uint32_t i_0 = slice_rem;")
            else:
                code.append(f"uint32_t i_{axis} = slice_rem % ${{slice_out_dim_{axis}}};")
                code.append(f"slice_rem /= ${{slice_out_dim_{axis}}};")

        code.append("while (slice_n < slice_end) {")
        # Input index of the current element, Horner form over the input buffer extents.
        inExpr = inIndex(0)
        for axis in range(1, rank):
            inExpr = f"({inExpr} * (int32_t) ${{slice_in_dim_{axis}}} + {inIndex(axis)})"
        code.append(f"int32_t slice_in_idx = {inExpr};")
        code.append(f"uint32_t slice_cnt = ${{slice_out_dim_{last}}} - i_{last};")
        code.append("if (slice_cnt > slice_end - slice_n) { slice_cnt = slice_end - slice_n; }")
        if lastStep == 1:
            code.append(f"memcpy(${{data_out}} + slice_n, ${{data_in}} + slice_in_idx, slice_cnt * {typeWidth});")
        else:
            code.append("for (uint32_t slice_j = 0; slice_j < slice_cnt; ++slice_j) {")
            code.append("${data_out}[slice_n + slice_j] = ${data_in}[slice_in_idx];")
            code.append(f"slice_in_idx += ({lastStep});")
            code.append("}")
        code.append("slice_n += slice_cnt;")
        # Advance to the start of the next innermost row (odometer over the outer axes). Overflow of i_0 after
        # the last row is harmless because the loop terminates.
        code.append(f"i_{last} = 0;")
        closing = []
        for axis in reversed(range(last)):
            code.append(f"if (++i_{axis} == ${{slice_out_dim_{axis}}}) {{" if axis > 0 else "++i_0;")
            if axis > 0:
                code.append(f"i_{axis} = 0;")
                closing.append("}")
        code.extend(closing)
        code.extend(["}", "}"])

        self.template = NodeTemplate("\n".join(code)).template
        return ctxt, rep, buffers


referenceTemplate = _TiledSliceTemplate("")
