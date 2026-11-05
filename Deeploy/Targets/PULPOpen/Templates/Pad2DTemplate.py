# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation
from Deeploy.Targets.Generic.Templates.PadTemplate import _Pad2DTemplate


class _TiledPad2DTemplate(NodeTemplate):
    """Pad2D whose per-tile geometry is given by scalar placeholders that can be replaced for each tile.

    Naming follows the Pad parsers: ``*_x`` is the height (H) and ``*_y`` the width (W) axis.

    ``Pad2DTileConstraint`` only splits the output along H, so these placeholders can differ between tiles and
    are then rendered as ``*ref`` instead of a literal: ``dim_im_out_x``, ``dim_im_in_x`` (height of the
    transferred input buffer), ``pad_y_top``, ``pad_y_bottom`` (tile-local top/bottom padding) and
    ``data_out_size``. They are therefore only used inside ``${...}`` and combined in C, never in Python.

    ``batch``, the widths, the channels, ``pad_x`` and ``value`` are constant across tiles because the policy
    constraint keeps batch, W and C whole; they are still read in C, so the kernel stays correct if that changes.
    Only ``channels_first``, the data type and (integer kernel) ``value`` are evaluated in Python; none of them is
    a tiling replacement.

    Untiled, ``alignToContext`` sets ``pad_y_top = pad_y_bottom = pad_y``. With ``alignValue`` (integer kernels)
    it also aligns ``value`` to the input signedness like the Generic integer Pad template.

    The number of copied rows is ``dim_im_out_x - pad_y_top - pad_y_bottom`` and not ``dim_im_in_x``: the tiler
    transfers at least one input row even for a tile that only contains bottom padding, and that row must not be
    copied into the output. ``dim_im_in_x`` is only used as the row count of the input buffer (stride between
    channels / batches).
    """

    def __init__(self, templateStr: str, alignValue: bool):
        super().__init__(templateStr)
        self.alignValue = alignValue

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        nameList: List[str] = []
        if self.alignValue:
            ctxt, operatorRepresentation, nameList = _Pad2DTemplate.alignToContext(self, ctxt, operatorRepresentation)
        operatorRepresentation['pad_y_top'] = operatorRepresentation['pad_y']
        operatorRepresentation['pad_y_bottom'] = operatorRepresentation['pad_y']
        return ctxt, operatorRepresentation, nameList


_fillTemplates = {
    # memset only replicates the low byte, which is exact for 8-bit types and for a zero value.
    "int":
        """\
    % if data_out_type.referencedType.typeWidth == 8 or value == 0:
    memset(${data_out}, ${value}, pad_out_size * sizeof(${data_out_type.referencedType.typeName}));
    % else:
    for (uint32_t i = 0; i < pad_out_size; i++) {
        ${data_out}[i] = ${value};
    }
    % endif
""",
    "float":
        """\
    for (uint32_t i = 0; i < pad_out_size; i++) {
        ${data_out}[i] = ${value};
    }
""",
}

_bodyTemplate = """
// 2D Pad (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    const uint32_t pad_batch = ${batch};
    const uint32_t pad_ch = ${dim_im_in_ch};
    const uint32_t pad_h_in = ${dim_im_in_x};
    const uint32_t pad_w_in = ${dim_im_in_y};
    const uint32_t pad_h_out = ${dim_im_out_x};
    const uint32_t pad_w_out = ${dim_im_out_y};
    const uint32_t pad_top = ${pad_y_top};
    const uint32_t pad_bottom = ${pad_y_bottom};
    const uint32_t pad_left = ${pad_x};
    const uint32_t pad_out_size = ${data_out_size};
    // Rows of this tile that hold input data; zero for a tile made only of padding.
    const uint32_t pad_rows = (pad_h_out > pad_top + pad_bottom) ? (pad_h_out - pad_top - pad_bottom) : 0;

FILL_PLACEHOLDER
    % if channels_first:
    // NCHW Layout
    const uint32_t pad_plane_in = pad_h_in * pad_w_in;
    const uint32_t pad_plane_out = pad_h_out * pad_w_out;
    for (uint32_t n = 0; n < pad_batch; n++) {
        for (uint32_t c = 0; c < pad_ch; c++) {
            const uint32_t pad_nc = n * pad_ch + c;
            uint32_t pad_offset_in = pad_nc * pad_plane_in;
            uint32_t pad_offset_out = pad_nc * pad_plane_out + pad_top * pad_w_out + pad_left;
            for (uint32_t h = 0; h < pad_rows; h++) {
                memcpy(${data_out} + pad_offset_out, ${data_in} + pad_offset_in,
                       pad_w_in * sizeof(${data_out_type.referencedType.typeName}));
                pad_offset_out += pad_w_out;
                pad_offset_in += pad_w_in;
            }
        }
    }
    % else:
    // NHWC Layout
    const uint32_t pad_row_in = pad_w_in * pad_ch;
    const uint32_t pad_row_out = pad_w_out * pad_ch;
    for (uint32_t n = 0; n < pad_batch; n++) {
        uint32_t pad_offset_in = n * pad_h_in * pad_row_in;
        uint32_t pad_offset_out = (n * pad_h_out + pad_top) * pad_row_out + pad_left * pad_ch;
        for (uint32_t h = 0; h < pad_rows; h++) {
            memcpy(${data_out} + pad_offset_out, ${data_in} + pad_offset_in,
                   pad_row_in * sizeof(${data_out_type.referencedType.typeName}));
            pad_offset_out += pad_row_out;
            pad_offset_in += pad_row_in;
        }
    }
    % endif
END_SINGLE_CORE
"""

referenceInt2DTemplate = _TiledPad2DTemplate(_bodyTemplate.replace("FILL_PLACEHOLDER\n", _fillTemplates["int"]),
                                             alignValue = True)
referenceFloat2DTemplate = _TiledPad2DTemplate(_bodyTemplate.replace("FILL_PLACEHOLDER\n", _fillTemplates["float"]),
                                               alignValue = False)
