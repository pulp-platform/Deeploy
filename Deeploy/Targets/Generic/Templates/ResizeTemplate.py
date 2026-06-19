# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import FloatImmediate
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

_TYPE_TAG_MAP = {
    'fp32': 'RESIZE_TYPE_FLOAT32',
    's8': 'RESIZE_TYPE_INT8',
    'u8': 'RESIZE_TYPE_UINT8',
    's16': 'RESIZE_TYPE_INT16',
    'u16': 'RESIZE_TYPE_UINT16',
    's32': 'RESIZE_TYPE_INT32',
    'u32': 'RESIZE_TYPE_UINT32',
}

_MODE_MAP = {
    "nearest": "RESIZE_MODE_NEAREST",
    "linear": "RESIZE_MODE_LINEAR",
    "cubic": "RESIZE_MODE_CUBIC",
}

_COORD_MAP = {
    "asymmetric": "RESIZE_COORD_ASYMMETRIC",
    "half_pixel": "RESIZE_COORD_HALF_PIXEL",
    "half_pixel_symmetric": "RESIZE_COORD_HALF_PIXEL_SYMMETRIC",
    "align_corners": "RESIZE_COORD_ALIGN_CORNERS",
    "pytorch_half_pixel": "RESIZE_COORD_PYTORCH_HALF_PIXEL",
    "tf_crop_and_resize": "RESIZE_COORD_TF_CROP_AND_RESIZE",
}

_NEAREST_MAP = {
    "floor": "RESIZE_NEAREST_FLOOR",
    "ceil": "RESIZE_NEAREST_CEIL",
    "round_prefer_floor": "RESIZE_NEAREST_ROUND_PREFER_FLOOR",
    "round_prefer_ceil": "RESIZE_NEAREST_ROUND_PREFER_CEIL",
}


def _typeSuffix(ref_type) -> str:
    if issubclass(ref_type, FloatImmediate):
        return f'fp{ref_type.typeWidth}'
    elif ref_type.signed:
        return f's{ref_type.typeWidth}'
    else:
        return f'u{ref_type.typeWidth}'


class _ResizeTemplate(NodeTemplate):

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        rep = operatorRepresentation

        if rep.get('roi', None) is not None:
            raise ValueError("Resize: 'roi' input is not supported.")
        if rep.get('scales', None) is not None:
            raise ValueError("Resize: 'scales' input is not supported; use 'sizes' instead. ")
        if rep.get('antialias', 0) != 0:
            raise ValueError(f"Resize: antialias={rep['antialias']} is not supported by this kernel.")
        if rep.get('exclude_outside', 0) != 0:
            raise ValueError(f"Resize: exclude_outside={rep['exclude_outside']} is not supported by this kernel.")
        if rep.get('axes', None) is not None:
            raise ValueError(f"Resize: axes={rep['axes']} is not supported; all axes must be resized.")
        if rep.get('keep_aspect_ratio_policy', 'stretch') != 'stretch':
            raise ValueError(
                f"Resize: keep_aspect_ratio_policy='{rep['keep_aspect_ratio_policy']}' is not supported by this kernel."
            )
        if rep.get('coord_mode', 'half_pixel') == 'tf_crop_and_resize':
            raise ValueError(
                "Resize: coordinate_transformation_mode='tf_crop_and_resize' is not supported by this kernel.")
        if rep.get('mode', 'nearest') == 'cubic':
            raise ValueError("Resize: mode='cubic' is not supported by this kernel.")

        data_in = ctxt.lookup(rep['data_in'])
        type_suffix = _typeSuffix(data_in._type.referencedType)
        rep['type_tag'] = _TYPE_TAG_MAP[type_suffix]
        rep['mode'] = _MODE_MAP[rep['mode']]
        rep['coord_mode'] = _COORD_MAP[rep['coord_mode']]
        rep['nearest_mode'] = _NEAREST_MAP[rep['nearest_mode']]

        return ctxt, rep, []


referenceTemplate = _ResizeTemplate("""
// Resize (Name: ${nodeName}, Op: ${nodeOp})
Resize(
    ${data_in}, ${data_out}, ${type_tag},
    ${batch_size}, ${channels}, ${spatial_dims},
    (int32_t[]){${', '.join(str(s) for s in input_shape)}},
    (int32_t[]){${', '.join(str(s) for s in output_shape)}},
    ${mode}, ${coord_mode}, ${nearest_mode}
);
""")
