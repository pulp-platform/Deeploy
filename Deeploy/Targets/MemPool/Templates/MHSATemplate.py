# ----------------------------------------------------------------------
#
# File: MHSATemplate.py
#
# Last edited: 30.10.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import int8_t
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation

# ITA Configuration
_SPLIT = 4
_ITA_S = 64
_ITA_E = 64
_ITA_P = 64


class _MHSATemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:
        nameList = []

        nodeName = operatorRepresentation['nodeName']
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        wq_bias = ctxt.lookup(operatorRepresentation['wq_bias'])
        wk_bias = ctxt.lookup(operatorRepresentation['wk_bias'])
        wv_bias = ctxt.lookup(operatorRepresentation['wv_bias'])
        wo_bias = ctxt.lookup(operatorRepresentation['wo_bias'])
        wq_weight = ctxt.lookup(operatorRepresentation['wq_weight'])
        wk_weight = ctxt.lookup(operatorRepresentation['wk_weight'])
        wv_weight = ctxt.lookup(operatorRepresentation['wv_weight'])
        wo_weight = ctxt.lookup(operatorRepresentation['wo_weight'])
        q = ctxt.lookup(operatorRepresentation['q'])
        k = ctxt.lookup(operatorRepresentation['k'])

        # Disable buffers
        wq_bias._deploy = False
        wk_bias._deploy = False
        wv_bias._deploy = False
        wo_bias._deploy = False
        wq_weight._deploy = False
        wk_weight._deploy = False
        wv_weight._deploy = False
        wo_weight._deploy = False

        operatorRepresentation['S'] = operatorRepresentation['dim']
        operatorRepresentation['P'] = operatorRepresentation['dim_head']

        N = operatorRepresentation['heads']
        S = operatorRepresentation['S']
        E = operatorRepresentation['E']
        P = operatorRepresentation['P']

        PAD_S = _ITA_S - S
        PAD_E = _ITA_E - E
        PAD_P = _ITA_P - P

        # Extract values and transform them to layout required by ITA
        wq_bias_ita = wq_bias.values.reshape(N, S, P)
        wq_bias_ita = np.pad(wq_bias_ita, ((0, 0), (0, PAD_S), (0, PAD_P)))
        wq_bias_ita = np.reshape(np.split(wq_bias_ita, _SPLIT, axis = 2), (N, _ITA_S, _ITA_P))

        wk_bias_ita = wk_bias.values.reshape(N, S, P)
        wk_bias_ita = np.pad(wk_bias_ita, ((0, 0), (0, PAD_S), (0, PAD_P)))
        wk_bias_ita = np.reshape(np.split(wk_bias_ita, _SPLIT, axis = 2), (N, _ITA_S, _ITA_P))

        wv_bias_ita = wv_bias.values.reshape(N, S, P)
        wv_bias_ita = np.pad(wv_bias_ita, ((0, 0), (0, PAD_S), (0, PAD_P)))
        wv_bias_ita = np.reshape(np.split(np.reshape(np.transpose(wv_bias_ita), (N, _ITA_P, _ITA_S)), _SPLIT, axis = 2),
                                 (N, _ITA_P, _ITA_S))

        wo_bias_ita = wo_bias.values.reshape(N, S, E)
        wo_bias_ita = np.pad(wo_bias_ita, ((0, 0), (0, PAD_S), (0, PAD_E)))
        wo_bias_ita = np.reshape(np.split(wo_bias_ita, _SPLIT, axis = 2), (N, _ITA_S, _ITA_E))

        wq_weight_ita = wq_weight.values.reshape(N, E, P)
        wq_weight_ita = np.pad(wq_weight_ita, ((0, 0), (0, PAD_E), (0, PAD_P)))
        wq_weight_ita = np.concatenate(
            [np.concatenate(np.split(np.transpose(wq_weight_ita[i]), _SPLIT, axis = 1)) for i in range(N)])
        wq_weight_ita = np.reshape(wq_weight_ita, (N, _ITA_P, _ITA_E))

        wk_weight_ita = wk_weight.values.reshape(N, E, P)
        wk_weight_ita = np.pad(wk_weight_ita, ((0, 0), (0, PAD_E), (0, PAD_P)))
        wk_weight_ita = np.concatenate([np.transpose(wk_weight_ita[i]) for i in range(N)])
        wk_weight_ita = np.reshape(wk_weight_ita, (N, _ITA_P, _ITA_E))

        wv_weight_ita = wv_weight.values.reshape(N, E, P)
        wv_weight_ita = np.pad(wv_weight_ita, ((0, 0), (0, PAD_E), (0, PAD_P)))
        wv_weight_ita = np.concatenate([np.transpose(wv_weight_ita[i]) for i in range(N)])
        wv_weight_ita = np.reshape(wv_weight_ita, (N, _ITA_P, _ITA_E))

        wo_weight_ita = wo_weight.values.reshape(N, P, E)
        wo_weight_ita = np.pad(wo_weight_ita, ((0, 0), (0, PAD_P), (0, PAD_E)))
        wo_weight_ita = np.concatenate([np.transpose(wo_weight_ita[i]) for i in range(N)])
        wo_weight_ita = np.reshape(wo_weight_ita, (N, _ITA_E, _ITA_P))

        # Create dummy array for key and values
        q_ita = np.zeros((1, _ITA_S, _ITA_E))
        k_ita = np.zeros((1, _ITA_S, _ITA_E))

        # Fuse all inputs together and store in L2
        data = np.stack([
            wo_weight_ita,
            wv_weight_ita,
            wk_weight_ita,
            q_ita,
            k_ita,
            wq_weight_ita,
            wo_bias_ita,
            wv_bias_ita,
            wk_bias_ita,
            wq_bias_ita,
        ])

        data_in = ctxt.ConstantBuffer(name = f'{nodeName}_input', shape = data.shape, values = data)
        ctxt.add(data_in, 'global')
        data_in._type = PointerClass(int8_t)
        operatorRepresentation['data_in'] = data_in.name
        nameList += [data_in.name]

        requant_mult_data = np.array([
            operatorRepresentation['wq_requant_mul'],
            operatorRepresentation['wk_requant_mul'],
            operatorRepresentation['preattn_requant_mul'],
            operatorRepresentation['wv_requant_mul'],
            operatorRepresentation['postattn_requant_mul'],
            operatorRepresentation['wo_requant_mul'],
            0,
            0,
        ])
        requant_mult = ctxt.ConstantBuffer(name = f'{nodeName}_requant_mult',
                                           shape = requant_mult_data.shape,
                                           values = requant_mult_data)
        ctxt.add(requant_mult, 'global')
        requant_mult._type = PointerClass(int8_t)
        operatorRepresentation['requant_mult'] = requant_mult.name
        nameList += [requant_mult.name]

        requant_shift_data = np.array([
            int(np.log2(operatorRepresentation['wq_requant_div'])),
            int(np.log2(operatorRepresentation['wk_requant_div'])),
            int(np.log2(operatorRepresentation['preattn_requant_div'])),
            int(np.log2(operatorRepresentation['wv_requant_div'])),
            int(np.log2(operatorRepresentation['postattn_requant_div'])),
            int(np.log2(operatorRepresentation['wo_requant_div'])),
            0,
            0,
        ])
        requant_shift = ctxt.ConstantBuffer(name = f'{nodeName}_requant_shift',
                                            shape = requant_shift_data.shape,
                                            values = requant_shift_data)
        ctxt.add(requant_shift, 'global')
        requant_shift._type = PointerClass(int8_t)
        operatorRepresentation['requant_shift'] = requant_shift.name
        nameList += [requant_shift.name]

        requant_add_data = np.array([
            operatorRepresentation['wq_requant_add'],
            operatorRepresentation['wk_requant_add'],
            operatorRepresentation['preattn_requant_add'],
            operatorRepresentation['wv_requant_add'],
            operatorRepresentation['postattn_requant_add'],
            operatorRepresentation['wo_requant_add'],
            0,
            0,
        ])
        requant_add = ctxt.ConstantBuffer(name = f'{nodeName}_requant_add',
                                          shape = requant_add_data.shape,
                                          values = requant_add_data)
        ctxt.add(requant_add, 'global')
        requant_add._type = PointerClass(int8_t)
        operatorRepresentation['requant_add'] = requant_add.name
        nameList += [requant_add.name]

        operatorRepresentation['q_offset'] = (q._signed == 0) * int(q.nLevels // 2)
        operatorRepresentation['k_offset'] = (k._signed == 0) * int(k.nLevels // 2)
        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels // 2)

        # import IPython; IPython.embed()
        return ctxt, operatorRepresentation, nameList


MemPoolParallelTemplate = _MHSATemplate("""
// ITA MHSA (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);

<%
    ctxt = locals()['pageargs']
    data_in_strings = ", ".join([ctxt[f"data_in_head{h}"] for h in range(heads)])
    requant_mult_strings = ", ".join([ctxt[f"requant_mult_head{h}"] for h in range(heads)])
    requant_shift_strings = ", ".join([ctxt[f"requant_shift_head{h}"] for h in range(heads)])
    requant_add_strings = ", ".join([ctxt[f"requant_add_head{h}"] for h in range(heads)])
%>
int8_t *data_in_array[] = { ${data_in_strings} };
uint8_t const *requant_mult_array[] = { ${requant_mult_strings} };
uint8_t const *requant_shift_array[] = { ${requant_shift_strings} };
int8_t const *requant_add_array[] = { ${requant_add_strings} };

MHSA_s8_ITA(
    ${q}, ${k}, data_in_array,
    ${S}, ${E},
    requant_mult_array,
    requant_shift_array,
    requant_add_array,
    ${data_out},
    ${q_offset}, ${k_offset}, ${output_offset},
    core_id,
    numThreads
);
mempool_barrier(numThreads);
""")
