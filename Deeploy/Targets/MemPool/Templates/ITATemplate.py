# ----------------------------------------------------------------------
#
# File: ITATemplate.py
#
# Last edited: 16.11.2023
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
from Deeploy.CommonExtensions.DataTypes import int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation
from Deeploy.Targets.MemPool.DataTypes import MemPoolStructDataTypes

# ITA Configuration
_ITA_PE = 16


def _transformITAInputs(ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation):
    wq_bias = ctxt.lookup(operatorRepresentation['wq_bias'])
    wk_bias = ctxt.lookup(operatorRepresentation['wk_bias'])
    wv_bias = ctxt.lookup(operatorRepresentation['wv_bias'])
    wo_bias = ctxt.lookup(operatorRepresentation['wo_bias'])
    wq_weight = ctxt.lookup(operatorRepresentation['wq_weight'])
    wk_weight = ctxt.lookup(operatorRepresentation['wk_weight'])
    wv_weight = ctxt.lookup(operatorRepresentation['wv_weight'])
    wo_weight = ctxt.lookup(operatorRepresentation['wo_weight'])

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

    # Extract values and transform them to layout required by ITA
    wq_bias_ita = wq_bias.values.reshape(N, 1, P)

    wk_bias_ita = wk_bias.values.reshape(N, 1, P)

    wv_bias_ita = wv_bias.values.reshape(N, 1, P)

    wo_bias_ita = wo_bias.values.reshape(N, 1, E)

    wq_weight_ita = wq_weight.values.reshape(N, E, P)
    wq_weight_ita = np.concatenate(
        [np.concatenate(np.split(np.transpose(wq_weight_ita[i]), E // _ITA_PE, axis = 1)) for i in range(N)])
    wq_weight_ita = np.reshape(wq_weight_ita, (N, P, E))

    wk_weight_ita = wk_weight.values.reshape(N, E, P)
    wk_weight_ita = np.concatenate([np.transpose(wk_weight_ita[i]) for i in range(N)])
    wk_weight_ita = np.reshape(wk_weight_ita, (N, P, E))

    wv_weight_ita = wv_weight.values.reshape(N, E, P)
    wv_weight_ita = np.concatenate([np.transpose(wv_weight_ita[i]) for i in range(N)])
    wv_weight_ita = np.reshape(wv_weight_ita, (N, P, E))

    wo_weight_ita = wo_weight.values.reshape(N, P, E)
    wo_weight_ita = np.concatenate([np.transpose(wo_weight_ita[i]) for i in range(N)])
    wo_weight_ita = np.reshape(wo_weight_ita, (N, E, P))

    q_ita = np.zeros((N, S, E))
    k_ita = np.zeros((N, S, E))

    return ctxt, operatorRepresentation, wq_bias_ita, wk_bias_ita, wv_bias_ita, wo_bias_ita, wq_weight_ita, wk_weight_ita, wv_weight_ita, wo_weight_ita, q_ita, k_ita


class _1HSATemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:
        nameList = []

        ctxt, operatorRepresentation, wq_bias_ita, wk_bias_ita, wv_bias_ita, wo_bias_ita, wq_weight_ita, wk_weight_ita, wv_weight_ita, wo_weight_ita, q_ita, k_ita = _transformITAInputs(
            ctxt, operatorRepresentation)

        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        q = ctxt.lookup(operatorRepresentation['q'])
        k = ctxt.lookup(operatorRepresentation['k'])

        nodeName = operatorRepresentation['nodeName']

        # Fuse all inputs together and store in L2
        wo_weight_ita = np.reshape(wo_weight_ita, (-1,))
        wv_weight_ita = np.reshape(wv_weight_ita, (-1,))
        wk_weight_ita = np.reshape(wk_weight_ita, (-1,))
        q_ita = np.reshape(q_ita, (-1,))
        k_ita = np.reshape(k_ita, (-1,))
        wq_weight_ita = np.reshape(wq_weight_ita, (-1,))
        wo_bias_ita = np.reshape(wo_bias_ita.astype(np.int32), (-1,)).view(np.int8)
        wv_bias_ita = np.reshape(wv_bias_ita.astype(np.int32), (-1,)).view(np.int8)
        wk_bias_ita = np.reshape(wk_bias_ita.astype(np.int32), (-1,)).view(np.int8)
        wq_bias_ita = np.reshape(wq_bias_ita.astype(np.int32), (-1,)).view(np.int8)

        data = np.concatenate([
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
            int(operatorRepresentation['wq_requant_mul']),
            int(operatorRepresentation['wk_requant_mul']),
            int(operatorRepresentation['preattn_requant_mul']),
            int(operatorRepresentation['wv_requant_mul']),
            int(operatorRepresentation['postattn_requant_mul']),
            int(operatorRepresentation['wo_requant_mul']),
            0,
            0,
        ])
        requant_mult = ctxt.ConstantBuffer(name = f'{nodeName}_requant_mult',
                                           shape = requant_mult_data.shape,
                                           values = requant_mult_data)
        ctxt.add(requant_mult, 'global')
        requant_mult._type = PointerClass(uint8_t)
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
        requant_shift._type = PointerClass(uint8_t)
        operatorRepresentation['requant_shift'] = requant_shift.name
        nameList += [requant_shift.name]

        requant_add_data = np.array([
            int(operatorRepresentation['wq_requant_add']),
            int(operatorRepresentation['wk_requant_add']),
            int(operatorRepresentation['preattn_requant_add']),
            int(operatorRepresentation['wv_requant_add']),
            int(operatorRepresentation['postattn_requant_add']),
            int(operatorRepresentation['wo_requant_add']),
            0,
            0,
        ])
        requant_add = ctxt.ConstantBuffer(name = f'{nodeName}_requant_add',
                                          shape = requant_add_data.shape,
                                          values = requant_add_data)
        ctxt.add(requant_add, 'global')
        requant_add._type = PointerClass(int32_t)
        operatorRepresentation['requant_add'] = requant_add.name
        nameList += [requant_add.name]

        quant_dict = {
            'eps_mult': operatorRepresentation['requant_mult'],
            'right_shift': operatorRepresentation['requant_shift'],
            'add': operatorRepresentation['requant_add']
        }
        nameList += [ctxt.hoistStruct(quant_dict, f'{nodeName}_quant_param', MemPoolStructDataTypes.ita_quant_t)]
        operatorRepresentation['quant_param'] = ctxt.lookup(f'{nodeName}_quant_param').name

        operatorRepresentation['q_offset'] = (q._signed == 0) * int(q.nLevels // 2)
        operatorRepresentation['k_offset'] = (k._signed == 0) * int(k.nLevels // 2)
        operatorRepresentation['output_offset'] = 0
        if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
            operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels // 2)

        # import IPython; IPython.embed()
        return ctxt, operatorRepresentation, nameList


class _MHSATemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:
        nameList = []

        ctxt, operatorRepresentation, wq_bias_ita, wk_bias_ita, wv_bias_ita, wo_bias_ita, wq_weight_ita, wk_weight_ita, wv_weight_ita, wo_weight_ita, q_ita, k_ita = _transformITAInputs(
            ctxt, operatorRepresentation)

        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        q = ctxt.lookup(operatorRepresentation['q'])
        k = ctxt.lookup(operatorRepresentation['k'])

        nodeName = operatorRepresentation['nodeName']
        N = operatorRepresentation['heads']

        data_in = N * [None]
        requant_mult = N * [None]
        requant_shift = N * [None]
        requant_add = N * [None]

        for h in range(N):
            # Create dummy array for key and values

            # Fuse all inputs together and store in L2
            data = np.concatenate([
                np.reshape(wo_weight_ita[h], -1),
                np.reshape(wv_weight_ita[h], -1),
                np.reshape(wk_weight_ita[h], -1),
                np.reshape(q_ita[h], -1),
                np.reshape(k_ita[h], -1),
                np.reshape(wq_weight_ita[h], -1),
                np.reshape(wo_bias_ita[h].astype(np.int32), (-1,)).view(np.int8),
                np.reshape(wv_bias_ita[h].astype(np.int32), (-1,)).view(np.int8),
                np.reshape(wk_bias_ita[h].astype(np.int32), (-1,)).view(np.int8),
                np.reshape(wq_bias_ita[h].astype(np.int32), (-1,)).view(np.int8),
            ])

            data_in[h] = ctxt.ConstantBuffer(name = f'{nodeName}_input_head{h}', shape = data.shape, values = data)
            ctxt.add(data_in[h], 'global')
            data_in[h]._type = PointerClass(int8_t)
            operatorRepresentation[f'data_in_head{h}'] = data_in[h].name
            nameList += [data_in[h].name]

            requant_mult_data = np.array([
                operatorRepresentation['wq_requant_mul'][h], operatorRepresentation['wk_requant_mul'][h],
                operatorRepresentation['preattn_requant_mul'][h], operatorRepresentation['wv_requant_mul'][h],
                operatorRepresentation['postattn_requant_mul'][h], operatorRepresentation['wo_requant_mul'][h], 0, 0
            ])
            requant_mult[h] = ctxt.ConstantBuffer(name = f'{nodeName}_requant_mult_head{h}',
                                                  shape = requant_mult_data.shape,
                                                  values = requant_mult_data)
            ctxt.add(requant_mult[h], 'global')
            requant_mult[h]._type = PointerClass(uint8_t)
            operatorRepresentation[f'requant_mult_head{h}'] = requant_mult[h].name
            nameList += [requant_mult[h].name]

            requant_shift_data = np.array([
                int(np.log2(operatorRepresentation['wq_requant_div'][h])),
                int(np.log2(operatorRepresentation['wk_requant_div'][h])),
                int(np.log2(operatorRepresentation['preattn_requant_div'][h])),
                int(np.log2(operatorRepresentation['wv_requant_div'][h])),
                int(np.log2(operatorRepresentation['postattn_requant_div'][h])),
                int(np.log2(operatorRepresentation['wo_requant_div'][h])), 0, 0
            ])
            requant_shift[h] = ctxt.ConstantBuffer(name = f'{nodeName}_requant_shift_head{h}',
                                                   shape = requant_shift_data.shape,
                                                   values = requant_shift_data)
            ctxt.add(requant_shift[h], 'global')
            requant_shift[h]._type = PointerClass(uint8_t)
            operatorRepresentation[f'requant_shift_head{h}'] = requant_shift[h].name
            nameList += [requant_shift[h].name]

            requant_add_data = np.array([
                operatorRepresentation['wq_requant_add'][h], operatorRepresentation['wk_requant_add'][h],
                operatorRepresentation['preattn_requant_add'][h], operatorRepresentation['wv_requant_add'][h],
                operatorRepresentation['postattn_requant_add'][h], operatorRepresentation['wo_requant_add'][h], 0, 0
            ])
            requant_add[h] = ctxt.ConstantBuffer(name = f'{nodeName}_requant_add_head{h}',
                                                 shape = requant_add_data.shape,
                                                 values = requant_add_data)
            ctxt.add(requant_add[h], 'global')
            requant_add[h]._type = PointerClass(int32_t)
            operatorRepresentation[f'requant_add_head{h}'] = requant_add[h].name
            nameList += [requant_add[h].name]

            quant_dict = {
                'eps_mult': operatorRepresentation[f'requant_mult_head{h}'],
                'right_shift': operatorRepresentation[f'requant_shift_head{h}'],
                'add': operatorRepresentation[f'requant_add_head{h}']
            }

            nameList += [
                ctxt.hoistStruct(quant_dict, f'{nodeName}_quant_params_head{h}', MemPoolStructDataTypes.ita_quant_t)
            ]
            operatorRepresentation[f'quant_params_head{h}'] = f'{nodeName}_quant_params_head{h}'

            operatorRepresentation['q_offset'] = (q._signed == 0) * int(q.nLevels // 2)
            operatorRepresentation['k_offset'] = (k._signed == 0) * int(k.nLevels // 2)
            operatorRepresentation['output_offset'] = 0
            if hasattr(data_out, "_signed") and hasattr(data_out, "nLevels"):
                operatorRepresentation['output_offset'] = -(data_out._signed == 0) * int(data_out.nLevels // 2)

        operatorRepresentation['data_in_array'] = ctxt._mangle(operatorRepresentation['nodeName'] + f"_data_in_array")
        operatorRepresentation['quant_params_array'] = ctxt._mangle(operatorRepresentation['nodeName'] +
                                                                    f"_quant_params_array")

        return ctxt, operatorRepresentation, nameList


MemPoolParallelTemplate_1H = _1HSATemplate("""
// ITA M1HSA (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);
M1HSA_s8_ITA(
    ${q}, ${k}, ${data_in},
    ${S}, ${E}, ${P},
    &${quant_param},
    ${data_out},
    ${q_offset}, ${k_offset}, ${output_offset},
    core_id,
    numThreads
);
mempool_barrier(numThreads);
""")

MemPoolParallelTemplate_2H = _MHSATemplate("""
// ITA M2HSA (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);

int8_t *${data_in_array}[] = { ${data_in_head0}, ${data_in_head1} };
ita_quant_t const *${quant_params_array}[] = { &${quant_params_head0}, &${quant_params_head1}};

M2HSA_s8_ITA(
    ${q}, ${k}, ${data_in_array},
    ${S}, ${E}, ${P},
    ${quant_params_array},
    ${data_out},
    ${q_offset}, ${k_offset}, ${output_offset},
    core_id,
    numThreads
);
mempool_barrier(numThreads);
""")

MemPoolParallelTemplate_4H = _MHSATemplate("""
// ITA M4HSA (Name: ${nodeName}, Op: ${nodeOp})
mempool_barrier(numThreads);

int8_t *${data_in_array}[] = { ${data_in_head0}, ${data_in_head1}, ${data_in_head2}, ${data_in_head3} };
ita_quant_t const *${quant_params_array}[] = { &${quant_params_head0}, &${quant_params_head1}, &${quant_params_head2}, &${quant_params_head3}};

M4HSA_s8_ITA(
    ${q}, ${k}, ${data_in_array},
    ${S}, ${E}, ${P},
    ${quant_params_array},
    ${data_out},
    ${q_offset}, ${k_offset}, ${output_offset},
    core_id,
    numThreads
);
mempool_barrier(numThreads);
""")
