# ----------------------------------------------------------------------
#
# File: Future.py
#
# Last edited: 07.06.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
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

from typing import Optional, Type

from Deeploy.AbstractDataTypes import BaseType, Pointer
from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, StructBuffer


class Future(Pointer):

    __slots__ = ['stateReference']
    stateReferenceType: Type[Pointer]
    resolveCheckTemplate: NodeTemplate
    dispatchCheckTemplate: NodeTemplate

    def assignStateReference(self, stateReference: StructBuffer, ctxt: Optional[NetworkContext] = None):
        if self.stateReferenceType.checkPromotion(stateReference.structDict, ctxt):  # type: ignore
            self.stateReference = stateReference
        else:
            raise Exception(f"Can't assign {stateReference} to {self}!")

    def _bufferRepresentation(self):
        return {"stateReference": self.stateReference.name}


def FutureClass(underlyingType: BaseType, stateReferenceType: Type[Pointer], resolveCheckTemplate: NodeTemplate,
                dispatchCheckTemplate: NodeTemplate) -> Type[Future]:

    typeName = stateReferenceType.typeName + "Future"
    if typeName not in globals().keys():
        retCls = type(
            typeName, (Future,), {
                "typeName": underlyingType.typeName + "*",
                "typeWidth": 32,
                "referencedType": underlyingType,
                "stateReferenceType": stateReferenceType,
                "resolveCheckTemplate": resolveCheckTemplate,
                "dispatchCheckTemplate": dispatchCheckTemplate
            })
        globals()[typeName] = retCls
    else:
        retCls = globals()[typeName]

    return retCls
