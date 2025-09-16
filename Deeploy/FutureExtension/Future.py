# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
