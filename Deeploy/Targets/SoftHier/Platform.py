# ----------------------------------------------------------------------
#
# File: SoftHierPlatform.py
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author:
# - Bowen Wang <bowwang@iis.ee.ethz.ch>, ETH Zurich
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


import numpy as np

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Bindings import BasicAddBindings
from Deeploy.Targets.Generic.Layers import AddLayer
from Deeploy.Targets.Generic.Parsers import AddParser
from Deeploy.Targets.SoftHier.Templates import AllocateTemplate, FreeTemplate

# Basic bindings
Add_Mapper = NodeMapper(AddParser(), BasicAddBindings)

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
# DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

SoftHierlMapping = {'Add': AddLayer([Add_Mapper])}


class SoftHierVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.SoftHierInitTemplate
    allocTemplate = AllocateTemplate.SoftHierAllocateTemplate
    deallocTemplate = FreeTemplate.SoftHierLocalTemplate

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {
            "type": self._instance,
            "name": self.name,
            "size": int(np.prod(self.shape)),
            "_memoryLevel": memoryLevel
        }


class SoftHierTransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.SoftHierInitTemplate
    allocTemplate = AllocateTemplate.SoftHierAllocateTemplate
    deallocTemplate = FreeTemplate.SoftHierLocalTemplate

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {
            "type": self._instance,
            "name": self.name,
            "size": int(np.prod(self.shape)),
            "_memoryLevel": memoryLevel
        }


class SoftHierConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.SoftHierGlobalInitTemplate
    allocTemplate = AllocateTemplate.SoftHierGlobalAllocateTemplate
    deallocTemplate = FreeTemplate.SoftHierGlobalTemplate

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {
            "type": self._instance,
            "name": self.name,
            "size": int(np.prod(self.shape)),
            "_memoryLevel": memoryLevel
        }


class SoftHierStructBuffer(StructBuffer):

    initTemplate = AllocateTemplate.SoftHierStructInitTemplate
    allocTemplate = AllocateTemplate.SoftHierStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


SoftHierOptimizer = TopologyOptimizer([])
includeList = ["DeeployBasicMath.h", "flex_alloc_api.h"]


class SoftHierEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = SoftHierlMapping, initCode: str = "", includeList = includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class SoftHierPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [SoftHierEngine("SoftHier")],
                 variableBuffer = SoftHierVariableBuffer,
                 constantBuffer = SoftHierConstantBuffer,
                 structBuffer = SoftHierStructBuffer,
                 transientBuffer = SoftHierTransientBuffer):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
