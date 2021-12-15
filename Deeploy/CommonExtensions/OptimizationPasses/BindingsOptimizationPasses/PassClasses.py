# ----------------------------------------------------------------------
#
# File: PassClasses.py
#
# Last edited: 21.11.2023
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

import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import Pass, ReplaceSequentialPatternPass, SequentialPass
from Deeploy.DeeployTypes import NetworkContext


class BindingAwarePassMixIn():
    # DO NOT OVERWRITE this function in custom pass subclasses unless you have
    # a very good reason!
    def apply(self, ctxt, graph, layerBinding):
        ctxt, layerBinding = self.retarget(ctxt, graph, layerBinding)
        ctxt, layerBinding = self.run_pass(ctxt, graph, layerBinding)
        return ctxt, layerBinding

    def __call__(self, ctxt: NetworkContext, graph: gs.Graph, layerBinding):
        return self.apply(ctxt, graph, layerBinding)

    # overwrite this if your pass is specific to a graph instance (e.g., most
    # "dynamic" SequentialPass derivatives will be, as the list of passes to
    # execute probably depends on the graph. See e.g.
    # ReplaceSequentialPatternPass for an example)
    def retarget(self, ctxt: NetworkContext, graph: gs.Graph, layerBinding):
        return ctxt, layerBinding


class BindingAwareSequentialPassMixIn(BindingAwarePassMixIn):

    def run_pass(self, ctxt: NetworkContext, graph: gs.Graph, layerBinding):
        for p in self.named_subpasses().values():
            ctxt, layerBinding = p.apply(ctxt, graph, layerBinding)
        return ctxt, layerBinding


class BindingAwareReplaceSequentialPatternPassMixIn(BindingAwareSequentialPassMixIn):

    def retarget(self, ctxt: NetworkContext, graph: gs.Graph, layerBinding):
        # to retarget to a new graph, clear all registered subpasses.
        for k in self.named_subpasses().keys():
            self.remove_subpass(k)
        self.matches = self.matcher.match(graph, self.pattern)
        for i, m in enumerate(self.matches):
            ctxt, layerBinding = self.replacement_fn(ctxt, layerBinding, m, f"{self.name}_{i}", **self.kwargs)
        return ctxt, layerBinding


def bindingaware(cls):
    mixinClass = None
    # These need to be sorted from most specific parent class to least specific parent class!
    # if issubclass(cls, ReplaceMatchWithModulePass):
    #     mixinClass = BindingAwareReplaceMatchWithModulePassMixIn
    if issubclass(cls, ReplaceSequentialPatternPass):
        mixinClass = BindingAwareReplaceSequentialPatternPassMixIn
    elif issubclass(cls, SequentialPass):
        mixinClass = BindingAwareSequentialPassMixIn
    elif issubclass(cls, Pass):
        mixinClass = BindingAwarePassMixIn
    else:
        raise Exception(f"Tried to decorate class {cls} as bindingaware, but failed!")
    return type(cls.__name__, (cls, mixinClass), {})
