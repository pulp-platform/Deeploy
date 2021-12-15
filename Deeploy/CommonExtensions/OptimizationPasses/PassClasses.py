# ----------------------------------------------------------------------
#
# File: PassClasses.py
#
# Last edited: 28.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Author:
# Moritz Scherer, ETH Zurich
# Georg Rutishauser, ETH Zurich
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

from typing import List, Optional

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext

from .Matchers import Match, NonBranchingMatcher, SubgraphMatcher


class _MemoReach():

    def __init__(self, graph, inputTensors, outputTensors):
        self.memo = {}
        self.graph = graph
        self.inputTensors = inputTensors
        self.outputTensors = outputTensors

    def reachingSet(self):
        reachingSet = []
        for inTensor in self.inputTensors:
            for user in inTensor.outputs:
                reachingSet += self._reachingSet(user)

        nodeNames = {node.name for node in reachingSet}
        retList = [node for node in self.graph.nodes if node.name in nodeNames]
        return retList

    def _reachingSet(self, node: gs.Node) -> List[gs.Node]:

        if node.name in self.memo.keys():
            return self.memo[node.name]

        reachingSet = []

        if any([output in self.outputTensors for output in node.outputs]):
            self.memo[node.name] = [node]
            return [node]

        oSet = []
        for outp in node.outputs:
            for out in outp.outputs:
                if outp not in self.inputTensors:
                    oSet.append(out)

        for potentialNode in oSet:
            if potentialNode.name in self.memo.keys():
                nodeRet = self.memo[potentialNode.name]
            else:
                nodeRet = self._reachingSet(potentialNode)

            reachingSet += nodeRet

        if reachingSet != []:
            reachingSet.append(node)

        self.memo[node.name] = reachingSet

        return reachingSet


@gs.Graph.register()
def deleteNode(self, node: gs.Node):
    # LMACAN: Assume only one input and only one output tensor

    inputTensor = node.inputs[0]
    outputTensor = node.outputs[0]

    isGlobalOutputTensor = len(outputTensor.outputs) == 0

    if isGlobalOutputTensor:
        inputTensor.name = outputTensor.name  # Preserve the output tensor name
        outputTensor.name = outputTensor.name + "_throwaway"  # Avoid same named tensors in graph; gets immediately removed with cleanup
        self.outputs[self.outputs.index(outputTensor)] = inputTensor
    else:
        for outputNode in list(outputTensor.outputs):
            # Swap the outputTensor with inputTensor in the downstream nodes
            outputNode.inputs[outputNode.inputs.index(outputTensor)] = inputTensor
        node.inputs.clear()
        node.outputs.clear()

    self.cleanup()


def _reachableNodes(graph: gs.Graph, inputTensors: List[gs.Tensor], outputTensors: List[gs.Tensor]) -> List[gs.Node]:

    _inputTensors = [tensor for tensor in inputTensors.copy() if tensor.name in graph.tensors().keys()]
    _outputTensors = [tensor for tensor in outputTensors.copy() if tensor.name in graph.tensors().keys()]

    retList = _MemoReach(graph, _inputTensors, _outputTensors).reachingSet()

    return retList


@gs.Graph.register()
def replaceInsertNode(self, inputs, outputs, newNode):
    reachableSet = _reachableNodes(self, inputs, outputs)

    ret = self.layer(op = newNode.op, name = newNode.name, attrs = newNode.attrs, inputs = inputs, outputs = outputs)

    for node in reachableSet:
        node.outputs = []

    self.toposort().cleanup()


class Pass():

    def __init__(self):
        self.parent = None
        self._subpasses = {}

    def __setattr__(self, attribute, value):
        if isinstance(value, Pass) and attribute != 'parent':
            self.register_subpass(attribute, value)
        super(Pass, self).__setattr__(attribute, value)

    def register_subpass(self, name, value):
        if name in self._subpasses.keys():
            del self._subpasses[name]

        value.parent = self
        self._subpasses[name] = value

    def remove_subpass(self, name):
        try:
            del self._subpasses[name]
        except KeyError:
            print(f"No subpass with name {name}, cannot remove!")
        except AttributeError:
            raise AttributeError("Cannot remove sub-pass before calling Pass.__init__!")

    def __getattr__(self, attribute):
        if self._subpasses is not None and attribute in self._subpasses.keys():
            return self._subpasses[attribute]

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attribute}")

    def named_subpasses(self):
        return self._subpasses.copy()


class ContextAwarePassMixIn():
    # DO NOT OVERWRITE this function in custom pass subclasses unless you have
    # a very good reason!
    def apply(self, ctxt, graph):
        ctxt, graph = self.retarget(ctxt, graph)
        ctxt, graph = self.run_pass(ctxt, graph)
        return ctxt, graph

    def __call__(self, ctxt: NetworkContext, graph: gs.Graph):
        return self.apply(ctxt, graph)

    # overwrite this if your pass is specific to a graph instance (e.g., most
    # "dynamic" SequentialPass derivatives will be, as the list of passes to
    # execute probably depends on the graph. See e.g.
    # ReplaceSequentialPatternPass for an example)
    def retarget(self, ctxt: NetworkContext, graph: gs.Graph):
        return ctxt, graph


class ContextAgnosticPassMixIn():
    # DO NOT OVERWRITE this function in custom pass subclasses unless you have
    # a very good reason!
    def apply(self, graph: gs.Graph) -> gs.Graph:
        graph = self.retarget(graph)
        graph = self.run_pass(graph)
        return graph

    def __call__(self, graph: gs.Graph):
        return self.apply(graph)

    # overwrite this if your pass is specific to a graph instance (e.g., most
    # "dynamic" SequentialPass derivatives will be, as the list of passes to
    # execute probably depends on the graph. See e.g.
    # ReplaceSequentialPatternPass for an example)
    def retarget(self, graph: gs.Graph) -> gs.Graph:
        return graph


class ContextAwareSequentialPassMixIn(ContextAwarePassMixIn):

    def run_pass(self, ctxt: NetworkContext, graph: gs.Graph):
        for p in self.named_subpasses().values():
            ctxt, graph = p.apply(ctxt, graph)
        return ctxt, graph


class ContextAgnosticSequentialPassMixIn(ContextAgnosticPassMixIn):

    def run_pass(self, graph: gs.Graph):
        for p in self.named_subpasses().values():
            graph = p.apply(graph)
        return graph


class SequentialPass(Pass):

    def __init__(self, *passes, name_prefix = ''):
        super(SequentialPass, self).__init__()
        self.name_prefix = name_prefix
        self.setup_passes(passes)

    def setup_passes(self, passes):
        for i, p in enumerate(passes):
            self.register_subpass(self.name_prefix + '_' + str(i), p)


class ContextAwareReplaceMatchWithModulePassMixIn(ContextAwarePassMixIn):

    def run_pass(self, ctxt: NetworkContext, graph: gs.Graph):
        if self.replacementNode is not None:
            graph.replaceInsertNode(self.replacementNode)
        return ctxt, graph


class ContextAgnosticReplaceMatchWithModulePassMixIn(ContextAgnosticPassMixIn):

    def run_pass(self, graph: gs.Graph) -> gs.Graph:
        if self.replacementNode is not None:
            graph.replaceInsertNode(self.replacementNode)
        return graph


class ReplaceMatchWithModulePass(Pass):
    #Matches are specific to graph instances, so don't use this type of pass on its
    #own if you want to reuse it!
    def __init__(self, match: Match, module: gs.Node):
        # this class needs a name field because the inserted submodules will be named
        super(ReplaceMatchWithModulePass, self).__init__()
        self.match = match
        self.replacementNode = module


class ContextAwareReplaceSequentialPatternPassMixIn(ContextAwareSequentialPassMixIn):

    def retarget(self, ctxt: NetworkContext, graph: gs.Graph):
        # to retarget to a new graph, clear all registered subpasses.
        for k in self.named_subpasses().keys():
            self.remove_subpass(k)
        self.matches = self.matcher.match(graph, self.pattern)
        for i, m in enumerate(self.matches):
            ctxt, graph = self.replacement_fn(ctxt, graph, m, f"{self.name}_{i}", **self.kwargs)
        graph.cleanup().toposort()
        return ctxt, graph


class ContextAgnosticReplaceSequentialPatternPassMixIn(ContextAgnosticSequentialPassMixIn):

    def retarget(self, graph: gs.Graph):
        # to retarget to a new graph, clear all registered subpasses.
        for k in self.named_subpasses().keys():
            self.remove_subpass(k)
        self.matches = self.matcher.match(graph, self.pattern)
        for i, m in enumerate(self.matches):
            graph = self.replacement_fn(graph, m, f"{self.name}_{i}", **self.kwargs)
        graph.cleanup().toposort()
        return graph


class ReplaceSequentialPatternPass(SequentialPass):
    # finds all instances of pattern in the graph, calls the replacement_fn on
    # the matches and replaces the matched nodes with the module returned by
    # replacement_fn.
    def __init__(self,
                 pattern: gs.Graph,
                 replacement_fn: callable,
                 name: str,
                 matcher: Optional[SubgraphMatcher] = None,
                 **kwargs):
        super().__init__(name_prefix = name)
        self.pattern = pattern
        self.matcher = matcher
        if matcher is None:
            self.matcher = NonBranchingMatcher()
        self.replacement_fn = replacement_fn
        self.name = name
        self.kwargs = kwargs


def contextagnostic(cls):
    mixinClass = None
    # These need to be sorted from most specific parent class to least specific parent class!
    if issubclass(cls, ReplaceMatchWithModulePass):
        mixinClass = ContextAgnosticReplaceMatchWithModulePassMixIn
    elif issubclass(cls, ReplaceSequentialPatternPass):
        mixinClass = ContextAgnosticReplaceSequentialPatternPassMixIn
    elif issubclass(cls, SequentialPass):
        mixinClass = ContextAgnosticSequentialPassMixIn
    elif issubclass(cls, Pass):
        mixinClass = ContextAgnosticPassMixIn
    else:
        raise Exception(f"Tried to decorate class {cls} as contextagnostic, but failed!")
    return type(cls.__name__, (cls, mixinClass), {})


def contextaware(cls):
    mixinClass = None
    # These need to be sorted from most specific parent class to least specific parent class!
    if issubclass(cls, ReplaceMatchWithModulePass):
        mixinClass = ContextAwareReplaceMatchWithModulePassMixIn
    elif issubclass(cls, ReplaceSequentialPatternPass):
        mixinClass = ContextAwareReplaceSequentialPatternPassMixIn
    elif issubclass(cls, SequentialPass):
        mixinClass = ContextAwareSequentialPassMixIn
    elif issubclass(cls, Pass):
        mixinClass = ContextAwarePassMixIn
    else:
        raise Exception(f"Tried to decorate class {cls} as contextaware, but failed!")
    return type(cls.__name__, (cls, mixinClass), {})
