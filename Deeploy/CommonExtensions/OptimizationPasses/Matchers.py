# ----------------------------------------------------------------------
#
# File: Matchers.py
#
# Last edited: 28.04.2023
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

import re
from typing import Dict, Literal, NamedTuple, Optional

import onnx_graphsurgeon as gs


class Match(NamedTuple):
    anchor: gs.Node
    nodes_map: Dict[str, gs.Node]


class SubgraphMatcher:

    def __init__(self, regex_op: bool = False):
        # operation matching policy
        self.regex_op = regex_op

    def is_op_match(self, patternNode: gs.Node, graphNode: gs.Node):
        if self.regex_op:
            return re.fullmatch(patternNode.op, graphNode.op) is not None
        else:
            return patternNode.op == graphNode.op

    # Override this
    def _valid_pattern(self, pattern: gs.Graph) -> None:
        _ = pattern

    # Override this
    def _nodes_map_from_anchor(self, anchor: gs.Node, pattern: gs.Graph) -> Optional[Dict[str, gs.Node]]:
        _, _ = anchor, pattern

    def _match_from_anchor(self, anchor: gs.Node, pattern: gs.Graph) -> Optional[Match]:
        nodes_map = self._nodes_map_from_anchor(anchor, pattern)

        if nodes_map is not None and len(nodes_map.keys()) == len(pattern.nodes):
            return Match(anchor, nodes_map)
        else:
            return None

    def match(self, graph: gs.Graph, pattern: gs.Graph):
        self._valid_pattern(pattern)

        # Return a list of non-overlapping matches of pattern
        # self.pattern in the graph.
        matches = []
        # Nodes are not hashable so we are using their names
        matched_node_names = set()

        def node_names(match: Match):
            return [node.name for node in match.nodes_map.values()]

        def is_overlap(match: Match):
            return not matched_node_names.isdisjoint(node_names(match))

        for node in graph.nodes:
            match = self._match_from_anchor(node, pattern)
            if match is not None and not is_overlap(match):
                matches.append(match)
                matched_node_names.update(node_names(match))
        return matches


class NonBranchingMatcher(SubgraphMatcher):
    # simplified matcher which matches call_module ops more reasonably
    def __init__(self, regex_op: bool = False):
        # This checking is sufficient - iff the graph is acyclic and connected (checked by parser)
        # and every node has one output, the graph is sequential
        super().__init__(regex_op)

    def _valid_pattern(self, pattern: gs.Graph):
        assert len(pattern.outputs) == 1, "Found more than one output"
        for node in pattern.nodes:
            assert len(node.outputs) == 1, "Graph needs to be purely sequential!"

    def _match_nodes_recursive(self, pn: gs.Node, gn: gs.Node, pattern_length: int,
                               nodes_map: dict) -> Optional[Dict[str, gs.Node]]:
        # as we do sequential traversal, the first step (checking if nodes
        # already traversed) of the original _match_nodes function can be
        # discarded

        # the following submethod is a modified version of the one from the
        # original SubgraphMatcher
        def attributes_are_equal(pn: gs.Node, gn: gs.Node) -> bool:
            return self.is_op_match(pn, gn)

        # from here on, proceed as in the original implementation.
        if not attributes_are_equal(pn, gn):
            return None

        # Graph has a branch
        if len(gn.outputs) > 1:
            return None

        nodes_map[pn.name] = gn

        # End of pattern
        if pattern_length == 1:
            return nodes_map

        # if we are in the "active" pattern, the graph node has to be
        # single-output and single-use
        # if (pn.op not in ("output", "placeholder") and
        # (len(gn.all_input_nodes) != 1) or (len(gn.users) > 1 and not
        # first_active_node)):
        if len(gn.outputs[0].outputs) > 1:
            # if the gn has >1 users, the pattern is "leaking" and we don't
            # want to match it
            return None

        # otherwise we are on a "matching track", so move one node down in
        # pattern and graph. We know that gn has only 1 input!
        if len(pn.outputs[0].outputs) < 1 or len(gn.outputs[0].outputs) < 1:
            return None

        return self._match_nodes_recursive(pn.o(), gn.o(), pattern_length - 1, nodes_map)

    def _nodes_map_from_anchor(self, anchor: gs.Node, pattern: gs.Graph) -> Optional[Dict[str, gs.Node]]:
        pattern_anchor = next(iter(pattern.nodes))
        return self._match_nodes_recursive(pattern_anchor, anchor, len(pattern.nodes), {})


class BranchingMatcher(SubgraphMatcher):
    # simplified matcher which matches call_module ops more reasonably
    def __init__(self, regex_op: bool = False):
        super().__init__(regex_op)

    def _valid_pattern(self, pattern: gs.Graph):
        assert len(pattern.outputs) == 1, "Found more than one output"

    def _match_nodes_recursive(self, pn: gs.Node, gn: gs.Node, nodes_map: dict,
                               direction: Literal["Forward", "Reverse"]) -> Optional[Dict]:
        assert direction in ["Forward", "Reverse"], f"'{direction}' is not a valid matching direction!"

        # Check if nodes are identical
        def attributes_are_equal(pn: gs.Node, gn: gs.Node):
            ret = True
            pn_inputs = [pn_input for pn_input in pn.inputs if len(pn_input.inputs) > 0]
            gn_inputs = [gn_input for gn_input in gn.inputs if len(gn_input.inputs) > 0]
            if len(pn.inputs) != 1 or len(pn.inputs[0].inputs) != 0:
                ret &= (len(pn_inputs) == len(gn_inputs))

            pn_outputs = [pn_output for pn_output in pn.outputs if len(pn_output.outputs) > 0]
            gn_outputs = [gn_output for gn_output in gn.outputs if len(gn_output.outputs) > 0]
            if len(pn.outputs) != 1 or len(pn.outputs[0].outputs) != 0:
                ret &= (len(pn_outputs) == len(gn_outputs))

            ret &= self.is_op_match(pn, gn)
            return ret, (pn_inputs, gn_inputs, pn_outputs, gn_outputs)

        ret, (pn_inputs, gn_inputs, pn_outputs, gn_outputs) = attributes_are_equal(pn, gn)
        if not ret:
            return None

        # Add candidate match to list
        nodes_map[pn.name] = gn

        # Check if we are done
        if direction == 'Reverse':
            # If our pattern is fully matched until here
            if len(pn.inputs) == 0:
                return nodes_map
            # If our pattern is not fully matched completely but we reached leaf node
            if len(pn.inputs) > 0 and len(gn.inputs) == 0:
                return None

        if direction == 'Forward':
            # If our pattern is fully matched until here
            if len(pn.outputs) == 0:
                return nodes_map
            # If our pattern is not fully matched completely but we reached leaf node
            if len(pn.outputs) > 0 and len(gn.outputs) == 0:
                return None

        # If pn and gn have multiple parent nodes, we have want to traverse upwards
        if len(pn.inputs) > 0 and len(gn.inputs) > 0:
            for pn_input in pn.inputs:
                # Check if parent node of pn is constant or input node (in this case it has no additional inputs)
                # and if node was already matched
                if len(pn_input.inputs) > 0 and pn_input.inputs[0].name not in nodes_map.keys():
                    tmp = None
                    for gn_input in gn.inputs:
                        # Check if parent node of gn is constant or input node (in this case it has no additional inputs)
                        # and if node was already matched
                        if len(gn_input.inputs) > 0 and gn_input.inputs[0] not in nodes_map.values():
                            # Search for valid subgraphs
                            tmp = self._match_nodes_recursive(pn_input.inputs[0],
                                                              gn_input.inputs[0],
                                                              nodes_map,
                                                              direction = 'Reverse')
                            if tmp is not None:
                                nodes_map = tmp

                    # If it was not possible to map parent node of pn to a parent node of gn
                    if tmp == None:
                        return None

            # If it was possible to map all parent nodes of pn to a parent node of gn
            if direction == 'Reverse':
                return nodes_map

        # If pn and gn have multiple child nodes, we have want to traverse downwards
        if len(pn.outputs) > 0 and len(gn.outputs) > 0:
            for pn_input in pn.outputs:
                # Check if parent node of pn is is output node (in this case it has no additional outputs)
                # and if node was already matched
                if len(pn_input.outputs) > 0 and pn_input.outputs[0].name not in nodes_map.keys():
                    tmp = None
                    for gn_input in gn.outputs:
                        # Check if parent node of gn is is output node (in this case it has no additional outputs)
                        # and if node was already matched
                        if len(gn_input.outputs) > 0 and gn_input.outputs[0] not in nodes_map.values():
                            # Search for valid subgraphs
                            tmp = self._match_nodes_recursive(pn_input.outputs[0],
                                                              gn_input.outputs[0],
                                                              nodes_map,
                                                              direction = 'Forward')
                            if tmp is not None:
                                nodes_map = tmp

                    # If it was not possible to map parent node of pn to a parent node of gn
                    if tmp == None:
                        return None

            # If it was possible to map all child nodea of pn to a child node of gn
            if direction == 'Forward':
                return nodes_map

        assert False, "This statement should never be reached!"

    def _nodes_map_from_anchor(self, anchor: gs.Node, pattern: gs.Graph) -> Optional[Dict[str, gs.Node]]:
        pattern_anchor = next(iter(pattern.nodes))
        return self._match_nodes_recursive(pattern_anchor, anchor, {}, 'Forward')
