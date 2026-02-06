# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Dict, Literal, NamedTuple, Optional

import onnx_graphsurgeon as gs


class Match(NamedTuple):
    """
    Represents a successful pattern match in a computational graph.

    This named tuple encapsulates the result of matching a pattern graph
    against a larger computational graph. It contains both the anchor node
    (starting point of the match) and a complete mapping between pattern
    nodes and their corresponding matched nodes in the target graph.

    Attributes
    ----------
    anchor : gs.Node
        The node in the target graph that serves as the starting point
        for the pattern match. This is typically the first node that
        matched the pattern and from which the full match was discovered.
    nodes_map : Dict[str, gs.Node]
        A dictionary mapping pattern node names to their corresponding
        matched nodes in the target graph. The keys are pattern node names
        and the values are the actual matched nodes from the target graph.

    Notes
    -----
    This class is used by pattern matching algorithms to represent successful
    matches. The nodes_map provides a complete correspondence between the
    pattern structure and the matched subgraph, enabling transformations
    and optimizations to be applied to the matched regions.
    """
    anchor: gs.Node
    nodes_map: Dict[str, gs.Node]


class SubgraphMatcher:
    """
    Base class for pattern matching in computational graphs.

    This class provides the foundation for matching pattern graphs against
    larger computational graphs. It supports both exact string matching and
    regular expression matching for operation types, enabling flexible
    pattern recognition for graph optimization and transformation.

    The matcher identifies non-overlapping instances of a pattern within
    a target graph, returning Match objects that can be used for subsequent
    transformations or analysis.

    Notes
    -----
    This is an abstract base class that defines the interface for pattern
    matching. Concrete implementations must override the abstract methods
    `_valid_pattern` and `_nodes_map_from_anchor` to define specific
    matching algorithms.

    The matching process ensures non-overlapping matches, meaning each
    node in the target graph can only participate in at most one match.
    """

    def __init__(self, regex_op: bool = False):
        """
        Initialize the SubgraphMatcher.

        Parameters
        ----------
        regex_op : bool, optional
            Whether to use regular expression matching for operation types.
            Default is False for exact string matching.
        """
        # operation matching policy
        self.regex_op = regex_op

    def is_op_match(self, patternNode: gs.Node, graphNode: gs.Node):
        """
        Check if a pattern node operation matches a graph node operation.

        Compares the operation types of two nodes according to the configured
        matching policy (regex or exact match).

        Parameters
        ----------
        patternNode : gs.Node
            The pattern node whose operation type serves as the match criterion.
        graphNode : gs.Node
            The target graph node to check for a match.

        Returns
        -------
        bool
            True if the operations match according to the configured policy,
            False otherwise.

        Notes
        -----
        When regex_op is True, the pattern node's operation is treated as a
        regular expression pattern and matched against the graph node's
        operation using `re.fullmatch`. When False, exact string equality
        is used.
        """
        if self.regex_op:
            return re.fullmatch(patternNode.op, graphNode.op) is not None
        else:
            return patternNode.op == graphNode.op

    # Override this
    def _valid_pattern(self, pattern: gs.Graph) -> None:
        """
        Validate that a pattern graph meets the requirements for matching.

        This abstract method should be overridden by subclasses to implement
        pattern validation logic specific to their matching algorithm.

        Parameters
        ----------
        pattern : gs.Graph
            The pattern graph to validate.

        Raises
        ------
        AssertionError
            If the pattern does not meet the required constraints.

        Notes
        -----
        This method is called before attempting to match a pattern and should
        verify that the pattern has the correct structure for the specific
        matching algorithm being used.
        """
        _ = pattern

    # Override this
    def _nodes_map_from_anchor(self, anchor: gs.Node, pattern: gs.Graph) -> Optional[Dict[str, gs.Node]]:
        """
        Attempt to match a pattern starting from an anchor node.

        This abstract method should be overridden by subclasses to implement
        the core matching logic specific to their algorithm.

        Parameters
        ----------
        anchor : gs.Node
            The potential starting node for matching the pattern.
        pattern : gs.Graph
            The pattern graph to match.

        Returns
        -------
        Optional[Dict[str, gs.Node]]
            A dictionary mapping pattern node names to matched graph nodes
            if the pattern matches starting from the anchor, None otherwise.

        Notes
        -----
        This method contains the core pattern matching algorithm and should
        return a complete mapping from pattern nodes to graph nodes if a
        valid match is found.
        """
        _, _ = anchor, pattern

    def _match_from_anchor(self, anchor: gs.Node, pattern: gs.Graph) -> Optional[Match]:
        """
        Attempt to create a complete match starting from an anchor node.

        Uses the subclass-specific matching algorithm to find a node mapping
        and validates that the mapping covers all nodes in the pattern.

        Parameters
        ----------
        anchor : gs.Node
            The potential starting node for pattern matching.
        pattern : gs.Graph
            The pattern graph to match against.

        Returns
        -------
        Optional[Match]
            A Match object containing the anchor and complete node mapping
            if successful, None if the pattern doesn't match from this anchor.

        Notes
        -----
        This method ensures that a valid match covers all nodes in the
        pattern graph before considering it successful.
        """
        nodes_map = self._nodes_map_from_anchor(anchor, pattern)

        if nodes_map is not None and len(nodes_map.keys()) == len(pattern.nodes):
            return Match(anchor, nodes_map)
        else:
            return None

    def match(self, graph: gs.Graph, pattern: gs.Graph):
        """
        Find all non-overlapping matches of a pattern in a target graph.

        Systematically searches the target graph for instances of the pattern,
        ensuring that each node participates in at most one match to avoid
        conflicts during transformations.

        Parameters
        ----------
        graph : gs.Graph
            The target graph to search for pattern matches.
        pattern : gs.Graph
            The pattern graph to find instances of.

        Returns
        -------
        List[Match]
            A list of Match objects representing all non-overlapping instances
            of the pattern found in the target graph.

        Notes
        -----
        The algorithm:
        1. Validates the pattern using the subclass-specific validation
        2. Iterates through all nodes in the target graph as potential anchors
        3. Attempts to match the pattern from each anchor
        4. Collects only non-overlapping matches to avoid conflicts

        Non-overlapping means that if a node is part of one match, it cannot
        be part of any other match in the returned list.
        """
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
    """
    Pattern matcher for sequential computational graphs without branching.

    This matcher is optimized for patterns that form a simple chain of operations
    without splits or merges in the computational flow. It uses a recursive
    algorithm to follow the linear path of operations.

    The matching algorithm follows edges from the anchor node to build a complete
    mapping between pattern nodes and graph nodes, verifying operation types
    and attributes at each step.
    Notes
    -----
    This matcher is efficient for linear operation sequences such as:
    - Conv -> BatchNorm -> ReLU chains
    - Linear -> Dropout -> Activation sequences
    - Simple preprocessing pipelines

    The algorithm assumes that each node in the pattern has at most one
    output connection to the next node in the sequence.
    """

    # simplified matcher which matches call_module ops more reasonably
    def __init__(self, regex_op: bool = False):
        """
        Initialize the non-branching pattern matcher.

        Parameters
        ----------
        regex_op : bool, optional
            Enable regex-based operation type matching. Default is False.
        """
        # This checking is sufficient - iff the graph is acyclic and connected (checked by parser)
        # and every node has one output, the graph is sequential
        super().__init__(regex_op)

    def _valid_pattern(self, pattern: gs.Graph):
        """
        Validate that the pattern is suitable for non-branching matching.

        Ensures that the pattern graph forms a simple sequential chain
        without branching or multiple outputs.

        Parameters
        ----------
        pattern : gs.Graph
            The pattern graph to validate.

        Raises
        ------
        AssertionError
            If the pattern has more than one output or any node has
            multiple outputs (indicating branching).

        Notes
        -----
        Valid patterns for non-branching matching must satisfy:
        1. Exactly one graph output
        2. Each node has exactly one output (no branching)
        3. Forms a simple chain of operations
        """
        assert len(pattern.outputs) == 1, "Found more than one output"
        for node in pattern.nodes:
            assert len(node.outputs) == 1, "Graph needs to be purely sequential!"

    def _match_nodes_recursive(self, pn: gs.Node, gn: gs.Node, pattern_length: int,
                               nodes_map: dict) -> Optional[Dict[str, gs.Node]]:
        """
        Recursively match nodes in a sequential pattern.

        Follows the linear chain of operations from the current nodes,
        building a complete mapping between pattern and graph nodes.

        Parameters
        ----------
        pn : gs.Node
            Current node in the pattern graph.
        gn : gs.Node
            Current node in the target graph.
        pattern_length : int
            Total number of nodes in the pattern (for termination).
        nodes_map : dict
            Accumulated mapping from pattern node names to graph nodes.

        Returns
        -------
        Optional[Dict[str, gs.Node]]
            Complete node mapping if the pattern matches from this point,
            None if the pattern doesn't match.

        Notes
        -----
        The algorithm:
        1. Verifies that current nodes are compatible (type and attributes)
        2. Adds the current mapping to nodes_map
        3. If pattern is complete, returns the mapping
        4. Otherwise, recursively matches the next nodes in the sequence

        This simplified approach works because we've already validated
        that the pattern is purely sequential.
        """

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
        """
        Create a complete node mapping starting from an anchor node.

        Initiates the recursive matching process from the first node in the
        pattern and the provided anchor node in the target graph.

        Parameters
        ----------
        anchor : gs.Node
            The starting node in the target graph.
        pattern : gs.Graph
            The pattern graph to match.

        Returns
        -------
        Optional[Dict[str, gs.Node]]
            A complete mapping from pattern node names to graph nodes
            if the pattern matches starting from the anchor, None otherwise.

        Notes
        -----
        This method selects the first node from the pattern as the pattern
        anchor and delegates to the recursive matching algorithm. For
        sequential patterns, the choice of pattern anchor doesn't affect
        the result since there's only one valid traversal order.
        """
        pattern_anchor = next(iter(pattern.nodes))
        return self._match_nodes_recursive(pattern_anchor, anchor, len(pattern.nodes), {})


class BranchingMatcher(SubgraphMatcher):
    """
    Pattern matcher for computational graphs with branching and merging.

    This matcher handles complex patterns that contain splits, merges, and
    other non-sequential structures. It uses a more sophisticated
    algorithm that can traverse graphs in both forward and reverse directions
    to handle branching patterns.

    The matching algorithm explores multiple paths through the graph and
    can handle patterns with:
    - Multiple inputs/outputs per node
    - Fan-out (one node feeding multiple nodes)
    - Fan-in (multiple nodes feeding one node)
    - Complex DAG structures

    Parameters
    ----------
    regex_op : bool, optional
        If True, enables regex-based operation type matching instead of
        exact string matching. Default is False.

    Notes
    -----
    This matcher is suitable for complex patterns such as:
    - ResNet skip connections
    - Attention mechanisms with multiple branches
    - Feature pyramid networks
    - Any pattern with non-linear control flow

    The algorithm is more computationally intensive than NonBranchingMatcher
    but provides full generality for arbitrary DAG patterns.
    """

    # simplified matcher which matches call_module ops more reasonably
    def __init__(self, regex_op: bool = False):
        """
        Initialize the branching pattern matcher.

        Parameters
        ----------
        regex_op : bool, optional
            Enable regex-based operation type matching. Default is False.
        """
        super().__init__(regex_op)

    def _valid_pattern(self, pattern: gs.Graph):
        """
        Validate that the pattern is suitable for branching matching.

        Ensures that the pattern has exactly one output, but allows for
        complex internal structure with branching and merging.

        Parameters
        ----------
        pattern : gs.Graph
            The pattern graph to validate.

        Raises
        ------
        AssertionError
            If the pattern has more than one output.

        Notes
        -----
        Unlike NonBranchingMatcher, this validator only checks for a single
        output but allows arbitrary internal complexity including:
        - Nodes with multiple inputs (fan-in)
        - Nodes with multiple outputs (fan-out)
        - Complex DAG structures
        """
        assert len(pattern.outputs) == 1, "Found more than one output"

    def _match_nodes_recursive(self, pn: gs.Node, gn: gs.Node, nodes_map: dict,
                               direction: Literal["Forward", "Reverse"]) -> Optional[Dict]:
        """
        Recursively match nodes in a branching pattern.

        Explores the graph in the specified direction, handling both forward
        traversal (following outputs) and reverse traversal (following inputs)
        to match complex branching patterns.

        Parameters
        ----------
        pn : gs.Node
            Current node in the pattern graph.
        gn : gs.Node
            Current node in the target graph.
        nodes_map : dict
            Accumulated mapping from pattern node names to graph nodes.
        direction : Literal["Forward", "Reverse"]
            Direction of graph traversal - "Forward" follows outputs,
            "Reverse" follows inputs.

        Returns
        -------
        Optional[Dict]
            Updated node mapping if the pattern continues to match,
            None if the pattern doesn't match from this point.

        Raises
        ------
        AssertionError
            If direction is not "Forward" or "Reverse".

        Notes
        -----
        The algorithm:
        1. Validates that current nodes are compatible
        2. Adds the current mapping if not already present
        3. Recursively explores neighbors in the specified direction
        4. Handles both fan-out and fan-in scenarios
        5. Terminates when all pattern nodes are matched
        """
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
        """
        Create a complete node mapping starting from an anchor node.

        Initiates the recursive branching matching process from the first node
        in the pattern and the provided anchor node in the target graph.

        Parameters
        ----------
        anchor : gs.Node
            The starting node in the target graph.
        pattern : gs.Graph
            The pattern graph to match.

        Returns
        -------
        Optional[Dict[str, gs.Node]]
            A complete mapping from pattern node names to graph nodes
            if the pattern matches starting from the anchor, None otherwise.

        Notes
        -----
        This method:
        1. Selects the first node from the pattern as the pattern anchor
        2. Initiates forward traversal from the anchor nodes
        3. Uses the full branching matching algorithm to handle complex patterns

        The forward direction is used initially, but the recursive algorithm
        may switch to reverse direction as needed to properly explore
        branching structures.
        """
        pattern_anchor = next(iter(pattern.nodes))
        return self._match_nodes_recursive(pattern_anchor, anchor, {}, 'Forward')
