# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity


class ProfilingCodeGeneration(CodeTransformationPass):
    """
    Code transformation pass for inserting cycle measurement profiling code.

    This class extends CodeTransformationPass to automatically insert profiling
    code around execution blocks. It adds cycle counting instrumentation before
    and after the target code, enabling performance measurement and analysis
    of individual operations during runtime.

    The generated profiling code uses a `getCycles()` function to measure
    execution time and prints the results to stdout. This is useful for
    performance analysis, optimization, and debugging of neural network
    operations.

    Notes
    -----
    This transformation requires that the target platform provides a
    `getCycles()` function that returns the current cycle count as a uint32_t.
    The transformation also assumes printf functionality is available for
    output formatting.

    The profiling code is non-intrusive and can be easily enabled or disabled
    by including or excluding this transformation pass from the compilation
    pipeline.
    """

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:
        """
        Apply cycle measurement profiling to an execution block.

        Wraps the given execution block with cycle counting code that measures
        and reports the execution time. The profiling code is added before
        (left) and after (right) the original execution block.

        Parameters
        ----------
        ctxt : NetworkContext
            The network context for code generation. This parameter is passed
            through unchanged as cycle measurement doesn't modify the context.
        executionBlock : ExecutionBlock
            The execution block to instrument with cycle measurement code.
            The original block remains unchanged, with profiling code added
            around it.
        name : str
            The name of the operation being profiled. This name is used to
            generate unique variable names and is included in the output
            message for identification.
        verbose : CodeGenVerbosity, optional
            The verbosity level for code generation. Default is _NoVerbosity.
            This parameter is not used by the cycle measurement transformation.

        Returns
        -------
        Tuple[NetworkContext, ExecutionBlock]
            A tuple containing:
            - The unchanged network context
            - The modified execution block with profiling code added
        """
        executionBlock.addLeft(NodeTemplate("""
        uint32_t ${op}_cycles = getCycles();
        """), {"op": name})
        executionBlock.addRight(
            NodeTemplate("""
        uint32_t ${op}_endCycles = getCycles();
        printf("${op} took %u cycles \\n", ${op}_endCycles - ${op}_cycles);
        """), {"op": name})
        return ctxt, executionBlock
