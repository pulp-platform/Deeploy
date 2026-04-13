# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, \
    NodeTemplate, _NoVerbosity


class PULPMicrobenchmark(CodeTransformationPass):

    _preTemplate = NodeTemplate("""
    perf_stats_t ${op}_perf_start, ${op}_perf_end, ${op}_perf_total;
    if (pi_core_id() == 0) {
        perf_bench_init();
        perf_bench_start();
        perf_bench_read(&${op}_perf_start);
    }
    """)

    _postTemplate = NodeTemplate("""
    if (pi_core_id() == 0) {
        perf_bench_stop();
        perf_bench_read(&${op}_perf_end);
        perf_bench_diff(&${op}_perf_total, &${op}_perf_end, &${op}_perf_start);
        perf_bench_print("${op}", &${op}_perf_total);
    }
    """)

    def apply(self,
              ctxt: NetworkContext,
              executionBlock: ExecutionBlock,
              name: str,
              verbose: CodeGenVerbosity = _NoVerbosity) -> Tuple[NetworkContext, ExecutionBlock]:

        if not verbose.microbenchmarkProfiling:
            return ctxt, executionBlock

        executionBlock.addLeft(self._preTemplate, {"op": name})
        executionBlock.addRight(self._postTemplate, {"op": name})
        return ctxt, executionBlock
