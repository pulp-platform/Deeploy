from Deeploy.DeeployTypes import CodeGenVerbosity, CodeTransformationPass, ExecutionBlock, NetworkContext, NodeTemplate, CodeSnippet, _NoVerbosity


class SpatzBenchmarkInnerPass(CodeTransformationPass):
    def apply(self, ctxt: NetworkContext, executionBlock: ExecutionBlock, name: str, verbose: CodeGenVerbosity = _NoVerbosity):
        if "include_benchmark" not in ctxt.globalObjects:
            ctxt.hoistGlobalDefinition("include_benchmark", "#include <benchmark.h>\n")
        if "include_printf" not in ctxt.globalObjects:
            ctxt.hoistGlobalDefinition("include_printf", "#include \"printf.h\"\n")
        tsop = NodeTemplate("""  tsop = benchmark_get_cycle();\n""")
        teop = NodeTemplate("""  teop = benchmark_get_cycle();\n""")
        executionBlock.codeSnippets.insert(1, CodeSnippet(tsop, {}))
        executionBlock.codeSnippets.append(CodeSnippet(teop, {}))
        return ctxt, executionBlock

class SpatzBenchmarkOuterPass(CodeTransformationPass):
    def apply(self, ctxt: NetworkContext, executionBlock: ExecutionBlock, name: str, verbose: CodeGenVerbosity = _NoVerbosity):
        t0 = NodeTemplate("""  uint32_t t0, tsop, teop, te;\n  t0 = benchmark_get_cycle();\n""")
        te = NodeTemplate(f"""te = benchmark_get_cycle();if (snrt_is_dm_core()) {{printf(\"Benchmark of {name}:\\n\");\nprintf(\"t0=%d; tsop=%d; teop=%d; te=%d\\n\", t0, tsop, teop, te);\nprintf(\"data_in=%d; op=%d; data_out=%d; total=%d\\n\\n\", tsop-t0, teop-tsop, te-teop, te-t0); }}\n""")
        
        executionBlock.addLeft(t0, {})
        executionBlock.addRight(te, {})
        return ctxt, executionBlock
