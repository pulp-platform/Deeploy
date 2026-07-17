"""Insert phase-level cycle counters into generated Deeploy training/optimizer C.

Phases printed per call:
    TrainingNetwork.c:  forward (entry → after SoftmaxCrossEntropyLoss closure)
                        backward (after SCE → before first GradientAccumulator)
                        accumulator (first accumulator → function exit)
    OptimizerNetwork.c: optimizer (function entry → exit)

Output format (one line per call):
    [BENCH-TRAIN] fwd=<u> bwd=<u> accum=<u>
    [BENCH-OPT]   opt=<u>

Idempotent: a sentinel comment is added on first edit; subsequent runs skip.

Run automatically after generateTrainingNetwork.py / generateOptimizerNetwork.py
(see core/execution.py post-codegen hook).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SENTINEL = "// __DEEPLOY_BENCH_INSTRUMENTED__"


def _already_instrumented(src: str) -> bool:
    return SENTINEL in src


def instrument_training_network(path: Path) -> bool:
    """Return True if the file was modified."""
    src = path.read_text()
    if _already_instrumented(src):
        return False

    # Locate void RunTrainingNetwork() { ... }
    m_open = re.search(r"^void\s+RunTrainingNetwork\s*\(\s*\)\s*\{", src, re.MULTILINE)
    if not m_open:
        print(f"  [WARN] RunTrainingNetwork not found in {path}", file=sys.stderr)
        return False

    # Find the matching closing brace for the function body
    body_start = m_open.end()
    depth = 1
    i = body_start
    n = len(src)
    while i < n and depth > 0:
        c = src[i]
        if c == "{": depth += 1
        elif c == "}": depth -= 1
        i += 1
    if depth != 0:
        print(f"  [WARN] could not find closing brace of RunTrainingNetwork", file=sys.stderr)
        return False
    body_end = i - 1  # position of the closing '}'

    body = src[body_start:body_end]

    # 1) After-LOSS marker: insert immediately after the LOSS forward op.
    #    Match any forward-pass loss closure call:
    #      SoftmaxCrossEntropyLoss: "_onnxSoftmaxCrossEntropyLoss<id>_SoftmaxCrossEntropyLoss_closure_L3"
    #      MSELoss               : "MSELoss_fused_MSELoss_closure_L3"
    #    Exclude *Grad closures so the marker stays in the forward window.
    sce_call = None
    for m in re.finditer(r"(\w*(?:SoftmaxCrossEntropy|MSE)Loss\w*_closure_L3\([^;]*?\);)",
                         body, re.DOTALL):
        if "Grad" in m.group(1):
            continue
        sce_call = m
        break
    # 2) Before-first-GradientAccumulator marker: insert immediately before the
    #    first "_GradientAccumulator..._closure_L3" call.
    accum_call = re.search(
        r"(_GradientAccumulator\w+_closure_L3\([^;]*?\);)",
        body, re.DOTALL,
    )

    if sce_call is None or accum_call is None:
        print(f"  [WARN] could not locate phase boundaries in {path}", file=sys.stderr)
        return False

    # Build instrumented body. Insert in REVERSE order to keep indices valid.
    # We're on the cluster (RunTrainingNetwork runs via pi_cluster_send_task_to_cl);
    # the training entry never calls ResetTimer/StartTimer, so we configure +
    # reset + start the cluster perf counter ourselves before reading.
    # Declare ALL bench vars at function top so they're visible from the
    # epilogue printf. The SCE / GradientAccumulator calls live inside
    # nested `if (pi_core_id() == 0) { ... }` scopes, so a `uint32_t t1 = ...`
    # inserted there would NOT be visible at the function-end printf.
    pre_decl = (
        f"\n  {SENTINEL}\n"
        f"  uint32_t __bench_t0, __bench_t1 = 0, __bench_t2 = 0, __bench_t3 = 0;\n"
        f"  pi_perf_conf((1 << PI_PERF_CYCLES));\n"
        f"  pi_perf_reset();\n"
        f"  pi_perf_start();\n"
        f"  __bench_t0 = pi_perf_read(PI_PERF_CYCLES);\n"
    )
    after_sce = (
        f"\n  __bench_t1 = pi_perf_read(PI_PERF_CYCLES);\n"
    )
    before_accum = (
        f"\n  __bench_t2 = pi_perf_read(PI_PERF_CYCLES);\n"
    )
    epilogue = (
        f"\n  __bench_t3 = pi_perf_read(PI_PERF_CYCLES);\n"
        f"  printf(\"[BENCH-TRAIN] fwd=%u bwd=%u accum=%u\\r\\n\","
        f" __bench_t1 - __bench_t0, __bench_t2 - __bench_t1, __bench_t3 - __bench_t2);\n"
    )

    new_body = body
    # 3) before first GradientAccumulator
    new_body = new_body[:accum_call.start()] + before_accum + new_body[accum_call.start():]
    # 2) after SCE — note SCE comes before accum, so its absolute offset in new_body
    #    is unchanged because we inserted AFTER it on the line above. Wait, we
    #    inserted before accum_call.start which is AFTER sce_call.end, so SCE
    #    indices in new_body are still valid.
    new_body = new_body[:sce_call.end()] + after_sce + new_body[sce_call.end():]
    # 1) at function entry
    new_body = pre_decl + new_body

    new_src = src[:body_start] + new_body + epilogue + src[body_end:]
    path.write_text(new_src)
    return True


def instrument_optimizer_network(path: Path) -> bool:
    src = path.read_text()
    if _already_instrumented(src):
        return False

    m_open = re.search(r"^void\s+RunOptimizerNetwork\s*\(\s*\)\s*\{", src, re.MULTILINE)
    if not m_open:
        print(f"  [WARN] RunOptimizerNetwork not found in {path}", file=sys.stderr)
        return False

    body_start = m_open.end()
    depth = 1; i = body_start; n = len(src)
    while i < n and depth > 0:
        c = src[i]
        if c == "{": depth += 1
        elif c == "}": depth -= 1
        i += 1
    if depth != 0:
        return False
    body_end = i - 1

    pre_decl = (
        f"\n  {SENTINEL}\n"
        f"  uint32_t __bench_opt_t0, __bench_opt_t1 = 0;\n"
        f"  pi_perf_conf((1 << PI_PERF_CYCLES));\n"
        f"  pi_perf_reset();\n"
        f"  pi_perf_start();\n"
        f"  __bench_opt_t0 = pi_perf_read(PI_PERF_CYCLES);\n"
    )
    epilogue = (
        f"\n  __bench_opt_t1 = pi_perf_read(PI_PERF_CYCLES);\n"
        f"  printf(\"[BENCH-OPT] opt=%u\\r\\n\", __bench_opt_t1 - __bench_opt_t0);\n"
    )
    new_src = src[:body_start] + pre_decl + src[body_start:body_end] + epilogue + src[body_end:]
    path.write_text(new_src)
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: instrument_bench.py <test_dir>", file=sys.stderr)
        sys.exit(1)

    test_dir = Path(sys.argv[1])
    train_c = test_dir / "TrainingNetwork.c"
    opt_c = test_dir / "OptimizerNetwork.c"

    if train_c.exists():
        if instrument_training_network(train_c):
            print(f"  [BENCH] instrumented {train_c}")
        else:
            print(f"  [BENCH] {train_c} already instrumented or unmodifiable")
    else:
        print(f"  [WARN] no TrainingNetwork.c at {train_c}", file=sys.stderr)

    # OptimizerNetwork.c lives in the optimizer-test dir, not the training dir.
    # Prefer to instrument it via a separate call.
    if opt_c.exists():
        if instrument_optimizer_network(opt_c):
            print(f"  [BENCH] instrumented {opt_c}")


if __name__ == "__main__":
    main()
