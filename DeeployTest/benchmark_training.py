#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark runner for Deeploy end-to-end training tests on Siracusa/GVSoC.

For each model in the training test registry (test_siracusa_tiled_config.py),
this script:
  1. Runs the tiled-Siracusa training runner (codegen + build + gvsoc sim).
  2. Parses the BENCH line from the simulator UART output.
  3. Reads arena sizes from the generated TrainingNetwork.h / OptimizerNetwork.h.
  4. Reads N_TRAIN_STEPS / N_ACCUM_STEPS / TRAINING_NUM_WEIGHT_INPUTS
     from the generated testinputs.h.
  5. Emits a CSV row for every model.

Usage (from DeeployTest/):
    python benchmark_training.py [--skipgen] [--l1 128000] [--cores 8] [--out bench.csv]

With --skipgen the codegen step is skipped (generated code must already exist
under TEST_SIRACUSA/Tests/...). The binary is still rebuilt so that the
updated deeploytraintest.c (with cycle counters) takes effect.
"""

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry (mirrors test_siracusa_tiled_config.py)
# ---------------------------------------------------------------------------

L2_MODELS = {
    "Models/Training/SimpleMLP/simplemlp_train": [64000],
    "Models/Training/Autoencoder/autoencoder_train": [128000],
    "Models/Training/DSCNN/dscnn_train": [128000],
}

L3_MODELS = {
    "Models/Training/ResNet8/resnet8_train": [128000],
    "Models/Training/MobileNetV1/mobilenetv1_train": [128000],
    "Models/Training/CCT/cct_train": [128000],
    "Models/Training/CCT_LoRA/cct_lora_train": [128000],
}

MODEL_OVERRIDES = {
    "Models/Training/CCT/cct_train": {
        "num_data_inputs": 1
    },
    "Models/Training/MobileNetV1/mobilenetv1_train": {
        "num_data_inputs": 2
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RUNNER = SCRIPT_DIR / "deeployTrainingRunner_tiled_siracusa.py"

BENCH_RE = re.compile(r"BENCH train_cycles=(\d+) opt_cycles=(\d+) weight_sram=(\d+)")
ERRORS_RE = re.compile(r"Errors:\s+(\d+)\s+out of\s+(\d+)")
ARENA_RE = re.compile(r"static const uint32_t (Deeploy(?:Opt)?Network_MEMORYARENA_\w+_len)\s*=\s*(\d+);")
DEFINE_RE = re.compile(r"#define\s+(\w+)\s+(\d+)")


def _parse_header(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(errors = "replace")
    result = {}
    for m in ARENA_RE.finditer(text):
        result[m.group(1)] = int(m.group(2))
    for m in DEFINE_RE.finditer(text):
        result[m.group(1)] = int(m.group(2))
    return result


def gen_dir_for(test_rel: str, platform: str = "SIRACUSA") -> Path:
    """Return the TEST_<PLATFORM>/Tests/<rel> directory for a test path."""
    test_name = Path(test_rel).name
    return SCRIPT_DIR / f"TEST_{platform}" / "Tests" / test_rel


def run_model(test_rel: str, mem_level: str, l1: int, cores: int, skipgen: bool, num_data_inputs = None) -> dict:
    name = Path(test_rel).name
    row = {"name": name, "mem_level": mem_level}

    cmd = [
        sys.executable,
        str(RUNNER),
        "-t",
        str(SCRIPT_DIR / "Tests" / test_rel),
        "--cores",
        str(cores),
        "--l1",
        str(l1),
        "--l2",
        "2000000",
        "--defaultMemLevel",
        mem_level,
        "--memAllocStrategy",
        "MiniMalloc",
        "--searchStrategy",
        "random-max",
    ]
    if skipgen:
        cmd.append("--skipgen")
    if num_data_inputs is not None:
        cmd += ["--num-data-inputs", str(num_data_inputs)]

    print(f"\n{'='*60}", flush = True)
    print(f"Running: {name}  [{mem_level}]  l1={l1}", flush = True)
    print(f"  cmd: {' '.join(cmd)}", flush = True)

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output = True,
            text = True,
            timeout = 900,
            cwd = str(SCRIPT_DIR),
        )
        wall_s = round(time.time() - t0, 1)
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        row.update({"success": False, "wall_s": 900, "error": "TIMEOUT"})
        print(f"  TIMEOUT after 900 s", flush = True)
        return row
    except Exception as e:
        row.update({"success": False, "wall_s": round(time.time() - t0, 1), "error": str(e)})
        return row

    row["wall_s"] = wall_s
    row["rc"] = proc.returncode

    # Parse BENCH line
    bm = BENCH_RE.search(output)
    if bm:
        row["train_cycles_total"] = int(bm.group(1))
        row["opt_cycles_total"] = int(bm.group(2))
        row["weight_sram"] = int(bm.group(3))
    else:
        row["train_cycles_total"] = row["opt_cycles_total"] = row["weight_sram"] = None

    # Parse Errors line
    em = ERRORS_RE.search(output)
    if em:
        errors = int(em.group(1))
        total = int(em.group(2))
        row["errors"] = errors
        row["loss_checks"] = total
        row["success"] = (proc.returncode == 0 and errors == 0)
    else:
        row["errors"] = None
        row["loss_checks"] = None
        row["success"] = (proc.returncode == 0)

    # Read static info from generated headers
    gen = gen_dir_for(test_rel)
    train_h = _parse_header(gen / "TrainingNetwork.h")
    opt_h = _parse_header(gen / "OptimizerNetwork.h")
    inp_h = _parse_header(gen / "testinputs.h")

    row["n_train_steps"] = inp_h.get("N_TRAIN_STEPS")
    row["n_accum_steps"] = inp_h.get("N_ACCUM_STEPS")
    n_steps = row["n_train_steps"] or 1
    n_accum = row["n_accum_steps"] or 1
    row["train_calls"] = n_steps * n_accum
    row["optimizer_calls"] = n_steps

    row["train_arena_l1"] = train_h.get("DeeployNetwork_MEMORYARENA_L1_len")
    row["train_arena_l2"] = train_h.get("DeeployNetwork_MEMORYARENA_L2_len")
    row["train_arena_l3"] = train_h.get("DeeployNetwork_MEMORYARENA_L3_len", 0)
    row["opt_arena_l1"] = opt_h.get("DeeployOptNetwork_MEMORYARENA_L1_len")
    row["opt_arena_l2"] = opt_h.get("DeeployOptNetwork_MEMORYARENA_L2_len")
    row["opt_arena_l3"] = opt_h.get("DeeployOptNetwork_MEMORYARENA_L3_len", 0)

    # Peak arenas
    def _max(*vals):
        vs = [v for v in vals if v is not None]
        return max(vs) if vs else None

    row["peak_l1"] = _max(row["train_arena_l1"], row["opt_arena_l1"])
    row["peak_l2_max"] = _max(row["train_arena_l2"], row["opt_arena_l2"])
    row["peak_l3_max"] = _max(row["train_arena_l3"], row["opt_arena_l3"])

    # Derived cycle stats
    tc = row["train_cycles_total"]
    oc = row["opt_cycles_total"]
    nc = row["train_calls"]
    ns = row["optimizer_calls"]
    row["train_cycles_per_minibatch"] = round(tc / nc, 1) if tc and nc else None
    row["optimizer_cycles_per_step"] = round(oc / ns, 1) if oc and ns else None
    row["cycles_per_step"] = round((tc + oc) / n_steps, 1) if tc and oc else None

    error_str = ""
    if proc.returncode != 0:
        error_str += f"rc={proc.returncode} "
    if row.get("errors"):
        error_str += f"errors={row['errors']}"
    if not bm:
        error_str += " no_bench_line"
    row["error"] = error_str.strip()

    print(
        f"  wall={wall_s}s  rc={proc.returncode}  "
        f"train_cycles={row.get('train_cycles_total')}  "
        f"opt_cycles={row.get('opt_cycles_total')}  "
        f"weight_sram={row.get('weight_sram')}  "
        f"errors={row.get('errors')}",
        flush = True)
    return row


# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------

COLUMNS = [
    "name",
    "mem_level",
    "success",
    "wall_s",
    "n_train_steps",
    "n_accum_steps",
    "train_calls",
    "optimizer_calls",
    "train_cycles_total",
    "optimizer_cycles_total",
    "train_cycles_per_minibatch",
    "optimizer_cycles_per_step",
    "cycles_per_step",
    "train_arena_l1",
    "train_arena_l2",
    "train_arena_l3",
    "opt_arena_l1",
    "opt_arena_l2",
    "opt_arena_l3",
    "peak_l1",
    "peak_l2_max",
    "peak_l3_max",
    "weight_sram",
    "error",
]

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description = "Deeploy training benchmark")
    parser.add_argument("--skipgen", action = "store_true", help = "Skip codegen (reuse existing generated code)")
    parser.add_argument("--l1",
                        type = int,
                        default = None,
                        help = "Override L1 size for all models (default: per-model)")
    parser.add_argument("--cores", type = int, default = 8)
    parser.add_argument("--out", default = "bench_training.csv", help = "Output CSV file (default: bench_training.csv)")
    parser.add_argument("--models", nargs = "*", default = None, help = "Run only these model names (default: all)")
    args = parser.parse_args()

    all_models = ([(k, "L2", v[0]) for k, v in L2_MODELS.items()] + [(k, "L3", v[0]) for k, v in L3_MODELS.items()])
    if args.models:
        all_models = [(k, ml, l1) for k, ml, l1 in all_models if Path(k).name in args.models or k in args.models]

    rows = []
    for test_rel, mem_level, default_l1 in all_models:
        l1 = args.l1 if args.l1 else default_l1
        overrides = MODEL_OVERRIDES.get(test_rel, {})
        row = run_model(
            test_rel,
            mem_level,
            l1,
            args.cores,
            skipgen = args.skipgen,
            num_data_inputs = overrides.get("num_data_inputs"),
        )
        # rename opt_cycles column to match header
        row["optimizer_cycles_total"] = row.pop("opt_cycles_total", None)
        rows.append(row)

    # Write CSV
    out_path = Path(args.out)
    with open(out_path, "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = COLUMNS, extrasaction = "ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {out_path.resolve()}")

    # Pretty-print summary
    print("\n" + "=" * 100)
    hdr = f"{'name':<22} {'mem':<4} {'ok':<5} {'wall_s':>7} {'train_cyc':>12} {'opt_cyc':>11} {'weight_sram':>12} {'peak_l1':>8} {'peak_l2':>8} {'peak_l3':>10}"
    print(hdr)
    print("-" * 100)
    for r in rows:
        print(f"{r.get('name',''):<22} {r.get('mem_level',''):<4} "
              f"{'Y' if r.get('success') else 'N':<5} "
              f"{r.get('wall_s',''):>7} "
              f"{str(r.get('train_cycles_total','') or ''):>12} "
              f"{str(r.get('optimizer_cycles_total','') or ''):>11} "
              f"{str(r.get('weight_sram','') or ''):>12} "
              f"{str(r.get('peak_l1','') or ''):>8} "
              f"{str(r.get('peak_l2_max','') or ''):>8} "
              f"{str(r.get('peak_l3_max','') or ''):>10}")


if __name__ == "__main__":
    main()
