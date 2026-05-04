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
  6. (Optional) With --profileTiling: parses per-tile profiling output and
     produces two breakdown tables:
       - By kernel category  (op type → compute cycles)
       - By transfer level   (compute | L1 DMA | L3 DMA)

Usage (from DeeployTest/):
    python benchmark_training.py [--skipgen] [--profileTiling] [--models NAME ...]

Profiling breakdown notes
-------------------------
With --profileTiling the runner injects `getCycles()` wrappers around every
tiled operator. Output lines look like:

  ===== Profiling {node} =====
  [{node}][SB][N ops][Tile i] Pre-Kernel :  XXXX cycles   <- DMA in
  [{node}][SB][N ops][Tile i] Kernel     :  XXXX cycles   <- compute
  [{node}][SB][N ops][Tile i] Post-Kernel:  XXXX cycles   <- DMA out

For L3 models every operator produces two interleaved profiling blocks per
L3-tile iteration:
  1. Cluster block  (Pre/Post = L2->L1 DMA, Kernel = actual compute)
  2. L3 block       (Pre/Post = L3->L2 DMA, Kernel = entire cluster dispatch)

Classification heuristic: if a section's tile count is strictly less than the
immediately preceding section with the same node name it is the L3-level block;
otherwise it is a cluster block.  This works because each L3 tile fires one
cluster dispatch over M L1 tiles (M > 1 typically), while the L3 block covers
exactly 1 L3 tile per iteration.
"""

import argparse
import collections
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
        "num_data_inputs": 1,
        "tolerance": 5e-3
    },
    "Models/Training/MobileNetV1/mobilenetv1_train": {
        "num_data_inputs": 2
    },
    "Models/Training/DSCNN/dscnn_train": {
        "tolerance": 3e-2
    },
}

# ---------------------------------------------------------------------------
# Paths & regexes
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "deeployTrainingRunner_tiled_siracusa.py"

BENCH_RE = re.compile(r"BENCH train_cycles=(\d+) opt_cycles=(\d+) weight_sram=(\d+)")
ERRORS_RE = re.compile(r"Errors:\s+(\d+)\s+out of\s+(\d+)")
ARENA_RE = re.compile(r"static const uint32_t (Deeploy(?:Opt)?Network_MEMORYARENA_\w+_len)\s*=\s*(\d+);")
DEFINE_RE = re.compile(r"#define\s+(\w+)\s+(\d+)")

# Profiling output regexes
_SECTION_RE = re.compile(r"===== Profiling (\S+) =====")
_TILE_RE = re.compile(r"\[(?P<node>\w+)\]\[(?P<buf>\w+)\]\[(?P<ops>\d+) ops\]\[Tile (?P<tile>\d+)\]"
                      r"\s+(?P<flavor>Pre-Kernel|Kernel\s+|Post-Kernel)\s*:\s*(?P<cycles>\d+)")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    return SCRIPT_DIR / f"TEST_{platform}" / "Tests" / test_rel


def _op_type(node_name: str) -> str:
    """Strip trailing _N index to get the operator type name."""
    return re.sub(r"_\d+$", "", node_name)


# ---------------------------------------------------------------------------
# Profiling parser
# ---------------------------------------------------------------------------


def parse_profiling(output: str):
    """Parse --profileTiling UART output into two breakdown dicts.

    Returns
    -------
    kernel_breakdown : dict[str, int]
        op_type -> total compute (Kernel) cycles summed across all tiles
        and all training steps.
    time_breakdown : dict[str, int]
        'compute'  -> cluster Kernel cycles
        'l1_dma'   -> cluster Pre-Kernel + Post-Kernel cycles (L2<->L1)
        'l3_dma'   -> L3-block Pre-Kernel + Post-Kernel cycles (L3<->L2)
    """

    # --- Step 1: split output into sections --------------------------------
    # A section is everything between two consecutive "===== Profiling" lines.
    # Each section belongs to one (node_name, tiling_level) pair.

    # List of (node_name, {tile_idx: {'pre': N, 'ker': N, 'post': N}})
    sections = []
    current_node = None
    current_tiles: dict = {}

    for line in output.splitlines():
        m = _SECTION_RE.search(line)
        if m:
            if current_node is not None:
                sections.append((current_node, current_tiles))
            current_node = m.group(1)
            current_tiles = {}
            continue

        if current_node is None:
            continue

        mt = _TILE_RE.search(line)
        if mt:
            tile = int(mt.group("tile"))
            flavor = mt.group("flavor").strip()
            cycles = int(mt.group("cycles"))
            entry = current_tiles.setdefault(tile, {"pre": 0, "ker": 0, "post": 0})
            if flavor == "Pre-Kernel":
                entry["pre"] += cycles
            elif flavor == "Post-Kernel":
                entry["post"] += cycles
            elif flavor == "Kernel":
                entry["ker"] += cycles

    if current_node is not None:
        sections.append((current_node, current_tiles))

    # --- Step 2: classify each section as cluster or L3 --------------------
    # Within each (node_name) sequence, sections alternate:
    #   cluster_0, l3_0, cluster_1, l3_1, ...
    # The L3 section covers 1 tile per L3-tile iteration; the cluster section
    # covers M L1 tiles.  Heuristic: if the tile count < previous section's
    # tile count for the same node -> L3 block.

    # Group section indices by node_name preserving order
    node_idx: dict = collections.defaultdict(list)
    for i, (node, _) in enumerate(sections):
        node_idx[node].append(i)

    # Classify each section index
    is_l3: dict = {}  # section_idx -> bool
    for node, idxs in node_idx.items():
        prev_count = None
        for i in idxs:
            count = len(sections[i][1])  # number of tiles in this section
            if prev_count is not None and count < prev_count:
                is_l3[i] = True
            else:
                is_l3[i] = False
            prev_count = count

    # --- Step 3: aggregate -------------------------------------------------
    kernel_breakdown: dict = {}
    time_breakdown = {"compute": 0, "l1_dma": 0, "l3_dma": 0}

    for i, (node, tiles) in enumerate(sections):
        pre = sum(t["pre"] for t in tiles.values())
        ker = sum(t["ker"] for t in tiles.values())
        post = sum(t["post"] for t in tiles.values())
        otype = _op_type(node)

        if is_l3.get(i, False):
            # L3-level block: Pre/Post = L3<->L2 DMA; Kernel = cluster dispatch
            time_breakdown["l3_dma"] += pre + post
            # Do NOT count Kernel here — it double-counts cluster work
        else:
            # Cluster block: Pre/Post = L2<->L1 DMA; Kernel = compute
            kernel_breakdown[otype] = kernel_breakdown.get(otype, 0) + ker
            time_breakdown["compute"] += ker
            time_breakdown["l1_dma"] += pre + post

    return kernel_breakdown, time_breakdown


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_model(test_rel: str,
              mem_level: str,
              l1: int,
              cores: int,
              skipgen: bool,
              profile_tiling: bool = False,
              num_data_inputs = None,
              tolerance = None,
              profile_nodes = None) -> dict:
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
    if profile_tiling:
        cmd.append("--profileTiling")
    if profile_nodes:
        cmd += [f"--profileNodes={','.join(profile_nodes)}"]
    if num_data_inputs is not None:
        cmd += ["--num-data-inputs", str(num_data_inputs)]
    if tolerance is not None:
        cmd += ["--tolerance", str(tolerance)]

    print(f"\n{'='*60}", flush = True)
    print(f"Running: {name}  [{mem_level}]  l1={l1}", flush = True)
    print(f"  cmd: {' '.join(cmd)}", flush = True)

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output = True, text = True, timeout = 900, cwd = str(SCRIPT_DIR))
        wall_s = round(time.time() - t0, 1)
        output = proc.stdout + proc.stderr
    except subprocess.TimeoutExpired:
        row.update({"success": False, "wall_s": 900, "error": "TIMEOUT"})
        print("  TIMEOUT after 900 s", flush = True)
        return row
    except Exception as e:
        row.update({"success": False, "wall_s": round(time.time() - t0, 1), "error": str(e)})
        return row

    row["wall_s"] = wall_s
    row["rc"] = proc.returncode

    bm = BENCH_RE.search(output)
    if bm:
        row["train_cycles_total"] = int(bm.group(1))
        row["opt_cycles_total"] = int(bm.group(2))
        row["weight_sram"] = int(bm.group(3))
    else:
        row["train_cycles_total"] = row["opt_cycles_total"] = row["weight_sram"] = None

    em = ERRORS_RE.search(output)
    if em:
        row["errors"] = int(em.group(1))
        row["loss_checks"] = int(em.group(2))
        row["success"] = (proc.returncode == 0 and row["errors"] == 0)
    else:
        row["errors"] = row["loss_checks"] = None
        row["success"] = (proc.returncode == 0)

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

    def _max(*vals):
        vs = [v for v in vals if v is not None]
        return max(vs) if vs else None

    row["peak_l1"] = _max(row["train_arena_l1"], row["opt_arena_l1"])
    row["peak_l2_max"] = _max(row["train_arena_l2"], row["opt_arena_l2"])
    row["peak_l3_max"] = _max(row["train_arena_l3"], row["opt_arena_l3"])

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

    # Profiling breakdown
    if profile_tiling:
        kb, tb = parse_profiling(output)
        row["prof_compute"] = tb["compute"]
        row["prof_l1_dma"] = tb["l1_dma"]
        row["prof_l3_dma"] = tb["l3_dma"]
        row["prof_kernel_breakdown"] = kb  # stored separately, not in CSV

    return row


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------


def _pct(part, total):
    return f"{100*part/total:.1f}%" if total else "—"


def print_kernel_breakdown(name: str, kb: dict, total_cycles: int):
    print(f"\n  Kernel-type breakdown — {name}")
    print(f"  {'op_type':<35} {'compute_cyc':>14} {'% of tracked':>13}")
    print(f"  {'-'*35} {'-'*14} {'-'*13}")
    tracked = sum(kb.values())
    for op, cyc in sorted(kb.items(), key = lambda x: -x[1]):
        print(f"  {op:<35} {cyc:>14,} {_pct(cyc, tracked):>13}")
    untracked = (total_cycles or 0) - tracked
    print(f"  {'(untracked / overhead)':<35} {untracked:>14,} {_pct(untracked, total_cycles or 1):>13}")
    print(f"  {'BENCH total':<35} {total_cycles or 0:>14,} {'100%':>13}")


def print_time_breakdown(name: str, tb: dict, total_cycles: int):
    compute = tb["compute"]
    l1 = tb["l1_dma"]
    l3 = tb["l3_dma"]
    tracked = compute + l1 + l3
    print(f"\n  Compute vs transfer breakdown — {name}")
    print(f"  {'category':<25} {'cycles':>14} {'% of BENCH':>11} {'% of tracked':>13}")
    print(f"  {'-'*25} {'-'*14} {'-'*11} {'-'*13}")
    for label, val in [("Compute (Kernel)", compute), ("L1 DMA (L2<->L1)", l1), ("L3 DMA (L3<->L2)", l3)]:
        print(f"  {label:<25} {val:>14,} {_pct(val, total_cycles or 1):>11} "
              f"{_pct(val, tracked):>13}")
    print(f"  {'Tracked subtotal':<25} {tracked:>14,} {_pct(tracked, total_cycles or 1):>11}")
    untracked = (total_cycles or 0) - tracked
    print(f"  {'Overhead / inter-op':<25} {untracked:>14,} {_pct(untracked, total_cycles or 1):>11}")


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
    "prof_compute",
    "prof_l1_dma",
    "prof_l3_dma",
    "error",
]

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description = "Deeploy training benchmark")
    parser.add_argument("--skipgen", action = "store_true", help = "Skip codegen (reuse existing generated code)")
    parser.add_argument("--profileTiling",
                        action = "store_true",
                        help = "Enable per-tile cycle profiling (two breakdown tables per model)")
    parser.add_argument(
        "--profileNodes",
        type = lambda s: s.split(','),
        default = None,
        metavar = "SUBSTR[,...]",
        help = "With --profileTiling: restrict to nodes whose name contains any comma-separated substring")
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
            profile_tiling = args.profileTiling,
            num_data_inputs = overrides.get("num_data_inputs"),
            tolerance = overrides.get("tolerance"),
            profile_nodes = args.profileNodes,
        )
        row["optimizer_cycles_total"] = row.pop("opt_cycles_total", None)
        rows.append(row)

    # Write CSV
    out_path = Path(args.out)
    with open(out_path, "w", newline = "") as f:
        writer = csv.DictWriter(f, fieldnames = COLUMNS, extrasaction = "ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {out_path.resolve()}")

    # --- Summary table ------------------------------------------------------
    print("\n" + "=" * 100)
    hdr = (f"{'name':<22} {'mem':<4} {'ok':<5} {'wall_s':>7} {'train_cyc':>13} "
           f"{'opt_cyc':>12} {'weight_sram':>12} {'peak_l1':>8} {'peak_l2':>8} {'peak_l3':>10}")
    print(hdr)
    print("-" * 100)
    for r in rows:
        print(f"{r.get('name',''):<22} {r.get('mem_level',''):<4} "
              f"{'Y' if r.get('success') else 'N':<5} "
              f"{r.get('wall_s',''):>7} "
              f"{str(r.get('train_cycles_total') or ''):>13} "
              f"{str(r.get('optimizer_cycles_total') or ''):>12} "
              f"{str(r.get('weight_sram') or ''):>12} "
              f"{str(r.get('peak_l1') or ''):>8} "
              f"{str(r.get('peak_l2_max') or ''):>8} "
              f"{str(r.get('peak_l3_max') or ''):>10}")

    # --- Profiling breakdowns (if requested) --------------------------------
    if args.profileTiling:
        print("\n" + "=" * 100)
        print("PROFILING BREAKDOWNS")
        for r in rows:
            if not r.get("prof_kernel_breakdown"):
                continue
            tc = r.get("train_cycles_total", 0)
            print(f"\n{'='*60}")
            print(f"  {r['name']}  [{r['mem_level']}]")
            print_kernel_breakdown(r["name"], r["prof_kernel_breakdown"], tc)
            print_time_breakdown(r["name"], {
                "compute": r.get("prof_compute", 0),
                "l1_dma": r.get("prof_l1_dma", 0),
                "l3_dma": r.get("prof_l3_dma", 0),
            }, tc)


if __name__ == "__main__":
    main()
