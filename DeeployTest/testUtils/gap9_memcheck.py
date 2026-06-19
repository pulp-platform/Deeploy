#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""GAP9 L1/L2 memory-budget validator (CI gate).

Turns silent over-subscription — the class of bug that manifests as a 10-minute
GVSoC hang in os_evt_release or a wild-pointer "Invalid access" far from the
cause — into an instant, precise, build-time error.

It models EVERY consumer of each physical level, including the ones the Deeploy
tiler does NOT model:
  * L1 tile arena            : pi_l1_malloc(...) in generated C
  * CC master stack          : conf.cc_stack_size, carved from L1 top at runtime
                               (NOT an ELF section, NOT a malloc -> invisible to
                               --plotMemAlloc; this is what overflowed)
  * static L1 sections       : from the ELF (addr 0x1000xxxx)
  * L2 code/data/bss         : from the ELF (addr 0x1cxxxxxx) — incl PE slave
                               stacks (cluster_slave_stacks, .bss in L2)
  * L2 tile arena + promoted : pi_l2_malloc(...) + PromoteTensorsToL2 log

Exit code 0 = fits; 1 = over-subscribed (with the knob to turn); 2 = usage error.

Usage:
  gap9_memcheck.py <build_dir> <gen_dir> [build_or_promote_log] [--cc-stack N]
"""
import glob
import os
import re
import subprocess
import sys

# ---- physical capacities (GAP9) ----
L1_SIZE = 128 * 1024  # 131072  TCDM
L2_SIZE = 0x190000  # 1572864 (1.5 MB), link.gap9.ld
L1_MIN_HEADROOM = 1024  # warn if free L1 < this (stacks have runtime slop)
READELF = "/app/install/gcc/gap9/bin/riscv32-unknown-elf-readelf"


def die(msg, code = 2):
    print(f"[gap9-memcheck] ERROR: {msg}", file = sys.stderr)
    sys.exit(code)


def parse_args(argv):
    pos, cc = [], None
    i = 0
    while i < len(argv):
        if argv[i] == "--cc-stack":
            cc = int(argv[i + 1])
            i += 2
        else:
            pos.append(argv[i])
            i += 1
    if len(pos) < 2:
        die("usage: gap9_memcheck.py <build_dir> <gen_dir> [log] [--cc-stack N]")
    return pos[0], pos[1], (pos[2] if len(pos) > 2 else None), cc


def find_elf(build):
    cands = []
    for f in glob.glob(os.path.join(build, "*")):
        if os.path.isfile(f) and os.access(f, os.X_OK) and not f.endswith((".bin", ".s", ".hex")):
            try:
                if open(f, "rb").read(4) == b"\x7fELF":
                    cands.append(f)
            except Exception:
                pass
    return max(cands, key = os.path.getmtime) if cands else None


def elf_sections(elf):
    """{region: {section: bytes}} for ALLOC sections, keyed by load address."""
    out = subprocess.run([READELF, "-SW", elf], capture_output = True, text = True).stdout
    regions = {"L1": {}, "L2": {}}
    for line in out.splitlines():
        m = re.search(r"\]\s+(\S+)\s+(PROGBITS|NOBITS)\s+([0-9a-f]{8})\s+[0-9a-f]+\s+([0-9a-f]+)", line)
        if not m:
            continue
        name, _, addr, szhex = m.groups()
        sz = int(szhex, 16)
        if sz == 0:
            continue
        if addr.startswith("1000") or addr.startswith("1001"):
            regions["L1"][name] = regions["L1"].get(name, 0) + sz
        elif addr.startswith("1c"):
            regions["L2"][name] = regions["L2"].get(name, 0) + sz
    return regions


def arena_mallocs(gen):
    res = {"L1": 0, "L2": 0, "L3": 0}
    for fn in ("TrainingNetwork.c", "Network.c", "OptimizerNetwork.c"):
        p = os.path.join(gen, fn)
        if not os.path.exists(p):
            continue
        t = open(p, errors = "ignore").read()
        for sz in re.findall(r"(?:pi_)?l1_malloc\([^;]*?\*\s*(\d+)\)", t):
            res["L1"] = max(res["L1"], int(sz))
        for sz in re.findall(r"pi_l2_malloc\(sizeof\([^)]*\)\s*\*\s*(\d+)\)", t):
            res["L2"] += int(sz)
        for sz in re.findall(r"cl_ram_malloc\((?:sizeof\([^)]*\)\s*\*\s*)?(\d+)\)", t):
            res["L3"] += int(sz)
    return res


def check_init_alloc_order(gen):
    """Codegen-time race check (pure source scan — no build, no sim).

    In any Init*Network function, flag a pi_l2_malloc that appears AFTER a
    cl_ram_malloc. On GAP9 Init runs on the cluster CC: cl_ram_malloc delegates
    to the FC (pi_cl_ram_alloc) while pi_l2_malloc runs pos_alloc directly on the
    CC; both touch the shared L2 freelist, so this ordering races -> FC RTOS
    corruption -> os_evt_release crash/hang at init. Fix = hoist all pi_l2_malloc
    before the first cl_ram_malloc (codeGenerateTraining._hoistL2AllocsBeforeL3).
    """
    issues = []
    for fn in ("TrainingNetwork.c", "Network.c", "OptimizerNetwork.c"):
        p = os.path.join(gen, fn)
        if not os.path.exists(p):
            continue
        lines = open(p, errors = "ignore").read().split("\n")
        i = 0
        while i < len(lines):
            if re.search(r"void\s+Init\w*Network\s*\(", lines[i]):
                depth = 0
                started = False
                seen_cl = False
                for j in range(i, len(lines)):
                    depth += lines[j].count("{") - lines[j].count("}")
                    if "{" in lines[j]:
                        started = True
                    if "cl_ram_malloc(" in lines[j]:
                        seen_cl = True
                    elif seen_cl and "pi_l2_malloc(" in lines[j]:
                        issues.append((fn, j + 1, lines[j].strip()[:70]))
                    if started and depth <= 0:
                        i = j
                        break
            i += 1
    return issues


def cc_stack_from_cache(build, override):
    if override is not None:
        return override, "override"
    cache = os.path.join(build, "CMakeCache.txt")
    if os.path.exists(cache):
        m = re.search(r"CC_STACK_SIZE\S*=(\d+)", open(cache, errors = "ignore").read())
        if m:
            return int(m.group(1)), "CMakeCache"
    return 8192, "default(8192)"  # deeploytraintest.c fallback


def promoted_bytes(log):
    if not log or not os.path.exists(log):
        return 0
    best = 0
    for m in re.finditer(r"promoted \d+ tensors, (\d+) /", open(log, errors = "ignore").read()):
        best = max(best, int(m.group(1)))
    return best


def report(title, cap, items):
    used = sum(v for _, v in items)
    free = cap - used
    print(f"\n=== {title}  (cap {cap} B = {cap/1024:.1f} KB) ===")
    for n, v in items:
        print(f"    {v:9d} B  {v/1024:7.1f} KB  {n}")
    print(f"    {'-'*9}")
    print(f"    {used:9d} B  {used/1024:7.1f} KB  TOTAL  ({100*used/cap:.1f}%)")
    print(f"    {free:9d} B  {free/1024:7.1f} KB  FREE")
    return used, free


def main():
    build, gen, log, cc_override = parse_args(sys.argv[1:])
    elf = find_elf(build)
    if not elf:
        die(f"no ELF found in {build}")
    print(f"[gap9-memcheck] ELF: {elf}")
    regs = elf_sections(elf)
    ar = arena_mallocs(gen)
    cc_stack, cc_src = cc_stack_from_cache(build, cc_override)
    prom = promoted_bytes(log)

    violations = []

    # ---- L1 / TCDM ----
    l1_items = []
    if ar["L1"]:
        l1_items.append(("tile arena  (pi_l1_malloc)", ar["L1"]))
    l1_items.append((f"CC master stack  (cc_stack_size, {cc_src})", cc_stack))
    for n, v in sorted(regs["L1"].items(), key = lambda x: -x[1]):
        l1_items.append((f"L1 section {n}", v))
    used1, free1 = report("L1 / TCDM", L1_SIZE, l1_items)
    if free1 < 0:
        violations.append(f"L1 over-subscribed by {-free1} B. arena {ar['L1']} + cc_stack {cc_stack} "
                          f"(+sections) > {L1_SIZE}. Fix: lower --l1 to <= {ar['L1']+free1} OR "
                          f"cc_stack to <= {cc_stack+free1}.")
    elif free1 < L1_MIN_HEADROOM:
        print(f"\n[gap9-memcheck] WARNING: L1 free {free1} B < {L1_MIN_HEADROOM} B headroom "
              f"(stack high-water is runtime-dependent; risky).")

    # ---- L2 ----
    # NOTE: do NOT add the promotion-log bytes to the total. The promoted pool
    # is a pi_l2_malloc (PROMOTED_POOL_L2) -> already in ar["L2"]; promoted const
    # bytes are baked into .data -> already in the ELF L2 sections. Adding `prom`
    # again double-counts and produces false over-subscription FAILs.
    l2_items = []
    for n, v in sorted(regs["L2"].items(), key = lambda x: -x[1]):
        l2_items.append((f"L2 section {n}", v))
    if ar["L2"]:
        l2_items.append(("tile arena + pools (pi_l2_malloc, incl PROMOTED_POOL)", ar["L2"]))
    used2, free2 = report("L2", L2_SIZE, l2_items)
    if prom:
        print(f"    (info: PromoteTensorsToL2 log reports {prom} B promoted — already "
              f"counted above in pi_l2_malloc / .data, not added again)")
    if free2 < 0:
        violations.append(f"L2 over-subscribed by {-free2} B (> {L2_SIZE}). Reduce promoted-pool "
                          f"headroom or L2 arena.")

    # ---- InitNetwork alloc-order race (codegen-time) ----
    order_issues = check_init_alloc_order(gen)
    if order_issues:
        print("\n=== InitNetwork alloc-order ===")
        for fn, ln, txt in order_issues[:6]:
            print(f"    RACE: {fn}:{ln}  pi_l2_malloc after cl_ram_malloc -> {txt}")
        violations.append(f"InitNetwork alloc-order race ({len(order_issues)} site(s), first "
                          f"{order_issues[0][0]}:{order_issues[0][1]}): pi_l2_malloc after cl_ram_malloc "
                          f"-> GAP9 FC/CC pos_alloc freelist race -> os_evt_release crash/hang at init. "
                          f"Hoist all pi_l2_malloc before the first cl_ram_malloc.")

    if ar["L3"]:
        print(f"\n=== L3 / HyperRAM (no enforced cap) ===\n    {ar['L3']} B  "
              f"{ar['L3']/1e6:.2f} MB  cl_ram arena")

    print()
    if violations:
        for v in violations:
            print(f"[gap9-memcheck] FAIL: {v}", file = sys.stderr)
        sys.exit(1)
    print("[gap9-memcheck] PASS: all levels fit within physical capacity.")
    sys.exit(0)


if __name__ == "__main__":
    main()
