#!/usr/bin/env bash
# Run CCT-2 and CCT-3 NOGCP/SEGSPLIT/SIZE* variants on Siracusa/gvsoc with
# L2-resident mode (--defaultMemLevel L2, --l2 4M).
set -u
cd "$(dirname "$0")/.."

LOG_DIR="$(pwd)/scripts/logs/benchmark"
CSV_OUT="$LOG_DIR/full_benchmark.csv"
mkdir -p "$LOG_DIR"

CSV_HEADER="variant,model,tag,l2_peak,fwd_cycles,bwd_cycles,accum_cycles,opt_cycles,train_calls,opt_calls"
[[ ! -f "$CSV_OUT" ]] && echo "$CSV_HEADER" > "$CSV_OUT"

VARIANTS=()
for N in 2 3; do
  VARIANTS+=( run_cct_${N}_nogcp run_cct_${N}_segsplit
              run_cct_${N}_size256 run_cct_${N}_size512
              run_cct_${N}_size1024 run_cct_${N}_size2048
              run_cct_${N}_size4096 run_cct_${N}_size8192
              run_cct_${N}_cost1 run_cct_${N}_cost2 run_cct_${N}_cost4
              run_cct_${N}_cost8 run_cct_${N}_cost12 run_cct_${N}_cost16
              run_cct_${N}_ln run_cct_${N}_softmax run_cct_${N}_gelu
              run_cct_${N}_ln_softmax run_cct_${N}_ln_gelu
              run_cct_${N}_korthikanti run_cct_${N}_ln_kc )
done
VARIANTS+=( run_cct_3_cost18 )
echo "CCT-2/3 variants (${#VARIANTS[@]}): ${VARIANTS[*]}"

emit_row() {
  local v="$1"
  python3 - "$v" "$LOG_DIR" "$(pwd)/TEST_SIRACUSA/Tests/Models/Training" "$CSV_OUT" <<'PYEOF'
import os, re, json, csv, sys
v, log_dir, dt, csv_out = sys.argv[1:]
# v = run_cct_<N>_<tag>
m = re.match(r"run_cct_(\d)_(.+)", v)
if not m: print(f"  [skip CSV emit]: {v}"); sys.exit(0)
N, tag = m.group(1), m.group(2)
model = f"CCT_{N}"
test_dir = model if tag == "nogcp" else f"{model}_{tag.upper()}"
html = f"{dt}/{test_dir}/cct_{N}_train/deeployStates/memory_alloc.html"

def peak_l2(p):
    if not os.path.exists(p): return ""
    with open(p) as f: html = f.read()
    pat = re.compile(r'var fig = (\{)', re.DOTALL)
    target = None
    for m in pat.finditer(html):
        s = m.start(1); depth = 0; i = s
        while i < len(html):
            c = html[i]
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    end = i+1
                    pm = re.search(r'Plotly\.newPlot\("(plot-L\d+)"', html[end:end+200])
                    if pm and pm.group(1) == 'plot-L2':
                        target = html[s:end]
                    break
            i += 1
        if target: break
    if not target: return ""
    try: d = json.loads(target)
    except Exception: return ""
    pk = 0
    for tr in d.get('data', []):
        if 'Memory Size' in tr.get('name',''): continue
        ys = [v for v in (tr.get('y') or []) if v is not None]
        if ys: pk = max(pk, max(ys))
    return pk

TRAIN_RE = re.compile(r"\[BENCH-TRAIN\]\s+fwd=(\d+)\s+bwd=(\d+)\s+accum=(\d+)")
OPT_RE   = re.compile(r"\[BENCH-OPT\]\s+opt=(\d+)")
fwd, bwd, accum, opt = [], [], [], []
log = f"{log_dir}/{v}.log"
if os.path.exists(log):
    with open(log) as f: txt = f.read()
    for m in TRAIN_RE.finditer(txt):
        fwd.append(int(m.group(1))); bwd.append(int(m.group(2))); accum.append(int(m.group(3)))
    for m in OPT_RE.finditer(txt):
        opt.append(int(m.group(1)))
avg = lambda xs: sum(xs)//len(xs) if xs else 0
row = [v, model.upper(), tag, peak_l2(html), avg(fwd), avg(bwd), avg(accum),
       avg(opt), len(fwd), len(opt)]
with open(csv_out, "a", newline="") as f:
    csv.writer(f).writerow(row)
print(f"  -> {model}/{tag}: L2={row[3]} fwd={row[4]} bwd={row[5]} "
      f"accum={row[6]} opt={row[7]} (#train={row[8]} #opt={row[9]})")
PYEOF
}

already_done() { grep -qE "^${1}," "$CSV_OUT" 2>/dev/null; }

PASS=(); FAIL=(); SKIP=()
for v in "${VARIANTS[@]}"; do
  [[ -x "scripts/$v.sh" ]] || chmod +x "scripts/$v.sh" 2>/dev/null
  if already_done "$v"; then echo "SKIP $v (already in CSV)"; SKIP+=("$v"); continue; fi
  log="$LOG_DIR/$v.log"
  start=$SECONDS
  echo "[$(date +%H:%M:%S)]  $v"
  if bash "scripts/$v.sh" > "$log" 2>&1; then
    dur=$((SECONDS - start))
    n_train=$(grep -c "BENCH-TRAIN" "$log" 2>/dev/null; true); n_train=${n_train:-0}
    n_opt=$(grep -c "BENCH-OPT" "$log" 2>/dev/null; true);     n_opt=${n_opt:-0}
    echo "  PASS (${dur}s)  bench-train=$n_train  bench-opt=$n_opt"
    emit_row "$v"
    PASS+=("$v")
  else
    echo "  FAIL  -- see $log"; FAIL+=("$v")
  fi
done

echo
echo "================================================================"
echo " CCT-2/3 sweep done   PASS=${#PASS[@]} SKIP=${#SKIP[@]} FAIL=${#FAIL[@]}"
echo " CSV: $CSV_OUT"
echo "================================================================"
[[ ${#FAIL[@]} -eq 0 ]] && exit 0 || exit 1
