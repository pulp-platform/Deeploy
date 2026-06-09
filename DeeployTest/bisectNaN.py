# SPDX-License-Identifier: Apache-2.0
"""
NaN-bisection helper for the on-board fp32 flow.

Truncates a network so that a chosen intermediate tensor becomes the single
graph output, recomputes the golden values with onnxruntime, and writes a
self-contained test directory. Run that directory through the normal
deployRunner; the [SUMMARY] line printed by the testbench tells you whether
that tensor is already NaN on-board.

Usage:
    # List candidate cut points (tensor produced by each node, in order)
    python3 bisectNaN.py --src Tests/Models/MCUNet --list

    # Cut at a given output tensor name (or node index from --list)
    python3 bisectNaN.py --src Tests/Models/MCUNet --at conv2d_12
    python3 bisectNaN.py --src Tests/Models/MCUNet --at-index 20

Then:
    python3 deployRunner_tiled_gap9.py -t Tests/Models/MCUNet_cut -s board --defaultMemLevel=L3
and read the [SUMMARY] line:  nan=0  -> tensor is clean, NaN is downstream
                              nan>0  -> NaN already here, move upstream
"""
import argparse
import os
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper


def node_outputs(model):
    """Ordered list of (index, node_name, output_tensor_name) for every node."""
    out = []
    for i, n in enumerate(model.graph.node):
        for o in n.output:
            out.append((i, n.name, o))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source test dir (with network.onnx, inputs.npz)")
    ap.add_argument("--at", default=None, help="output tensor name to cut at")
    ap.add_argument("--at-index", type=int, default=None, help="cut at the Nth entry from --list")
    ap.add_argument("--list", action="store_true", help="list candidate cut points and exit")
    ap.add_argument("--dst", default=None, help="destination test dir (default: <src>_cut)")
    args = ap.parse_args()

    model = onnx.load(os.path.join(args.src, "network.onnx"))
    cuts = node_outputs(model)

    if args.list:
        for k, (idx, name, tname) in enumerate(cuts):
            print(f"[{k:3d}] node#{idx:3d} {model.graph.node[idx].op_type:12s} {name:24s} -> {tname}")
        return

    if args.at_index is not None:
        target = cuts[args.at_index][2]
    elif args.at is not None:
        target = args.at
    else:
        raise SystemExit("Provide --at <tensor> or --at-index <k> (see --list)")

    dst = args.dst or (args.src.rstrip("/") + "_cut")
    os.makedirs(dst, exist_ok=True)

    # Load input(s)
    inputs = np.load(os.path.join(args.src, "inputs.npz"))
    feed = {k: inputs[k] for k in inputs.files}

    # Map graph input names to npz keys positionally if names differ
    sess_in_names = [i.name for i in model.graph.input]
    if set(feed.keys()) != set(sess_in_names):
        feed = {sess_in_names[i]: inputs[k] for i, k in enumerate(inputs.files)}

    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    # Need the intermediate as an output: re-export model with target added as output
    vi = helper.ValueInfoProto()
    vi.name = target
    m2 = onnx.ModelProto()
    m2.CopyFrom(model)
    if target not in [o.name for o in m2.graph.output]:
        m2.graph.output.append(vi)
    sess = ort.InferenceSession(m2.SerializeToString(), providers=["CPUExecutionProvider"])
    golden = sess.run([target], feed)[0]
    print(f"Cut at '{target}': golden shape {golden.shape}, "
          f"min={golden.min():.4f} max={golden.max():.4f} "
          f"nan={int(np.isnan(golden).sum())}")

    # Build a truncated onnx whose only output is `target`
    cut_model = onnx.utils.extract_model  # available in onnx>=1.8
    in_names = [i.name for i in model.graph.input]
    tmp_src = os.path.join(dst, "_full.onnx")
    onnx.save(model, tmp_src)
    onnx.utils.extract_model(tmp_src, os.path.join(dst, "network.onnx"), in_names, [target])
    os.remove(tmp_src)

    # Write inputs (unchanged) and golden outputs for the cut tensor
    np.savez(os.path.join(dst, "inputs.npz"), **{k: inputs[k] for k in inputs.files})
    np.savez(os.path.join(dst, "outputs.npz"), output=golden.astype(np.float32))

    print(f"Wrote {dst}/  (network.onnx, inputs.npz, outputs.npz)")
    print(f"Now run:  python3 deployRunner_tiled_gap9.py -t {dst} -s board --defaultMemLevel=L3")


if __name__ == "__main__":
    main()
