# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Gradient-checkpointing-aware topological scheduler."""
from typing import List

import onnx_graphsurgeon as gs


def verifySchedule(graph: gs.Graph, schedule, *, strict: bool = False, verbose: bool = True):
    """Validate a scheduler's output against structural invariants."""
    flat = []
    for item in schedule:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    graph_nodes = list(graph.nodes)
    graph_set = {n.name for n in graph_nodes}
    sched_set = {n.name for n in flat}

    pos = {n.name: i for i, n in enumerate(flat)}
    seen_count = {}
    for n in flat:
        seen_count[n.name] = seen_count.get(n.name, 0) + 1

    duplicates = [name for name, c in seen_count.items() if c > 1]
    missing = sorted(graph_set - sched_set)

    producer_of = {}
    for n in graph_nodes:
        for o in n.outputs:
            producer_of[o.name] = n
    ordering_violations = []
    for n in flat:
        for inp in n.inputs:
            prod = producer_of.get(inp.name)
            if prod is None:
                continue
            if prod.name not in pos:
                continue
            if pos[prod.name] >= pos[n.name]:
                ordering_violations.append((n.name, prod.name))

    loss_pos = next(
        (pos[n.name] for n in flat
         if n.op == "SoftmaxCrossEntropyLoss"
         or ("Loss" in n.op and "Grad" not in n.op)),
        None,
    )

    recompute = [n for n in flat if "_recompute" in n.name]
    recompute_in_forward = []
    if loss_pos is not None:
        recompute_in_forward = [n.name for n in recompute if pos[n.name] < loss_pos]

    consumers_of = {}
    for n in flat:
        for inp in n.inputs:
            consumers_of.setdefault(inp.name, []).append(n)
    liveness = []
    for n in recompute:
        first_consumer_pos = None
        for o in n.outputs:
            for c in consumers_of.get(o.name, []):
                cp = pos.get(c.name)
                if cp is not None and cp > pos[n.name]:
                    if first_consumer_pos is None or cp < first_consumer_pos:
                        first_consumer_pos = cp
        if first_consumer_pos is not None:
            liveness.append(first_consumer_pos - pos[n.name])

    avg_live = sum(liveness) / len(liveness) if liveness else 0.0
    max_live = max(liveness) if liveness else 0

    report = {
        "ok": (len(missing) == 0 and len(duplicates) == 0
               and len(ordering_violations) == 0),
        "n_graph_nodes": len(graph_nodes),
        "n_sched": len(flat),
        "missing": missing,
        "duplicates": duplicates,
        "ordering_violations": ordering_violations,
        "loss_boundary_pos": loss_pos,
        "n_recompute": len(recompute),
        "recompute_in_forward": recompute_in_forward,
        "avg_recompute_liveness": avg_live,
        "max_recompute_liveness": max_live,
    }

    if verbose:
        ok_mark = "OK" if report["ok"] else "FAIL"
        rec_mark = "OK" if not recompute_in_forward else "WARN"
        print(f"\n[verifySchedule] {ok_mark}  {len(flat)}/{len(graph_nodes)} nodes scheduled")
        if missing:
            print(f"  FAIL  missing: {len(missing)}  e.g. {missing[:3]}")
        if duplicates:
            print(f"  FAIL  duplicates: {duplicates[:5]}")
        if ordering_violations:
            print(f"  FAIL  ordering violations: {len(ordering_violations)}  e.g. {ordering_violations[:3]}")
        if loss_pos is not None:
            print(f"  loss-boundary at position {loss_pos}  "
                  f"(forward: {loss_pos + 1} nodes, backward: {len(flat) - loss_pos - 1} nodes)")
        if recompute:
            n_bwd = len(recompute) - len(recompute_in_forward)
            print(f"  {rec_mark}  recompute in backward: {n_bwd}/{len(recompute)}  "
                  f"(in forward: {len(recompute_in_forward)})")
            print(f"  recompute liveness window: avg {avg_live:.1f}, max {max_live} steps")

    if strict:
        assert report["ok"], f"Schedule invariants violated: {report}"

    return report


def defaultScheduler(graph: gs.Graph) -> List[gs.Node]:
    """Defer `_recompute` nodes into the backward pass, just before their first consumer."""
    nodes = list(graph.nodes)
    has_recompute = any("_recompute" in n.name for n in nodes)
    if not has_recompute:
        return nodes

    def _is_backward_node(n):
        return ("Grad" in n.op
                or "_backward" in n.name
                or "Grad" in n.name)

    recompute_node_names = {n.name for n in nodes if "_recompute" in n.name}
    producers_by_tensor = {}
    for n in nodes:
        for out in n.outputs:
            if out.name:
                producers_by_tensor[out.name] = n

    def _is_chain_internal_alias(n) -> bool:
        if n.op not in ("Transpose", "Reshape"):
            return False
        if "_recompute" not in n.name:
            return False
        for inp in n.inputs:
            prod = producers_by_tensor.get(inp.name)
            if prod is None:
                continue
            if "_recompute" not in prod.name:
                return False
        return True

    forward_nodes, backward_nodes, recompute_nodes = [], [], []
    for n in nodes:
        if _is_chain_internal_alias(n):
            recompute_nodes.append(n)
        elif _is_backward_node(n):
            backward_nodes.append(n)
        elif "_recompute" in n.name:
            recompute_nodes.append(n)
        else:
            forward_nodes.append(n)

    consumers_of = {}
    for n in nodes:
        for inp in n.inputs:
            consumers_of.setdefault(inp.name, []).append(n)

    recompute_set = {n.name for n in recompute_nodes}
    forward_set = {n.name for n in forward_nodes}

    deferrable = set()
    non_deferrable = set()
    for rc in recompute_nodes:
        ok = True
        for out in rc.outputs:
            for c in consumers_of.get(out.name, []):
                if c.name in forward_set:
                    ok = False
                    break
            if not ok:
                break
        (deferrable if ok else non_deferrable).add(rc.name)

    recompute_producer = {out.name: rc for rc in recompute_nodes for out in rc.outputs}
    changed = True
    while changed:
        changed = False
        for rc in recompute_nodes:
            if rc.name not in non_deferrable:
                continue
            for inp in rc.inputs:
                src = recompute_producer.get(inp.name)
                if src is not None and src.name in deferrable:
                    deferrable.discard(src.name)
                    non_deferrable.add(src.name)
                    changed = True

    print(f"[defaultScheduler] recomputes={len(recompute_nodes)} "
          f"backward={len(backward_nodes)}", flush=True)

    ALIAS_PASSTHROUGH_OPS = {"Reshape"}
    deferrable_recompute_outs = {
        out.name for rc in recompute_nodes if rc.name in deferrable
        for out in rc.outputs
    }
    graph_input_names = {vi.name for vi in getattr(graph, "inputs", []) or []}
    try:
        for t_name, t in graph.tensors().items():
            if isinstance(t, gs.Constant):
                graph_input_names.add(t_name)
    except Exception:
        pass

    deferred_bwd_set: set = set()
    deferred_bwd_outs: set = set()
    extended_producer = dict(recompute_producer)

    changed = True
    while changed:
        changed = False
        for bwd in backward_nodes:
            if bwd.op not in ALIAS_PASSTHROUGH_OPS:
                continue
            if bwd.name in deferred_bwd_set:
                continue
            ok = True
            for inp in bwd.inputs:
                n = inp.name
                if not n:
                    continue
                if n in graph_input_names:
                    continue
                if n in deferrable_recompute_outs:
                    continue
                if n in deferred_bwd_outs:
                    continue
                ok = False
                break
            if ok:
                deferred_bwd_set.add(bwd.name)
                for o in bwd.outputs:
                    deferred_bwd_outs.add(o.name)
                    extended_producer[o.name] = bwd
                changed = True

    backward_set = {b.name for b in backward_nodes}
    forward_section = []
    parked = []
    for n in nodes:
        if n.name in backward_set:
            if n.name in deferred_bwd_set:
                parked.append(n)
            continue
        if n.name in recompute_set and n.name in deferrable:
            parked.append(n)
        else:
            forward_section.append(n)

    emitted = set()
    new_backward = []
    parked_by_name = {p.name: p for p in parked}

    def _schedule_parked(node):
        if node.name in emitted or node.name not in parked_by_name:
            return
        for inp in node.inputs:
            src = extended_producer.get(inp.name)
            if src is not None and src.name in parked_by_name:
                _schedule_parked(src)
        new_backward.append(node)
        emitted.add(node.name)

    for bwd in backward_nodes:
        if bwd.name in deferred_bwd_set:
            continue
        for inp in bwd.inputs:
            src = extended_producer.get(inp.name)
            if src is not None:
                _schedule_parked(src)
        new_backward.append(bwd)

    for p in parked:
        if p.name not in emitted:
            new_backward.append(p)
            emitted.add(p.name)

    print(f"[defaultScheduler] deferred {len(deferred_bwd_set)} alias-bwd, "
          f"forward={len(forward_section)} backward={len(new_backward)}",
          flush=True)

    return forward_section + new_backward
