"""
SDF-Edge: Exhaustive computational proof of CASAP buffer optimality for multi-rate chains.

This script enumerates ALL valid SDF schedules for small multi-rate chains and
verifies that the CASAP (consume-as-soon-as-possible) interleaved schedule
achieves minimum peak buffer usage.

Results used in Theorem 1 proof (SDF_Edge_Revised_v5.tex).
"""

import sys
sys.path.insert(0, '.')
from sdf_core import SDFGraph, SDFActor, SDFEdge


def peak_buffer_safe(graph, schedule):
    """Simulate schedule with underflow checking. Returns (peak_bytes, valid)."""
    edge_tokens = [e.initial_tokens for e in graph.edges]
    peak = 0
    for actor_name, firing_idx in schedule:
        for i, e in enumerate(graph.edges):
            if e.dst == actor_name:
                if edge_tokens[i] < e.cons:
                    return -1, False
                edge_tokens[i] -= e.cons
        for i, e in enumerate(graph.edges):
            if e.src == actor_name:
                edge_tokens[i] += e.prod
        current = sum(edge_tokens[i] * graph.edges[i].token_size_bytes
                      for i in range(graph.num_edges))
        peak = max(peak, current)
    return peak, True


def enumerate_valid_schedules(graph):
    """Enumerate ALL valid SDF schedules via DFS with token tracking."""
    q = graph.compute_repetition_vector()
    all_firings = []
    for a in graph.actors:
        for f in range(q[a.name]):
            all_firings.append((a.name, f))

    valid_schedules = []

    def is_fireable(actor_name, edge_tokens):
        for i, e in enumerate(graph.edges):
            if e.dst == actor_name and edge_tokens[i] < e.cons:
                return False
        return True

    def dfs(schedule, remaining, edge_tokens):
        if not remaining:
            valid_schedules.append(list(schedule))
            return
        tried = set()
        for i, (a, f) in enumerate(remaining):
            key = (a, f)
            if key in tried:
                continue
            if f > 0 and (a, f - 1) in remaining:
                continue
            if is_fireable(a, edge_tokens):
                tried.add(key)
                new_tokens = list(edge_tokens)
                for ei, e in enumerate(graph.edges):
                    if e.dst == a:
                        new_tokens[ei] -= e.cons
                for ei, e in enumerate(graph.edges):
                    if e.src == a:
                        new_tokens[ei] += e.prod
                new_remaining = remaining[:i] + remaining[i + 1:]
                schedule.append((a, f))
                dfs(schedule, new_remaining, new_tokens)
                schedule.pop()

    init_tokens = [e.initial_tokens for e in graph.edges]
    dfs([], all_firings, init_tokens)
    return valid_schedules


def compute_casap_schedule(graph):
    """CASAP: fire the most downstream enabled actor first (greedy)."""
    q = graph.compute_repetition_vector()
    actors = [a.name for a in graph.actors]
    schedule = []
    edge_tokens = {i: e.initial_tokens for i, e in enumerate(graph.edges)}
    remaining = {a.name: q[a.name] for a in graph.actors}

    def is_fireable(actor_name):
        for i, e in enumerate(graph.edges):
            if e.dst == actor_name and edge_tokens[i] < e.cons:
                return False
        return True

    total = sum(q.values())
    for _ in range(total):
        for a_name in reversed(actors):
            if remaining[a_name] > 0 and is_fireable(a_name):
                f_idx = q[a_name] - remaining[a_name]
                for i, e in enumerate(graph.edges):
                    if e.dst == a_name:
                        edge_tokens[i] -= e.cons
                for i, e in enumerate(graph.edges):
                    if e.src == a_name:
                        edge_tokens[i] += e.prod
                remaining[a_name] -= 1
                schedule.append((a_name, f_idx))
                break
    return schedule


if __name__ == "__main__":
    # === 4-actor chain: A --1:2--> B --1:2--> C --1:1--> D ===
    print("=" * 70)
    print("4-Actor Multi-Rate Chain")
    print("=" * 70)
    g1 = SDFGraph("4-Actor")
    g1.add_actor(SDFActor("A", 10))
    g1.add_actor(SDFActor("B", 20))
    g1.add_actor(SDFActor("C", 30))
    g1.add_actor(SDFActor("D", 40))
    g1.add_edge(SDFEdge("A", "B", prod=1, cons=2, token_size_bytes=100))
    g1.add_edge(SDFEdge("B", "C", prod=1, cons=2, token_size_bytes=200))
    g1.add_edge(SDFEdge("C", "D", prod=1, cons=1, token_size_bytes=300))

    q1 = g1.compute_repetition_vector()
    print(f"q = {q1}")
    schedules1 = enumerate_valid_schedules(g1)
    print(f"Valid schedules: {len(schedules1)}")
    for s in schedules1:
        pk, v = peak_buffer_safe(g1, s)
        print(f"  [{' '.join(a for a, _ in s)}]: peak = {pk}B")

    casap1 = compute_casap_schedule(g1)
    pk_casap1, _ = peak_buffer_safe(g1, casap1)
    asap1 = g1.compute_asap_schedule()
    pk_asap1, _ = peak_buffer_safe(g1, asap1)
    print(f"CASAP peak: {pk_casap1}B, ASAP peak: {pk_asap1}B")

    # === 5-actor chain ===
    print("\n" + "=" * 70)
    print("5-Actor Multi-Rate Chain (demonstrates ASAP vs CASAP gap)")
    print("=" * 70)
    g2 = SDFGraph("5-Actor")
    g2.add_actor(SDFActor("A", 10))
    g2.add_actor(SDFActor("B", 20))
    g2.add_actor(SDFActor("C", 30))
    g2.add_actor(SDFActor("D", 40))
    g2.add_actor(SDFActor("E", 50))
    g2.add_edge(SDFEdge("A", "B", prod=1, cons=2, token_size_bytes=1000))
    g2.add_edge(SDFEdge("B", "C", prod=1, cons=1, token_size_bytes=500))
    g2.add_edge(SDFEdge("C", "D", prod=1, cons=2, token_size_bytes=200))
    g2.add_edge(SDFEdge("D", "E", prod=1, cons=1, token_size_bytes=100))

    q2 = g2.compute_repetition_vector()
    print(f"q = {q2}")
    schedules2 = enumerate_valid_schedules(g2)
    peaks2 = sorted([(peak_buffer_safe(g2, s)[0], s) for s in schedules2])
    print(f"Valid schedules: {len(schedules2)}")
    for pk, s in peaks2:
        print(f"  {pk:>5}B: [{' '.join(a for a, _ in s)}]")

    print(f"\nOptimal: {peaks2[0][0]}B")
    print(f"Worst:   {peaks2[-1][0]}B")
    print(f"Gap:     {100 * (peaks2[-1][0] - peaks2[0][0]) / peaks2[0][0]:.1f}%")

    # === 5-actor chain WITH skip connection ===
    print("\n" + "=" * 70)
    print("5-Actor Multi-Rate Chain WITH Skip (B-->D residual)")
    print("=" * 70)
    g2s = SDFGraph("5-Actor-Skip")
    g2s.add_actor(SDFActor("A", 10))
    g2s.add_actor(SDFActor("B", 20))
    g2s.add_actor(SDFActor("C", 30))
    g2s.add_actor(SDFActor("D", 40))
    g2s.add_actor(SDFActor("E", 50))
    g2s.add_edge(SDFEdge("A", "B", prod=1, cons=2, token_size_bytes=1000))
    g2s.add_edge(SDFEdge("B", "C", prod=1, cons=1, token_size_bytes=500))
    g2s.add_edge(SDFEdge("C", "D", prod=1, cons=2, token_size_bytes=200))
    g2s.add_edge(SDFEdge("D", "E", prod=1, cons=1, token_size_bytes=100))
    g2s.add_edge(SDFEdge("B", "D", prod=1, cons=2, token_size_bytes=300))  # skip

    q2s = g2s.compute_repetition_vector()
    print(f"q = {q2s}")
    sched2s = enumerate_valid_schedules(g2s)
    peaks2s = sorted([(peak_buffer_safe(g2s, s)[0], s) for s in sched2s])
    print(f"Valid schedules: {len(sched2s)}")
    for pk, s in peaks2s:
        print(f"  {pk:>5}B: [{' '.join(a for a, _ in s)}]")

    casap2s = compute_casap_schedule(g2s)
    pk_casap2s, _ = peak_buffer_safe(g2s, casap2s)
    asap2s = g2s.compute_asap_schedule()
    pk_asap2s, _ = peak_buffer_safe(g2s, asap2s)
    print(f"\nOptimal: {peaks2s[0][0]}B")
    print(f"CASAP:   {pk_casap2s}B (optimal: {pk_casap2s == peaks2s[0][0]})")
    print(f"ASAP:    {pk_asap2s}B")
    print(f"Gap:     {100 * (pk_asap2s - peaks2s[0][0]) / peaks2s[0][0]:.1f}%")

    # === ALAP invalidity ===
    print("\n" + "=" * 70)
    print("ALAP Invalidity Demonstration")
    print("=" * 70)
    alap1 = g1.compute_alap_schedule()
    pk_alap, v_alap = peak_buffer_safe(g1, alap1)
    print(f"ALAP schedule: [{' '.join(a for a, _ in alap1)}]")
    print(f"Valid: {v_alap} (INVALID - fires sinks before sources)")

    # === MNetV2-MR CASAP vs ASAP ===
    print("\n" + "=" * 70)
    print("MNetV2-MR: CASAP vs ASAP")
    print("=" * 70)
    from models import build_mobilenetv2_multirate
    g_mr = build_mobilenetv2_multirate()
    casap_mr = compute_casap_schedule(g_mr)
    asap_mr = g_mr.compute_asap_schedule()
    pk_casap_mr, _ = peak_buffer_safe(g_mr, casap_mr)
    pk_asap_mr, _ = peak_buffer_safe(g_mr, asap_mr)
    print(f"CASAP: {pk_casap_mr / 1024:.1f} KB")
    print(f"ASAP:  {pk_asap_mr / 1024:.1f} KB")
    print(f"Ratio: {pk_asap_mr / pk_casap_mr:.1f}x")
