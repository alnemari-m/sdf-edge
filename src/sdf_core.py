"""
SDF-Edge: Synchronous Dataflow formalism for neural network inference on edge devices.
Core module: topology matrix, repetition vector, buffer analysis, scheduling.
"""

from fractions import Fraction
from math import gcd
from functools import reduce
from collections import deque
import json
import time


def lcm(a, b):
    return abs(a * b) // gcd(a, b)


def lcm_list(lst):
    return reduce(lcm, lst, 1)


class SDFActor:
    """An SDF actor representing a DNN layer or fused block."""

    def __init__(self, name, exec_time_us, power_mw=350.0, layer_type="conv"):
        self.name = name
        self.exec_time_us = exec_time_us  # microseconds
        self.power_mw = power_mw           # dynamic power in mW
        self.layer_type = layer_type       # for scenario selection


class SDFEdge:
    """An SDF edge with production and consumption rates."""

    def __init__(self, src, dst, prod=1, cons=1, token_size_bytes=1024, initial_tokens=0):
        self.src = src
        self.dst = dst
        self.prod = prod
        self.cons = cons
        self.token_size_bytes = token_size_bytes
        self.initial_tokens = initial_tokens


class SDFGraph:
    """Complete SDF graph with topology matrix, repetition vector, and analysis."""

    def __init__(self, name):
        self.name = name
        self.actors = []
        self.edges = []
        self._actor_index = {}

    def add_actor(self, actor):
        self._actor_index[actor.name] = len(self.actors)
        self.actors.append(actor)

    def add_edge(self, edge):
        self.edges.append(edge)

    def actor_idx(self, name):
        return self._actor_index[name]

    @property
    def num_actors(self):
        return len(self.actors)

    @property
    def num_edges(self):
        return len(self.edges)

    def topology_matrix(self):
        """Build the topology matrix Gamma[edge, actor]."""
        V = self.num_actors
        E = self.num_edges
        gamma = [[0] * V for _ in range(E)]
        for i, e in enumerate(self.edges):
            src_idx = self.actor_idx(e.src)
            dst_idx = self.actor_idx(e.dst)
            gamma[i][src_idx] = e.prod
            gamma[i][dst_idx] = -e.cons
        return gamma

    def compute_repetition_vector(self):
        """
        BFS-based exact repetition vector using rational arithmetic.
        Algorithm 1 from Lee & Messerschmitt (1987).
        """
        V = self.num_actors
        # Build adjacency: actor_name -> [(neighbor_name, prod, cons)]
        adj = {a.name: [] for a in self.actors}
        for e in self.edges:
            adj[e.src].append((e.dst, e.prod, e.cons))
            adj[e.dst].append((e.src, e.cons, e.prod))  # reverse edge

        q = {self.actors[0].name: Fraction(1)}
        visited = {self.actors[0].name}
        queue = deque([self.actors[0].name])

        while queue:
            u = queue.popleft()
            for v_name, p, c in adj[u]:
                if v_name not in visited:
                    # q[v] = (p / c) * q[u]
                    q[v_name] = Fraction(p, c) * q[u]
                    visited.add(v_name)
                    queue.append(v_name)

        if len(visited) != V:
            raise ValueError(f"Graph is not connected: visited {len(visited)}/{V} actors")

        # Clear fractions: multiply by LCM of denominators
        denoms = [q[a.name].denominator for a in self.actors]
        scale = lcm_list(denoms)
        q_int = {name: int(val * scale) for name, val in q.items()}

        # Reduce by GCD
        g = reduce(gcd, q_int.values())
        q_int = {name: val // g for name, val in q_int.items()}

        # Verify: Gamma * q = 0
        gamma = self.topology_matrix()
        q_vec = [q_int[a.name] for a in self.actors]
        for i, row in enumerate(gamma):
            dot = sum(row[j] * q_vec[j] for j in range(V))
            if dot != 0:
                raise ValueError(f"Rate inconsistency on edge {i}: Gamma*q[{i}] = {dot}")

        return q_int

    def rank_topology(self):
        """Compute rank of topology matrix via Gaussian elimination with exact fractions."""
        gamma = self.topology_matrix()
        E = len(gamma)
        V = len(gamma[0]) if E > 0 else 0
        # Convert to Fraction
        mat = [[Fraction(gamma[i][j]) for j in range(V)] for i in range(E)]

        rank = 0
        for col in range(V):
            # Find pivot
            pivot = None
            for row in range(rank, E):
                if mat[row][col] != 0:
                    pivot = row
                    break
            if pivot is None:
                continue
            mat[rank], mat[pivot] = mat[pivot], mat[rank]
            # Eliminate
            for row in range(E):
                if row != rank and mat[row][col] != 0:
                    factor = mat[row][col] / mat[rank][col]
                    for j in range(V):
                        mat[row][j] -= factor * mat[rank][j]
            rank += 1

        return rank

    def is_consistent(self):
        """Check if rank(Gamma) = |V| - 1."""
        return self.rank_topology() == self.num_actors - 1

    def compute_asap_schedule(self):
        """Compute ASAP (as-soon-as-possible) schedule."""
        q = self.compute_repetition_vector()

        # Build firing sequence: topological order, repeat q[i] times
        # Build DAG adjacency
        in_degree = {a.name: 0 for a in self.actors}
        adj = {a.name: [] for a in self.actors}
        for e in self.edges:
            adj[e.src].append(e.dst)
            in_degree[e.dst] += 1

        # Topological sort (Kahn's algorithm)
        topo = []
        queue = deque([a.name for a in self.actors if in_degree[a.name] == 0])
        while queue:
            u = queue.popleft()
            topo.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        # ASAP schedule: fire each actor q[i] times in topological order
        schedule = []
        for name in topo:
            for firing in range(q[name]):
                schedule.append((name, firing))

        return schedule

    def compute_alap_schedule(self):
        """Compute ALAP (as-late-as-possible) schedule for buffer minimization."""
        q = self.compute_repetition_vector()

        # Reverse topological order
        in_degree = {a.name: 0 for a in self.actors}
        adj = {a.name: [] for a in self.actors}
        rev_adj = {a.name: [] for a in self.actors}
        for e in self.edges:
            adj[e.src].append(e.dst)
            rev_adj[e.dst].append(e.src)
            in_degree[e.dst] += 1

        # Topological sort
        topo = []
        queue = deque([a.name for a in self.actors if in_degree[a.name] == 0])
        while queue:
            u = queue.popleft()
            topo.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        # ALAP: reverse topological, fire each actor q[i] times
        schedule = []
        for name in reversed(topo):
            for firing in range(q[name]):
                schedule.append((name, firing))

        return schedule

    def analyze_buffers(self, schedule=None):
        """
        Simulate schedule and track peak SRAM usage.
        Returns (peak_bytes, per_edge_max).
        """
        if schedule is None:
            schedule = self.compute_asap_schedule()

        q = self.compute_repetition_vector()

        # Track token counts on each edge
        edge_tokens = {}
        for i, e in enumerate(self.edges):
            edge_tokens[i] = e.initial_tokens

        peak_sram = 0
        per_edge_max = [0] * self.num_edges

        for actor_name, firing_idx in schedule:
            # Consume tokens from input edges
            for i, e in enumerate(self.edges):
                if e.dst == actor_name:
                    edge_tokens[i] -= e.cons
                    if edge_tokens[i] < 0:
                        raise RuntimeError(
                            f"Token underflow on edge {e.src}->{e.dst} "
                            f"(actor {actor_name}, firing {firing_idx}): "
                            f"tokens={edge_tokens[i]+e.cons}, need={e.cons}"
                        )

            # Produce tokens on output edges
            for i, e in enumerate(self.edges):
                if e.src == actor_name:
                    edge_tokens[i] += e.prod

            # Track peak
            current_sram = sum(
                edge_tokens[i] * self.edges[i].token_size_bytes
                for i in range(self.num_edges)
            )
            peak_sram = max(peak_sram, current_sram)
            for i in range(self.num_edges):
                per_edge_max[i] = max(per_edge_max[i], edge_tokens[i])

        return peak_sram, per_edge_max

    def analytical_buffer_bound(self):
        """Schedule-independent analytical worst-case buffer bound."""
        q = self.compute_repetition_vector()
        total = 0
        per_edge = []
        for e in self.edges:
            bound = q[e.src] * e.prod + e.initial_tokens
            per_edge.append(bound * e.token_size_bytes)
            total += bound * e.token_size_bytes
        return total, per_edge

    def compute_timing(self):
        """Static iteration time: T_iter = sum(q_i * t_i)."""
        q = self.compute_repetition_vector()
        t_us = sum(q[a.name] * a.exec_time_us for a in self.actors)
        return t_us

    def compute_energy(self, p_idle_mw=60.0):
        """Energy per iteration: E = E_dyn + E_idle."""
        q = self.compute_repetition_vector()
        t_iter_us = self.compute_timing()

        e_dyn_uj = sum(q[a.name] * a.power_mw * a.exec_time_us / 1000.0 for a in self.actors)
        e_idle_uj = p_idle_mw * t_iter_us / 1000.0

        return {
            "e_total_uj": e_dyn_uj + e_idle_uj,
            "e_dyn_uj": e_dyn_uj,
            "e_idle_uj": e_idle_uj,
            "t_iter_us": t_iter_us,
            "p_avg_mw": (e_dyn_uj + e_idle_uj) / (t_iter_us / 1000.0) if t_iter_us > 0 else 0,
        }

    def generate_proof_certificate(self, schedule=None):
        """Generate JSON proof certificate."""
        t_start = time.perf_counter()

        q = self.compute_repetition_vector()
        rank = self.rank_topology()
        consistent = self.is_consistent()

        if schedule is None:
            schedule = self.compute_asap_schedule()

        peak_sram, _ = self.analyze_buffers(schedule)
        analytical_bound, _ = self.analytical_buffer_bound()
        timing = self.compute_timing()
        energy = self.compute_energy()

        t_elapsed = time.perf_counter() - t_start

        cert = {
            "model": self.name,
            "num_actors": self.num_actors,
            "num_edges": self.num_edges,
            "rank_gamma": rank,
            "consistent": consistent,
            "repetition_vector": q,
            "total_firings": sum(q.values()),
            "q_max": max(q.values()),
            "tier1": {
                "rate_consistency": "PROVEN" if consistent else "FAILED",
                "buffer_bound_bytes": analytical_bound,
                "timing_us": timing,
            },
            "schedule_specific": {
                "peak_sram_bytes": peak_sram,
                "schedule_length": len(schedule),
            },
            "energy": energy,
            "verification_time_ms": t_elapsed * 1000,
        }
        return cert


def qmax_closed_form(k, s=2, N=1):
    """Proposition 1: q_max = s^k * N for k stride-s stages + global pool over N tiles."""
    return (s ** k) * N
