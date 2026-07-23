"""
SDF-Edge: Scenario-Aware SDF Monte Carlo simulation.
Per-layer execution scenarios with probabilistic timing analysis.
"""

import random
import math
from sdf_core import SDFGraph


# Per-layer-type scenario probabilities (Table 3 in paper)
SCENARIO_PROBS = {
    "conv": {"nominal": 0.89, "cache_miss": 0.07, "dma_stall": 0.03, "worst_case": 0.01},
    "dw":   {"nominal": 0.92, "cache_miss": 0.05, "dma_stall": 0.02, "worst_case": 0.01},
    "fc":   {"nominal": 0.85, "cache_miss": 0.09, "dma_stall": 0.04, "worst_case": 0.02},
}

# Multipliers for each scenario
SCENARIO_MULTIPLIERS = {
    "nominal": 1.0,
    "cache_miss": 2.5,     # AXI bus penalty
    "dma_stall": 1.8,      # DMA contention
    "worst_case": 3.5,     # Combined worst case
}

# Thermal throttling model
THERMAL_THRESHOLD_MS = 50.0   # Clock drops to 80% after this
THERMAL_FACTOR = 1.25         # 1/0.8 = 1.25x slower

# RTOS / ISR overhead
RTOS_TICK_MS = 1.0
RTOS_OVERHEAD_US = 3.0
ISR_RATE_HZ = 200
ISR_OVERHEAD_US = 6.0


def sample_scenario(layer_type):
    """Sample an execution scenario for a given layer type."""
    probs = SCENARIO_PROBS.get(layer_type, SCENARIO_PROBS["conv"])
    r = random.random()
    cumulative = 0.0
    for scenario, prob in probs.items():
        cumulative += prob
        if r <= cumulative:
            return scenario
    return "nominal"


def simulate_iteration(graph, thermal_threshold_ms=THERMAL_THRESHOLD_MS):
    """
    Simulate one inference iteration with scenario-aware timing.
    Returns total iteration time in microseconds.
    """
    q = graph.compute_repetition_vector()
    schedule = graph.compute_asap_schedule()

    total_us = 0.0
    for actor_name, firing_idx in schedule:
        actor = next(a for a in graph.actors if a.name == actor_name)
        scenario = sample_scenario(actor.layer_type)
        t = actor.exec_time_us * SCENARIO_MULTIPLIERS[scenario]

        # Thermal throttling: if accumulated time exceeds threshold
        if total_us / 1000.0 > thermal_threshold_ms:
            t *= THERMAL_FACTOR

        total_us += t

    # Add ISR overhead
    iter_ms = total_us / 1000.0
    num_isr = int(iter_ms * ISR_RATE_HZ / 1000.0)
    total_us += num_isr * ISR_OVERHEAD_US

    # Add RTOS tick overhead
    num_ticks = int(iter_ms / RTOS_TICK_MS)
    total_us += num_ticks * RTOS_OVERHEAD_US

    return total_us


def compute_wcet(graph, k_safety=1.2):
    """Analytical WCET bound (Eq. 7 in paper)."""
    q = graph.compute_repetition_vector()

    # All actors at worst case
    wcet_sum = sum(
        q[a.name] * a.exec_time_us * SCENARIO_MULTIPLIERS["worst_case"]
        for a in graph.actors
    )

    # Thermal throttling on entire inference
    iter_ms = wcet_sum / 1000.0
    if iter_ms > THERMAL_THRESHOLD_MS:
        # Only portion after threshold gets throttled
        pre_thermal_us = THERMAL_THRESHOLD_MS * 1000
        post_thermal_us = wcet_sum - pre_thermal_us
        wcet_sum = pre_thermal_us + post_thermal_us * THERMAL_FACTOR

    # Safety factor
    wcet_sum *= k_safety

    # Worst-case ISR burst
    iter_ms = wcet_sum / 1000.0
    num_isr = int(iter_ms * ISR_RATE_HZ / 1000.0) + 10  # +10 burst
    wcet_sum += num_isr * ISR_OVERHEAD_US

    return wcet_sum


def monte_carlo_analysis(graph, n_iterations=10000, deadline_ms=100.0,
                         seed=42, thermal_threshold_ms=THERMAL_THRESHOLD_MS):
    """
    Run Monte Carlo simulation and return analysis results.
    """
    random.seed(seed)

    latencies = []
    for _ in range(n_iterations):
        t_us = simulate_iteration(graph, thermal_threshold_ms)
        latencies.append(t_us / 1000.0)  # Convert to ms

    latencies.sort()
    n = len(latencies)
    misses = sum(1 for t in latencies if t > deadline_ms)

    # Clopper-Pearson 95% confidence bound for 0 misses
    gamma = 0.95
    if misses == 0:
        cp_bound = 1.0 - (1.0 - gamma) ** (1.0 / n)
    else:
        cp_bound = misses / n  # Simplified for non-zero case

    wcet = compute_wcet(graph) / 1000.0  # ms

    return {
        "n_iterations": n,
        "t_static_ms": graph.compute_timing() / 1000.0,
        "t_mean_ms": sum(latencies) / n,
        "t_p50_ms": latencies[int(n * 0.50)],
        "t_p99_ms": latencies[int(n * 0.99)],
        "t_max_ms": latencies[-1],
        "t_wcet_ms": wcet,
        "deadline_ms": deadline_ms,
        "miss_count": misses,
        "miss_rate": misses / n,
        "clopper_pearson_bound": cp_bound,
        "latencies": latencies,
    }


def sensitivity_analysis(graph, param_name="cache_miss_rate",
                         values=None, n_iterations=10000, deadline_ms=100.0):
    """Sweep a parameter and return miss rates."""
    if values is None:
        values = [0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

    results = []
    original_probs = {k: dict(v) for k, v in SCENARIO_PROBS.items()}

    for val in values:
        # Modify cache miss rate for all layer types proportionally
        for lt in SCENARIO_PROBS:
            base = original_probs[lt].copy()
            base["cache_miss"] = val
            # Adjust nominal to keep sum = 1
            remainder = base["dma_stall"] + base["worst_case"]
            base["nominal"] = max(0.0, 1.0 - val - remainder)
            SCENARIO_PROBS[lt] = base

        mc = monte_carlo_analysis(graph, n_iterations, deadline_ms)
        results.append({
            "param_value": val,
            "miss_rate": mc["miss_rate"],
            "t_p99_ms": mc["t_p99_ms"],
            "t_mean_ms": mc["t_mean_ms"],
        })

    # Restore original
    for lt in SCENARIO_PROBS:
        SCENARIO_PROBS[lt] = original_probs[lt]

    return results
