# SDF-Edge — Artifact

**Paper:** *SDF-Edge: Synchronous Dataflow Formalism for Verifiable Neural
Network Inference on Edge Devices* (CASES 2026 / IEEE TCAD).

This artifact reproduces every table, buffer result, SDF3 comparison, framework
scalability measurement, and proof certificate in the paper, plus the bare-metal
Cortex-M7 validation. **Part A is pure Python 3 (standard library only) and runs
the entire experiment suite in under 30 seconds with no external dependencies.**

> **Patent notice.** The CASAP capacity-constrained scheduling method described
> in the paper and implemented here is the subject of a **pending U.S.
> provisional patent application; all patent rights are reserved.** The code is
> made available for academic evaluation, reproduction, and research use under
> the terms in [`LICENSE`](LICENSE). Reproducing the artifact for evaluation is
> expressly permitted; commercial use of the patented method requires a separate
> license.

---

## 1. Quick start

See [`INSTALL.md`](INSTALL.md) for setup (Part A needs only Python ≥ 3.8) and
[`REQUIREMENTS.md`](REQUIREMENTS.md) for the full environment spec.

```bash
cd src
python run_experiments.py     # reproduces Tables 2–8 + regenerates all certificates
```

---

## 2. Repository layout

```
src/                          Core framework (Python 3, stdlib only)
  sdf_core.py                 SDF graph, repetition vector, q_max, CASAP/ASAP buffering, timing
  models.py                   The 5 NN dataflow models (MCUNet, MobileNetV2, MNetV2-MR, EffNet-B0, ViT-Tiny)
  monte_carlo.py              WCET / sensitivity analysis
  run_experiments.py          Master runner — reproduces every paper table + writes certificates
  buffer_proof.py             CASAP-vs-ASAP buffer bound (headline 10.3x result)
  sdf3_comparison.py          Reimplemented SDF3 baseline; exports SDF3-loadable XML
  sdf3_benchmark_validation.py  Validation on 7 real SDF3 benchmark graphs
certificates/                 Machine-checkable proof certificates (JSON), one per model
sdf3_comparison/              SDF3 XML graphs + benchmark inputs for independent cross-checking
firmware/                     Bare-metal STM32H743 (Cortex-M7) firmware + Renode script + reference UART logs
response_revision/            Camera-ready sources and reviewer-response materials (not needed for reproduction)
```

---

## 3. What each command reproduces (claim → command → expected result)

All commands are run from the `src/` directory.

| Paper claim | Command | Expected key output |
|-------------|---------|---------------------|
| **Tables 2–8** (graph properties, static schedule, buffer analysis, DAG-underflow counterexample, scalability) + all 5 certificates | `python run_experiments.py` | Prints each table; ends with `ALL EXPERIMENTS COMPLETE`; regenerates `certificates/*.json` |
| **10.3× buffer reduction**, CASAP vs ASAP on 26-actor MNetV2-MR (1.5 KB vs 16.0 KB) | `python buffer_proof.py` | `CASAP: 1.5 KB`, `ASAP: 16.0 KB`, `Ratio: 10.3x` |
| **2.4× over SDF3's best** capacity-constrained result + SDF3-loadable XML export | `python sdf3_comparison.py` | Comparison table; writes `sdf3_comparison/mnetv2_mr.xml` |
| **CASAP validated on 7 SDF3 benchmark graphs** (improves on 5/7, 1.5×–18.4×) | `python sdf3_benchmark_validation.py` | Per-benchmark `ASAP=…, CASAP=…, ratio=…`; ends "correctly handles all 7 SDF3 benchmark graphs" |

### Proof certificates

`run_experiments.py` regenerates one JSON certificate per model under
`certificates/`. Each records the repetition vector, `q_max`, total firings,
the proven buffer bound (Tier 1), and verification timing (Tier 2). Diff the
regenerated files against the committed ones to confirm bit-stable
reproduction, e.g.:

```bash
git diff --stat certificates/
```

### DAG counterexample (Section 5.9)

`run_experiments.py` also demonstrates the silent-data-corruption scenario: a
naive round-robin DAG schedule triggers `TOKEN UNDERFLOW DETECTED` on
`stride2a->dw_s2a`, which SDF-Edge's Tier-2 verification catches. This is the
correctness argument, reproduced live.

---

## 4. Bare-metal validation (optional, Cortex-M7 via Renode)

No physical board required — the STM32H743 target is emulated in Renode.

```bash
cd firmware
renode sdf_edge_sim.resc
```

The firmware emits its schedule, per-actor timing, and peak SRAM over UART.
Reference captures are provided in `firmware/uart_output.txt` and
`firmware/renode_output.log`. These correspond to the paper's on-device numbers:
**5.3× buffer reduction on the 8-actor model, < 7.4% timing error, < 2.4 KB SRAM
error** vs. the framework prediction. To rebuild the binary from source, run
`make` (needs `arm-none-eabi-gcc`); a prebuilt `.elf` is included otherwise.

---

## 5. Reproduction environment

- Validated on **CPython 3.12.1**, Windows 11 and Ubuntu 22.04.
- Part A: standard library only, < 30 s wall-clock, < 256 MB RAM, offline.
- Part B: Renode ≥ 1.14.

See [`REQUIREMENTS.md`](REQUIREMENTS.md) and [`STATUS.md`](STATUS.md) (badge
justifications).

---

## 6. License and patent

Source-available for academic evaluation and research use under [`LICENSE`](LICENSE).
The CASAP method is **patent pending (U.S. provisional application); all patent
rights reserved.** See `LICENSE` for the full grant, restrictions, and patent
reservation.
