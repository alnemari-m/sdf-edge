# STATUS — Badges Requested

We apply for all four CASES 2026 artifact badges. Justifications below.

---

## Artifacts Available

The complete artifact — framework source, the five neural-network dataflow
models, all proof certificates, the SDF3 comparison and benchmark inputs, and
the Cortex-M7 firmware with its Renode setup and reference logs — is deposited
in a permanent, publicly accessible archival repository with a DOI (Zenodo).
The DOI resolves to the exact evaluated version. Nothing required to reproduce
the paper's results is withheld.

## Artifacts Evaluated — Functional

The artifact is complete, documented, and exercisable:

- **Complete** — every quantitative claim in the paper maps to a script in
  `src/` (see the claim→command table in `README.md`).
- **Exercisable** — `python run_experiments.py` runs the entire suite in under
  30 seconds using only the Python standard library (no external packages, no
  special hardware), so the evaluation environment is guaranteed compatible.
- **Documented** — `README.md`, `INSTALL.md`, and `REQUIREMENTS.md` give
  step-by-step setup, a smoke test, and expected outputs.
- **Consistent** — the shipped proof certificates in `certificates/` are exactly
  what the runner regenerates (`git diff --stat certificates/` shows no change).

## Artifacts Evaluated — Reusable

The artifact is structured for adaptation beyond the paper:

- Clean module separation (`sdf_core`, `models`, `monte_carlo`, comparison and
  validation drivers) with a documented public surface.
- New models are added by describing a dataflow graph in `models.py`; the
  framework then derives repetition vector, `q_max`, CASAP/ASAP buffers,
  schedule, timing, and a proof certificate automatically.
- Interoperates with external tooling: `sdf3_comparison.py` exports standard
  **SDF3 XML** (`sdf3_comparison/mnetv2_mr.xml`) that loads directly into the
  independent SDF3 `sdf3analysis-sdf` tool for third-party cross-verification.
- `sdf3_benchmark_validation.py` already applies the method to seven real SDF3
  benchmark graphs (4–22 actors, `q_max` up to 5292), demonstrating reuse on
  inputs outside the paper's NN models.

## Results Validated — Reproduced

Every headline result is reproducible from this artifact on a commodity machine:

| Result in paper | Reproduced by | Value |
|-----------------|---------------|-------|
| Buffer reduction, MNetV2-MR (26 actors) | `buffer_proof.py` | **10.3×** (1.5 KB vs 16.0 KB) |
| Improvement over SDF3 best capacity-constrained | `sdf3_comparison.py` | **2.4×** |
| SDF3 benchmark sweep | `sdf3_benchmark_validation.py` | improves on **5/7**, range **1.5×–18.4×** |
| Graph properties / schedule / scalability (Tables 2–8) | `run_experiments.py` | matches paper tables |
| Proof certificates (5 models) | `run_experiments.py` | regenerated bit-stable |
| On-device timing / SRAM (Cortex-M7, 8-actor) | Renode `sdf_edge_sim.resc` | **5.3×**, **< 7.4%** timing error, **< 2.4 KB** SRAM error |

The on-device measurements are reproducible **without physical hardware** via
the Renode STM32H743 model, with reference UART captures included for direct
comparison.

---

## Note on licensing

The artifact is source-available for academic evaluation and research
(`LICENSE`). The CASAP method is the subject of a **pending U.S. provisional
patent application; all patent rights are reserved.** This does not restrict
evaluation or reproduction of the artifact, which the license expressly permits.
