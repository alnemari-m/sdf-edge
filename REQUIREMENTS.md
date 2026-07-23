# REQUIREMENTS

Environment needed to evaluate the SDF-Edge artifact. The artifact is split
into two independent parts; **Part A alone reproduces every table, figure, and
proof certificate in the paper** and needs nothing beyond a Python interpreter.
Part B is the optional bare-metal validation.

---

## Part A — Formal framework and experiments (required, self-contained)

| Item | Requirement |
|------|-------------|
| OS | Any (Linux, macOS, or Windows). Tested on Windows 11 and Ubuntu 22.04. |
| Interpreter | **CPython >= 3.8** (validated on 3.12.1). |
| Python packages | **None.** Standard library only — see `requirements.txt`. |
| Hardware | Any x86-64 or ARM host. No GPU, no accelerator, no special hardware. |
| Disk | < 20 MB. |
| RAM | < 256 MB. |
| Network | Not required (fully offline). |
| Wall-clock | Full experiment suite completes in **under 30 seconds** on a laptop. |

This is the part reviewers should use for the **Functional**, **Reusable**, and
**Results Reproduced** badges. It has no external dependencies precisely so that
the hardware/software environment is guaranteed compatible with the evaluation
process (per the CASES AE compatibility requirement).

## Part B — Bare-metal firmware validation (optional)

Reproduces the on-device (Cortex-M7) timing and SRAM measurements in the paper.
It does **not require a physical board** — it runs on the Renode emulator, so
any reviewer can execute it.

| Item | Requirement |
|------|-------------|
| Emulator | **Renode >= 1.14** (https://renode.io) — free, open source. |
| Cross-compiler (only to rebuild) | `arm-none-eabi-gcc` (GNU Arm Embedded Toolchain). Optional: a prebuilt `sdf_edge_firmware.elf` is included. |
| Target | STM32H743 (Cortex-M7). Modeled in Renode; **no physical hardware needed**. |
| OS | Renode runs on Linux, macOS, and Windows. |

If a reviewer prefers not to install Renode, the reference outputs
(`firmware/uart_output.txt`, `firmware/renode_output.log`) are included so the
claimed on-device numbers can be inspected directly.

## Licensing note relevant to evaluation

The code is source-available for academic evaluation and reproduction under the
terms in `LICENSE`. **The CASAP scheduling method is the subject of a pending
U.S. provisional patent application; all patent rights are reserved** (see
`LICENSE`). Running and reproducing the artifact for evaluation is expressly
permitted.
