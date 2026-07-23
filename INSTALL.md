# INSTALL

Setup and a basic smoke test. Full reproduction steps are in `README.md`.

---

## Part A — Framework (required, ~1 minute total)

No installation is needed beyond a Python 3 interpreter; there are no
third-party packages.

### 1. Get a Python interpreter

```bash
python --version      # must report 3.8 or newer (tested on 3.12.1)
```

### 2. (Optional) create a clean virtual environment

```bash
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1

pip install -r requirements.txt   # succeeds; installs nothing (stdlib only)
```

### 3. Smoke test (should finish in a few seconds)

```bash
cd src
python buffer_proof.py
```

Expected tail of output — the paper's headline buffer result:

```
MNetV2-MR: CASAP vs ASAP
CASAP: 1.5 KB
ASAP:  16.0 KB
Ratio: 10.3x
```

If you see `Ratio: 10.3x`, the framework is working. Proceed to `README.md`
for the full table-by-table reproduction.

---

## Part B — Firmware validation (optional)

### 1. Install Renode

Download from https://renode.io (packages for Linux, macOS, Windows), or use
the official Docker image `renode/renode`.

### 2. Run the emulated Cortex-M7 target

```bash
cd firmware
renode sdf_edge_sim.resc      # boots STM32H743 model, runs sdf_edge_firmware.elf
```

The firmware prints its schedule, per-actor timing, and peak SRAM over UART.
Compare against the reference capture in `firmware/uart_output.txt`.

### 3. (Optional) rebuild the firmware from source

Requires `arm-none-eabi-gcc`:

```bash
cd firmware
make            # produces sdf_edge_firmware.elf from sdf_edge_firmware.c
```

A prebuilt `sdf_edge_firmware.elf` is included, so this step is only needed if
you want to verify the binary from source.
