# 2x MI25 (gfx900) + IQ2_XS: tiered optimization plan and Tier-1 A/B log

Rig: 2x AMD Instinct MI25 (gfx900, 16 GB HBM2 each), 32 GB DDR4, SATA SSD, engine
0.1.58-linux-x86_64-hip-gfx900. Model: Qwen3.8-Flash-Next GSQ-RCO IQ2_XS (full-expert set, qwen
family), 262-272k context, int8 KV. Current best measured config
(`docs/AMD_GFX900_MI25_REPORT.md`, `/data/install/tune/SWEEP.md`):
`--resident-budget-gib 16 --layer-split 27` — 260-261 tok/s prefill at 262k, 0 hangs;
decode 22-25 tok/s, GPU-bound, expert-cache hit 94-99.9%. The live service also runs the
DFlash2 drafter (auto-picked): `--draft dflash2 --draft-conf 0.35`, windows up to 8 tokens.

Already settled on this box, do not re-sweep:
- `--pcie-frac N>0` / "light" arm: 10% slower than split27 alone.
- "tuned" arm (`--pipeline-windows 2 --trim-stage-weights --kv-resident 16384`): 1 watchdog
  hang at layer 18 on the 90% needle.
- `STRATA_SPLIT_OWN` / `STRATA_PREFILL_HELP`: #2461, costs decode more than prompt gains when
  experts are (nearly) resident.
- `--vision off`: no GPU picture reader in this build.

Why gfx900 is special: no packed byte dot (`v_dot4` is gfx906+) — dp4a is a 5-instruction
software sequence (`third_party/strata-ggml/ggml/src/ggml-cuda/common.cuh:726-741`); no WMMA;
wave64; hipBLASLt is not used (plain hipBLAS/rocBLAS, FP16 GEMMs see
`src/prefill/gemm.cu:1266`).

## Tier 1 — config-only A/Bs (no code; being run now)

1. **`STRATA_SPLIT_OVERLAP=1`** — overlapped hand-off between the two stages. Measured +33%
   decode doc on 2x RX 7900 XTX (`docs/MULTI_GPU.md:292`); never tried on gfx900. Mutually
   exclusive with `--pipeline-windows 2` (which this box rejects: hang).
2. **MMVQ layout**: `STRATA_MMVQ_WAVE=0` restores the CUDA per-half-warp layout instead of the
   one-wavefront-per-row 64-lane butterfly (`docs/AMD_HIP.md:584`).
3. **Grouped expert decode**: `STRATA_EXP_MODE=2` (CUDA layout). The wave64 default (Mode 7)
   handles signs as `dp4a(g^m,u) − dp4a(m,u)` — 2x the dp4a emulation cost where dp4a is
   software, so the layout trade may favor `=2` on gfx900 (unconfirmed static analysis).
4. **rocBLAS tuning table** (`tools/`, `tune_rocblas`): calibrate prompt dense GEMMs for
   gfx900; plain hipBLAS is untuned here. Run later this box time permitting.

Not applicable as written: a bare `--spec` sweep targets the MTP path; this box's live drafter
is DFlash2 (`--draft-conf` controls its window cost). Deferred.

## Tier 2 — code-level, targeted at this arch (unconfirmed static analysis)

5. Wave64-native MMVQ/vecdot for IQ2_XS that amortizes the software-dp4a emulation
   (deeper than `native_mmvq.cu:2362`'s row mapping). Needs the wave64 parity harnesses
   (`native_arithmetic_parity`, `cmake/ggml_tq_overlay_test.cmake`) before submission.
6. Replace the resident-budget workaround for the 262k prefill hang: bound file-tier inflight
   page-ins instead of pinning 16 GiB (the hang is tier starvation, issue #29 watchdog).
7. Gate `STRATA_PREFILL_CPU_SHARE` (off on HIP + layer split) at big chunks with a pool-rate
   rule.

## Constraints

Methodology per `docs/BENCHMARK_PROTOCOL.md` + the box's own `/data/install/tune/SWEEP.md`
protocol: one fresh engine per arm-round, arms alternated across rounds, medians compared,
cards cooled between sweeps, engine/serve logs kept. No measurement is submitted that was not
produced here. `tools/ab_engine.py` executes the arms (env deltas; the live service is paused
for the sweep and restored afterwards).

## Tier-1 runs (2026-11-10)

Run 1 — decode layout A/B (`--rounds 2`, config `/data/install/tune/split27.json`):

    base        : (no delta)
    mmvq0       : STRATA_MMVQ_WAVE=0
    expmode2    : STRATA_EXP_MODE=2

Run 2 — hand-off overlap (`--rounds 3`):

    base        : (no delta)
    overlap     : STRATA_SPLIT_OVERLAP=1

Results: appended below after the runs; raw jsonl under `bench/results/2026-11-10-mi25-tier1/`.

## Outcome (filled in after execution)

TBD
