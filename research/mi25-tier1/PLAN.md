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

Executed 2026-11-10 on the operator's box itself (the live IQ2_XS service was paused for the
sweep and restored). Executor: `tools/ab_engine.py`, config `/data/install/tune/split27.json`,
env `STRATA_COMPONENTS_DIR=/data/components`. Full tables and per-request rows:
`bench/results/2026-11-10-mi25-tier1/README.md` (+ jsonl/log).

- **`STRATA_SPLIT_OVERLAP=1`: adopt.** Beat base in every decode cell over 3 alternating
  rounds (medians: 39.5 vs 35.0, 21.5 vs 20.2, 18.6 vs 15.8, 22.3 vs 21.4 tok/s) and both
  chat prompt medians, with zero hangs — the strongest and cleanest signal of the sweep.
  Config change only (`STRATA_SET="... overlap"` compose env or the preset's env table).
- **`STRATA_MMVQ_WAVE=0`: no case.** Ties with base within the run spread.
- **`STRATA_EXP_MODE=2`: no decode case, one prefill hint** (long prompt 78.6 vs 56.5
  tok/s vs base's missing-cell-affected 1 round; not called — needs a prefill A/B).
- Run 1 lost one cell (base round 1 long) to the file-tier watchdog timeout (#1407) and a
  503; the jsonl marks it implicitly, the README documents it; no number was invented.
- The `--spec` arm stays deferred: the live drafter here is DFlash2, whose window cut is
  governed by `--draft-conf`, not `--spec`.
- rocBLAS calibration table (Tier-1 item 4): build path done, calibration done, end-to-end
  slower — not adopted, FP16 route stays off. The `gfx906` cmake backend now compiles the
  `STRATA_ROCBLAS_AVAILABLE` consumer and `tune_rocblas`; the table made on this box is
  `tools/hip/gfx900-rocblas-5.5.0.9811f2ea52.txt` (111 rows tuned of 112). Details:
  `docs/AMD_GFX900.md` ("A rocBLAS table made on the MI25s"), nums in
  `bench/results/2026-11-10-mi25-tier2/README.md`.

## Tier-2 outcome (2026-11-10, later)

Raw numbers and cross-references: `bench/results/2026-11-10-mi25-tier2/README.md`.

- **Item 7 came in as a plain env arm instead of a code change**: the engine already gates
  the CPU share per arch family; on HIP + layer split nothing turns it on, but setting
  `STRATA_PREFILL_CPU_SHARE=auto` engages it as-is (prefill.cpp:3004). With the explicit
  arm the 2-round A/B won every median — long prompt 91.8 vs 61.4 tok/s, decode unchanged,
  no hangs — so the code-level gate was never written. Adopted in compose.
- **Item 6 struck for now**: `STRATA_IO_PF_SCHED` only exercises the file tier, and the
  resident config holds 94-99.9% cache hits; the budget would measure a path this workload
  barely enters. Revisit only if the config loses residency.
- **Item 5's iq2_xs gate/up route** (gu type 17): implemented, bitwise-parity-clean,
  default-inert (commit 0b3fdcf0) — and dead code for this model: every expert layer of the
  GSQ-RCO IQ2_XS set carries iq2_s / iq2_xxs / iq1_m gate/up with q2_0 down (all 47 shard-1
  layers probed with native_expert_bench; shard 2 holds no expert tensors). The name of the
  pack and its gate/up formats are not the same thing.
- gfx900 build tree lives at `build-gfx900/` (rocm 7.14 wheels from /opt/venv's
  `_rocm_sdk_devel`); the recipe is in the session notes, the parity/bench harness targets
  built clean (`native_grouped_parity: 0 failures`).
- **`STRATA_EXP_MODE=2` (the tier-1 prefill hint), measured on top of the adopted CPU share:
  no case.** 2 rounds, 4 server starts; chat1/chat2/long prompt medians land within the
  A/B noise, and round 1 went the other way on two cells. Not adopted; the default path
  stays. Numbers: `bench/results/2026-11-10-mi25-tier2/README.md` (share-only vs
  share+EXP_MODE=2 table).
