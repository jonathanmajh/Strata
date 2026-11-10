# MI25 (2x gfx900) tier-2 A/B: prefill CPU share

Continues `bench/results/2026-11-10-mi25-tier1` and `research/mi25-tier1/PLAN.md`.
Rig: 2x MI25 (gfx900), the live GSQ-RCO IQ2_XS config (`/data/install/tune/split27.json`
in r16 + split27, plus the tier-1 winner `STRATA_SPLIT_OVERLAP=1`). Quick methodology,
`tools/ab_engine.py`, one engine per arm-round, arms alternated.

## Item: `STRATA_PREFILL_CPU_SHARE=auto` (adopted)

The engine keeps the CPU share off where it was not measured (HIP, multi-GPU, layer
split), so nothing on this box benefits unless the env names it. Armed explicitly
(`auto`: measured shares per layer, no fixed ratio), 2 rounds:

| request | base | share=auto |
|---|---|---|
| chat0 prompt | 34.5 | 43.2 |
| chat1 prompt | 35.8 | 37.8 |
| chat2 prompt | 41.2 | 40.7 |
| long prompt (6,345 tok) | 61.4 | **91.8** |
| chat decode (medians) | 19.7-35.8 | 22.6-37.8 |

Token counts match the tier-1 cells (34 / 27 / 29 / 6,345 prompt tokens). The share
arm won every median; the long prompt's read +49%; decode unchanged within noise; no
hangs. Adopted in `docker/compose/docker-compose.gfx900-mi25.yml`
(`STRATA_PREFILL_CPU_SHARE: auto`).

Limits: 2 rounds per arm (the win on the long prompt is the size tier-1 also measured,
so both rounds moving the same way makes it hard to lose to noise, but 6+-round
numbers this size were not produced); the giant 262k prompts were not re-run.

## Item: the rocBLAS solution table (made; retest verified the load; narrow verdict)

Built the calibration path for the wave64 build: the gfx906-compat cmake backend now discovers
rocBLAS and sets `STRATA_ROCBLAS_AVAILABLE`, compiles the same loader as the gfx103x builds, and
creates the `tune_rocblas` tool. The sweep ran on the MI25s against rocBLAS 5.5.0.9811f2ea52
(`tune_rocblas --tuning-out tools/hip/gfx900-rocblas-5.5.0.9811f2ea52.txt`): 112 rows, 101 with a
faster solution (1.05x-12x on the dense shapes). The table is committed to `tools/hip/`.

### First pass (2 rounds, component binary — superseded)

The first A/B measured the component binary, which does not carry the loader; its "FP16 + table"
arm was FP16-only with a dead table, and the tables below are what that produced. Recorded as
history, not as a verdict:

| median (2 rounds) | base | component FP16 |
|---|---|---|
| long decode/prompt | 22.5 / 82.0 | 14.3 / 50.9 |

### Retest (4 rounds, `build-gfx900/strata` for every arm — the reference now)

Arms: A = plain, C = `STRATA_HIP_PROMPT_F16=1` (FP16 route, no table), B = FP16 + the table
(`STRATA_ROCBLAS_TUNING=/workspace/tools/hip`, `STRATA_ROCBLAS_VERBOSE=1`). 4 rounds, Latin
rotation of arm order (A C B / C B A / B A C / A C B), warm-up request, VRAM sampled,
`tools/ab_engine.py`. Verification (per the reviewer):

- Broken-header negative control round: engine prints
  `prefill gemm: rocBLAS version mismatch: file=5.5.0.9812f2ea52 runtime=5.5.0.9811f2ea52 ...
  default kernels run` and `tuning summary tuned_launches=0 fallbacks=0` — the check can fire.
- Every B round log: 617 named solutions per round, `tuned_launches=2617 / 1766` per device,
  `fallbacks=1`. The remaining `no row` lines are shapes the table does not cover
  (`N=1280 K=2560`, GDN internals; the N=1280 row was never in the 14 dense shapes).
- Every C round log: prints only the two neutral `HIP device N ... STRATA_HIP_PROMPT_F16=1`
  lines, zero `rocBLAS` dispatch lines — loader off.
- VRAM: B ended 34-44 MiB higher than A/C on every arm-round; the FP16 route does NOT shrink
  the expert-cache budget. Decode drop is not a VRAM artifact.

Medians (n=4, spread in brackets; raw rows in `f16-3arm.jsonl` and per-request jsonl rows):

| cell | A base | C FP16 | B FP16+table |
|---|---|---|---|
| chat0 prompt | 47.5 [44.2..48.0] | 46.0 [42.2..47.8] | 45.6 [7.3..47.8] |
| chat1 prompt | 39.0 [38.0..39.0] | 41.9 [16.2..42.4] | 38.5 [15.6..42.3] |
| chat2 prompt | 45.0 [41.2..45.2] | 41.2 [36.2..45.3] | 41.4 [36.4..44.7] |
| chat0 decode | 37.1 | 37.2 | 35.9 |
| chat1 decode | 20.0 | 18.5 | 19.8 |
| chat2 decode | 20.4 | 19.1 | 18.0 |
| long (6,345 tok) prompt | 69.2 [49.5..81.3] | 99.3 [72.8..123.9] | 84.8 [63.8..136.2] |
| long (16,904 tok) prompt | 121.4 [114.9..155.5] | 142.6 [113.5..188.2] | **176.2** [134.0..196.1] |
| long1 decode | 21.6 | 22.4 | 21.1 |

What survived the retest:

- **The table is not slower in isolation on the cells where it engages heavily**: on the
  16,904-token long cell B beats C by +33.6 tok/s (176.2 vs 142.6), the one cell where
  the dense projections carry a real share of the read time.
- **The FP16 route itself is not a loss on the long cells on this binary** (C vs A: +30 on
  6,345 tokens, +21 on 16,904; chat cells a wash). The earlier "FP16 slower" reading does
  not hold: that run measured the component binary without the loader, and n=2.
- The 6,345-token cell for B is mixed (−14.5 vs C), the chat cells are within the
  round-to-round spread, and B's chat2 decode is −1.1 tok/s vs A (noise band for this arm).
- Two cells hit the outlier pattern the reviewer warned about: B round-3 chat0 prompt came
  in at 7.3 tok/s, and C round-1 chat1 at 16.2, B round-0/1 chat0 decode 20.1/35.0
  against r2/r3's 36.8/37.7. They are flagged, not used to draw conclusions.

**Adoption: not flipped in compose.** 262k-row outcome (one successful pass per arm,
2026-11-10): FP16-off read 250.0 tok/s (258,363 tokens, decode 14.6) vs FP16+table
266.1 tok/s (decode 15.7), VRAM unchanged, 22.6k/12.2k tuned launches with 1 fallback per
device. The 6.5% margin is consistent in sign but not wide; the 16,904-token cell
showed +33.6 tok/s over FP16-only. `docs/AMD_GFX900.md` carries the opt-in numbers. No
compose change: the FP16 numerics change wants a per-site owner decision plus the
from-scratch 262k stall watch on this box. The build gained a plumbing path; no config
default was touched. Sweep raw rows: `rocblas-sweep.log` (not committed).

### What would settle the adoption question

FP16-off 1-2x at 262k prompt vs FP16+table at the same; if the result holds by a wide
margin, `STRATA_HIP_PROMPT_F16=1` + `STRATA_ROCBLAS_TUNING=tools/hip` as an opt-in note in
the gfx900 doc becomes the right recommendation (measured on this box, this model).


## Item: the 262k prompt battery (adopted as opt-in evidence, not in compose)

See `docs/AMD_GFX900.md` for the opt-in note; numbers in `rocblas-table.md` ("262k
prompt" section). Two stalls recorded on this box's from-scratch 262k attempts; the
outcome traced to the file-tier watchdog's busy-ness sampling, not to the arm. Both
arms can pass. VRAM delta <45 MiB.

## Item: `STRATA_EXP_MODE=2` on top of the CPU share (not adopted)

The tier-1 sweep noticed a prefill hint on this arm (78.6 vs 56.5 tok/s) that was never
confirmed. After adopting the share, ran the same 2-round A/B with both arms on the
adopted production stack (`STRATA_SPLIT_OVERLAP=1` + `STRATA_PREFILL_CPU_SHARE=auto`;
executor: same protocol):

| median | share only | share + EXP_MODE=2 |
|---|---|---|
| chat0 decode/prompt | 36.8 / 45.9 | 37.9 / 45.7 |
| chat1 decode/prompt | 17.6 / 37.6 | 20.9 / 42.1 |
| chat2 decode/prompt | 20.5 / 42.0 | 20.8 / 44.7 |
| long decode/prompt  | 21.1 / 92.8 | 21.4 / 96.2 |

Round 0 went 7/7 for the EXP_MODE arm; round 1 went the other way on two cells (chat2
decode 17.5 vs 20.8, long prompt 53.9 vs 58.8) — the long cell swings 126.9 to 58.8 for
A and 138.5 to 53.9 for B across rounds, so both arms occupy the same range. Medians favor the arm by at most
15-20% on cells that were a tie in tier-1, and nothing here beats the noise floor of
2 rounds. Not adopted: the default (implicit mode 7 on this gfx906-compat build) is
kept, per the tier-1 read. If someone wants it, raise the rounds to 6 first; the
counter-argument is that mode 7 measured faster down than every alternative.

## Finding: the GSQ-RCO IQ2_XS set has no iq2_xs gate/up rows

The planned iq2_xs gate/up code path (gu type 17) never engages for this model: all
47 expert layers of shard 1 (shard 2 holds no expert tensors) use iq2_s / iq2_xxs /
iq1_m gate/up with q2_0 down, measured with `build/native_expert_bench`
(`native_expert_bench: 0 differing` on every layer). The LDS implementation of gu 17
went in anyway (`src/kernels/cuda/iq_kernels.cu`, DotG<17> + GridOf<17>, commit
0b3fdcf0): default-inert (only `STRATA_EXP_MODE` opens it), bitwise-equal to the old
path on every IQ2_XS row of `native_grouped_parity` at 512x256 and 2560x640, 0
failures. Kept for packs that do carry iq2_xs gate/up rows.

Static-analysis caution, read off the same bench on the types this model does carry:
the em-7 LDS gate/up form measured slightly slower than the em-0 CUDA layout for
iq2_s / iq2_xxs G12/T1 (176-178 vs 146-151 us gu), which matches the tier-1 layout
A/B being a decode tie. No engine-level A/B of that was run on top of the adopted
split overlap.

## Item: the file-tier IO scheduler — no A/B, skipped

`STRATA_IO_PF_SCHED` (src/core/expert_source.cpp:1203) only touches the file tier,
and the resident config holds 94-99.9% hits in the tier-1 logs, so the budget would
buy a measurement of a path this workload barely enters. Left off; revisit only if
the config loses its residency.
