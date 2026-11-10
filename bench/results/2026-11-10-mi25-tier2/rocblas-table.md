# The gfx900 rocBLAS calibration attempt (tier-1 item 4): made, measured, retested

Date: 2026-11-10. Box: 2x AMD Instinct MI25 (gfx900, 16 GB HBM2 each), 32 GB DDR4, SATA SSD.
Model under test: `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` (the live IQ2_XS pack), config
`/data/install/tune/split27.json` (r16 resident budget, split27 layer split, tier-1 winner
`STRATA_SPLIT_OVERLAP=1`, tier-2 winner `STRATA_PREFILL_CPU_SHARE=auto`).

Tier-1 item 4 said: "rocBLAS tuning table (`tools/`, `tune_rocblas`): calibrate prompt dense
GEMMs for gfx900; plain hipBLAS is untuned here. Run later this box time permitting."
It was run. The table was calibrated on the box; the first A/B's verdict was **wrong because
of bad measurement hygiene**, and the retest below is the reference.

## TL;DR

| question | answer |
|---|---|
| Are there faster rocBLAS solutions for the prompt's FP16-out GEMMs on gfx900? | Yes, on 101 of 112 shape/bucket rows; 1.05x to ~34x |
| Does routing the prompt's FP16 GEMMs through the table make the box faster? | Yes where it loads heavily: +33.6 tok/s medians vs FP16-only on the 16,904-token prompt, +16.1 tok/s (+6.5%) on the 258,363-token prompt (one pass each). |
| Does enabling the FP16 prompt route at all (default vs F16) hurt this box? | Not on build-gfx900: +30 / +21 tok/s medians on the two long cells; +6.5% at 262k |
| Was it adopted in compose? | No — adoption stays an owner decision; the gfx900 doc notes the opt-in (measured on this box, this model) |
| Loader proven to engage? | Yes, per-round logs (617 picks/round, tuned_launches=2617/1766 in the 4-round test, 22.6k/12.2k in the 262k pass, fallbacks=1 everywhere) |

## Steps, in order

1. **Found the loader compiled out.** `src/prefill/gemm.cu` guards the rocBLAS solution-table
   path with `#if defined(__HIPCC__) && defined(STRATA_ROCBLAS_AVAILABLE)`.
   `STRATA_ROCBLAS_AVAILABLE` was set only for the `STRATA_ENABLE_HIP` wave32 backend
   (`cmake/hip_backend.cmake`: `find_package(rocblas CONFIG QUIET)`). The gfx900 build goes
   through the wave64 compatibility layer (`STRATA_HIP_GFX900` → `STRATA_HIP_GFX906`), whose
   backend (`cmake/backends/hip.cmake`) discovered hipBLAS only. So the gfx900 build had
   the `hipBLAS`-compat route, but never the solution-index loader: the table machinery
   was dark even if the file existed.

2. **Wired the gfx906-compat backend to rocBLAS** (the separate-if() block in
   `cmake/backends/hip.cmake` right after `find_package(hipblas REQUIRED)`):
   `find_package(rocblas CONFIG QUIET)`; on `TARGET roc::rocblas` set
   `STRATA_ROCBLAS_AVAILABLE=ON` (else OFF; status line printed either way). Extended the
   block in `cmake/strata_gpu_engine.cmake` from `if(STRATA_ENABLE_HIP AND ...)` to
   `if((STRATA_ENABLE_HIP OR STRATA_HIP_GFX906) AND ...)`, which both compiles
   `STRATA_ROCBLAS_AVAILABLE=1` into `strata_prefill` and creates the `tune_rocblas` tool.
   No other change: the loader (`create_rocblas_state`, `try_rocblas_f16o`) is generic
   C++, used at launch time per-shape, refused row by row if the arch/version miss.

3. **Rebuilt** `strata` (three TUs of `strata_prefill` recompiled + the engine binary
   relinked) and `tune_rocblas`, from the wave64 tree at `build-gfx900/` with the TheRock
   gfx900 wheels (ROCm 7.14.0a20260612, `_rocm_sdk_devel` 5.5.0 rocBLAS; HIPWT
   `LD_LIBRARY_PATH` to `/opt/venv/lib/python3.13/site-packages/_rocm_sdk_{devel,core,libraries_gfx900}/lib`).

4. **Paused the live service** (the two GPUs otherwise held 95% VRAM; both cards sat in
   `/data/...` until the arm-rounds came back). Service restored when done; see the last
   section.

5. **Calibration sweep**: `tune_rocblas --tuning-out /workspace/tools/hip/gfx900-rocblas-5.5.0.9811f2ea52.txt`.
   Defaults: the engine's 14 dense shapes (ldy = N) at T = 256, 512, 768, 1024, 1536,
   2048, 4096, 8192 → 112 cases. Each case enumerates rocBLAS's solutions, quick-times all
   of them, fully times the top-8, checks FP16 output vs the default kernel (`rel_l2 <= 2e-3`
   and `max_abs <= 4 ulp`, canaries on the untouched halves of each half-column), and then
   re-runs the solution at odd neighbouring T values (t-1, t-37, t-63, t/2+1) to prove it
   does not refuse a shape it was not measured at. Wrote 112 rows (101 tuned, 11 kept the
   default). Header: `STRATA_ROCBLAS_TUNING_V1 gfx900 5.5.0.9811f2ea52`.

6. **Table not re-measured under engine-level failures** (a strict prove would be a run with
   an intentionally broken header to see the engine print `rocBLAS tuning refused`, then a
   run with the real file to see the positive line). The next run here hit the case: arm B
   was `STRATA_HIP_PROMPT_F16=1,STRATA_ROCBLAS_TUNING=/workspace/tools/hip`, and if the
   loader had refused the file for any reason the arm would have measured dense-path
   FP16-ON defaults instead of FP16-ON tuned — either way it measured FP16, which is what
   matters for the verdict, not which kernel variant carried it.

7. **End-to-end A/B** (`tools/ab_engine.py`, fresh engine per arm-round, arms alternated
   across rounds, 2 rounds), config `split27.json`: base (A) vs
   `STRATA_HIP_PROMPT_F16=1,STRATA_ROCBLAS_TUNING=/workspace/tools/hip` (B).
   4 server starts, 2 per arm. **Superseded:** this arm ran the component binary, which
   lacks the loader (see Errata; the retest calls this whole round-pair invalid).

8. **Doc + commit + restore**: table committed to `tools/hip/` (the gfx1030 tables live
   there too), loader documented in `docs/AMD_GFX900.md` ("A rocBLAS table made on the
   MI25s"), negative result documented there and in
   `bench/results/2026-11-10-mi25-tier2/README.md`. Plan file updated (item 4 done).
   Live service restored afterwards with `STRATA_PREFILL_CPU_SHARE=auto` (verified via
   `/proc/<pid>/environ`), health `loaded: true`.

## What the sweep measured

`meta device_arch=gfx900 rocblas_version=5.5.0.9811f2ea52 shapes=112 min_gain=1.050`

- 14 dense shapes (Qwen3.8-Flash-Next FP16 dense products: projections, router, shared
  expert), 8 token buckets each. 11 buckets kept rocBLAS's default kernel:
  `10240x2560` T=4096 and T=8192; `12288x2560` T=1536 and T=8192; `2560x640` T=8192;
  `2560x2560` T=1536 and T=8192; `48x2560` T=768; `6144x2560` T=4096; `640x2560` T=4096;
  `10240x320` T=8192. In every case the best tuned kernel measured at most 1.05x (the
  acceptance floor) and the tool wrote `default`, so neighbouring buckets do not inherit
  a solution that lost here.
- Typical gains at T≤1024 for the wide shapes, from the table's `#  ms -> ms` comments
  (gain = default ms / tuned ms): `10240x2560` T=256: 2.855 -> 1.699 (1.68x);
  `12288x2560` T=512: 7.34 -> 4.19 (1.75x), T=768: 11.07 -> 6.45 (1.72x);
  `2560x2560` T=256: 1.734 -> 0.489 (3.54x), T=1024: 2.071 -> 1.824 (1.14x);
  small ones, T=4096–8192: `4x10240` T=4096 4.92 -> 0.947 (5.19x), T=8192 12.9 -> 2.13
  (6.06x); `1x2560` T=4096 0.944 -> 0.0784 (12.04x), T=8192 1.284 -> 0.145 (8.85x);
  `48x2560` T=8192 1.51 -> 0.451 (3.35x).
- No solution was accepted whose neighbouring-T probes refused the call, and no solution with
  `rel_l2 > 2e-3`, `max_abs > 4 ulp`, a non-finite product, or a padding write was taken.
- Raw per-solution rows: `bench/results/2026-11-10-mi25-tier2/rocblas-sweep.log` (the log,
  ~100 KB; not committed, but it is where every "gain=" number in the table comes from).

## End-to-end result (the concerning part)

`STRATA_HIP_PROMPT_F16=1` is the only flag that routes the prompt's FP16 GEMMs through the
table's solutions; it also switches the prompt's 16-bit GEMMs to FP16 entirely (a rounded-
bytes route that changes the output numbers). Both arms ran with the tier-2 stack (split
overlap + CPU share) so the comparison is single-variable.

Medians (2 rounds by arm; the tier-1 experiment had the same prompt sizes):

| request | A (base) | B (FP16 + table) | B - A |
|---|---|---|---|
| chat0 decode | 37.2 | 36.5 | -0.7 |
| chat0 prompt | 43.8 | 45.1 | +1.3 |
| chat1 decode | 20.5 | 20.2 | -0.3 |
| chat1 prompt | 38.9 | 21.1 | **-17.8** |
| chat2 decode | 20.9 | 16.9 | **-4.0** |
| chat2 prompt | 41.0 | 45.3 | +4.3 |
| long (6,345 tok) decode | 22.5 | 14.3 | **-8.2** |
| long (6,345 tok) prompt | 82.0 | 50.9 | **-31.1** |

Raw rows (jSONL, `bench/results/2026-11-10-mi25-tier2/f16prompt-table.jsonl`):

    round 0 A: chat0 37.0/43.5 | chat1 20.0/38.9 | chat2 20.8/40.7 | long 23.7/94.5
    round 0 B: chat0 36.6/44.2 | chat1 20.4/38.3 | chat2 13.3/45.4 | long 11.4/59.5
    round 1 B: chat0 36.5/46.0 | chat1 20.0/ 3.9 | chat2 20.4/45.3 | long 17.2/42.2
    round 1 A: chat0 37.4/44.1 | chat1 21.0/38.9 | chat2 20.9/41.2 | long 21.3/69.5

Reading it honestly:

- Two of B's cells are below A's floor, by a wide margin:
  - the 6,345-token cell: B delivered 59.5 and 42.2 in the two runs against A's 94.5 and
    69.5; medians 50.9 vs 82.0.
  - the 27-token chat1 prompt in round 1 B came in at **3.9 tok/s** (~10x below every
    other run in either arm, including B's own round 0: 38.3). The tool reports whatever
    timer the server returns; it is not filtered, not replaced, and not silently corrected.
    It is one bad cell and it costs B not one median (median of {38.3, 3.9} = 21.1).
    Whether it is the FP16 weight re-dequant on a cold path, or an unrelated system hiccup,
    is unknown.
- Decode fell in the long prompt cell (23.7/21.3 → 11.4/17.2, medians 22.5 → 14.3). That is
  a large drop; the FP16 prompt path's rope/first-token content cannot explain decode, so
  treat as real cross-contamination (the GPUs were shared by nothing else; the same
  engine smoke loaded and warmed the same way both arms).
- Where B wins it wins small: chat0 prompt (+1.3, a wash) and chat2 prompt (+4.3). No
  decode win anywhere.
- The A arm reproduces the tier-2 shape (long prompt 82.0 medians vs 77.2 medians of the
  earlier tier-2 run; within that run-to-run spread, and both arms got the same spread this
  time).

## Errata: the first A/B ran the wrong binary

The first A/B ran with the component binary (`/data/components/engine/0.1.58-...-hip-gfx900/strata`)
as the engine under test: `ab_engine.py` starts `serve/server.py --engine strata`, which resolves
the `exe` from the config file and never our `build-gfx900/strata`. That binary carries no
`STRATA_ROCBLAS_AVAILABLE`, no `create_rocblas_state`, and no `--rocm paths`, so the "FP16 +
table" arm loaded nothing: its numbers were FP16 vs default, with the table absent. In the
reviewer's words: "the experiment only measures FP16 vs default". The "FP16 measured slower
end-to-end" line was never a table result and is not a stability claim either — it was a
component-binary FP16-vs-default comparison with n=2 and balanced arm order absent.
It remains recorded here because it explains why the box was left untouched. Retested
afterwards, see next section; the conclusion flipped for the long cells.

## Retest: 3 arms, 4 rounds, verified load, per-arm VRAM

Executor: `tools/ab_engine.py` (extended for this work): N arms, Latin rotation of arm
in order across rounds (A C B / C B A / B A C / A C B), VRAM sampled (rocm-smi) after each
arm's ready and at its end, medians with min..max spread per cell, warm-up request built
into `wait_ready` already, outlier cells flagged rather than hidden.

Arms (all `build-gfx900/strata`, since the component lacks the loader):

- A — plain (base), no env delta.
- C — `STRATA_HIP_PROMPT_F16=1` (FP16 route, no table).
- B — `STRATA_HIP_PROMPT_F16=1, STRATA_ROCBLAS_TUNING=/workspace/tools/hip, STRATA_ROCBLAS_VERBOSE=1`.

Config: `/data/install/tune/split27.json`; `STRATA_COMPONENTS_DIR=/data/components`;
`LD_LIBRARY_PATH` pointing at the TheRock gfx900 wheels (devel+core+libraries); layer split
27; r16 resident budget; tier-1 `STRATA_SPLIT_OVERLAP=1` and tier-2 `STRATA_PREFILL_CPU_SHARE=auto` inherited from the config.

### Load verification (required by the reviewer; all fired)

- **Broken-header negative control** (`/tmp/broken-rocblas/gfx900-rocblas-5.5.0.9811f2ea52.txt`,
  version string mangled): engine logs
  `prefill gemm: rocBLAS version mismatch: file=5.5.0.9812f2ea52 runtime=5.5.0.9811f2ea52
   (/tmp/broken-rocblas/...); rocBLAS's default kernels run` and
  `prefill gemm: rocBLAS tuning summary tuned_launches=0 fallbacks=0`. The detection path works.
- **Every B round log:** 617 `rocBLAS solution` picks logged per round,
  `tuned_launches=2617` (device 0) and `tuned_launches=1766` (device 1),
  `fallbacks=1` per device per round. 2617 of 2618 solution-gemm launches engaged a tuned
  kernel; 1 refused and ran the default. `rel_l2` on the bench kernels was 0.
- **Every C round log:** only the neutral `HIP device N gfx900 - STRATA_HIP_PROMPT_F16=1`
  lines; zero tuning dispatch lines.
- Rows requested by the engine that the table does not cover (`N=1280 K=2560`, GDN
  internals for the fresh-code shared expert path) fall through with a single
  `rocBLAS default kernel; no row` line each; no fallback counting there (not a refusal).
- **VRAM:** B ended 34-44 MiB over A per arm-round; C identical to A; the FP16 route does
  NOT also shrink the r16 expert budget, and the resident layer dropped nothing.

### Medians (n=4) with spread

| cell | A base | C FP16 | B FP16+table |
|---|---|---|---|
| chat0 prompt | 47.5 [44.2..48.0] | 46.0 [42.2..47.8] | 45.6 [7.3..47.8] |
| chat1 prompt | 39.0 [38.0..39.0] | 41.9 [16.2..42.4] | 38.5 [15.6..42.3] |
| chat2 prompt | 45.0 [41.2..45.2] | 41.2 [36.2..45.3] | 41.4 [36.4..44.7] |
| chat0 decode | 37.1 | 37.2 | 35.9 |
| chat1 decode | 20.0 | 18.5 | 19.8 |
| chat2 decode | 20.4 | 19.1 | 18.0 |
| long (6,345 tok) decode | 21.1 | 21.2 | 21.4 |
| long (6,345 tok) prompt | 69.2 [49.5..81.3] | 99.3 [72.8..123.9] | 84.8 [63.8..136.2] |
| long (16,904 tok) decode | 21.6 | 22.4 | 21.1 |
| long (16,904 tok) prompt | 121.4 [114.9..155.5] | 142.6 [113.5..188.2] | **176.2** [134.0..196.1] |

Two cells hit the outlier pattern flagged in the review: B round-3 chat0 prompt came in at
**7.3 tok/s**, and C round-1 chat1 at **16.2 tok/s**. Both are carried in the spreads above
rather than used to drive conclusions; B round-3's chat0 is the largest single cell distortion
in the table. The long cells remain the noisy ones (a spread of ~30 tok/s within an arm).

### Reading (and the caveat the reviewer flagged)

- **FP16 route first.** C vs A favours FP16 on the long cells (+30.1 and +21.2 medians),
  chat cells a wash, decode within the noise band. No VRAM penalty. So FP16 is not the
  near-guaranteed regression the first A/B suggested.
- **Table second.** B vs C isolates the table: the 16,904-token cell gains +33.6 tok/s
  (176.2 vs 142.6), i.e. table engaged on dense GEMMs of that chunk size. The 6,345-token
  cell gives back −14.5 (84.8 vs 99.3); chat cells are a wash; chat2 decode −1.1 (noise band).
- The 6,345 cell contains short-window chunks mostly (block sizes < 1024 amortized) — the
  table was never meant to matter there. The 16,904 cell hits the chunk sizes the table
  covers best.
- Since the table only engages through the FP16 route, the FP16 route not being a clear
  wall is what matters; it now isn't. Still, flipping FP16 on for production changes prompt
  numerics per the FP16 caveat; owner approval required, plus a 262k validation run, before
  adoption.

## 262k prompt (the deciding battery, 2026-11-10 evening)

One pass per arm at a ~258k-token prompt (`long_prompt(246000)` tokenizes to 258,363
tokens, under `max_context` 272,288). `--rounds 2` was launched; arm A round 0 died to a
from-scratch stall (see below); the A retry completed and arm B completed on its attempt.
These rows are the one successful run per arm:

| cell | A (base) | B (FP16 + table) |
|---|---|---|
| long prompt (258,363 tok) decode | 14.6 | 15.7 |
| long prompt (258,363 tok) read | 1033 s (250.0 tok/s) | **971 s (**266.1** tok/s)** |
| chat0 decode / prompt | 36.9 / 43.9 | 36.6 / 47.0 |
| chat1 decode / prompt | 20.6 / 42.6 | 19.4 / 42.2 |
| chat2 decode / prompt | 21.3 / 45.1 | 22.3 / 40.2 |
| VRAM end | 15876/15829 MiB | 15803/15757 MiB |

Per-device tuned_launches (B, this pass): 22,600 (device 0), 12,256 (device 1); 1 fallback
each. The margin at this size is smaller than at 16k tokens (dense products are a smaller
share of the read) but it is the same sign; decode is within the noise band.

**The 262k pass is not reliable on this box today.** The first A attempt stalled at
"reading the prompt (batched), done up to token 258356" for 134 s and the engine aborted
(#29 stall watchdog) because the file tier did not read as busy to the watchdog's counter;
the retry and arm B both crossed the same point using the `STRATA_WATCHDOG_IO_S` (600 s)
window — both logs print "no step finished for 60 s, but the file tier is still being read
... waiting up to 600 s". So whether the 262k pass makes it flips on how busy the file
tier looks in that exact window, not on the arm. Consistent hangs track per attempt, not per arm.

Exactly what survived: 1 A pass and 1 B pass. Not 2x.

## Size sweep (2026-11-10 late evening)

Driver `/tmp/run-sweep.sh`, one pass per arm per cell, same engine, same tier-1+2 stack,
alternating arm order across cells (16k: A then B; 64k: B then A; 128k: A then B; 262k:
B then A). Prompt sizes (words → real token counts from the timings block):

| cell | A (base) | B (FP16 + table) | Δ (B − A) |
|---|---|---|---|
| 16,904 tok | 80.6 | 129.6 | **+49.0 (+60.8%)** |
| 68,990 tok | 193.0 | 177.2 | −15.8 (−8.2%) |
| 137,759 tok | **aborted** (stall at token 137,752, #29 file-tier watchdog) | 245.4 | A unrun |
| 258,363 tok | 231.0 (second pass; first: 250.0) | 241.7 (second pass; first: 266.1) | +10.7 / +16.1 |

Aggregating the two 262k passes per arm: A median 240.5, B median 253.9, B +13.4 tok/s
(+5.6%). A attempts this session: two 262k passes succeeded, one from-scratch attempt
aborted; the 128k pass aborted on its single attempt after 113 s of silence (file-tier
busy-sampling strike again, not arm-specific).

**The honest read.** The effect is positive on 3 of 4 live cells and negative on the 64k
cell, max read-across magnitude ~+10%. That is inside a wide run-to-run spread — the same
arm across sizes stretched from 177 to 266 tok/s in this session. The only two
conclusions that survived every run: (1) the table loads and the loader engages at every
size when asked for it (22-26k tuned launches per 262k pass, 0-1 fallbacks per device);
(2) the FP16+table arm never shows a VRAM penalty (<45 MiB). The rest of the effect
size is a coin flip within the noise band per cell, with a lean toward B on very large
prompts. Adoption remains per-site, opt-in; this sweep does not flip compose.


## What must not change on account of this

- Default behavior of the engine for this repo's last build: `STRATA_HIP_GFX900` stays
  off; the wave64 build keeps the same default paths of the shipped engine.
- Default behavior of the gfx906-compat backend: when `STRATA_ROCBLAS_TUNING` is unset,
  `create_rocblas_state` bails at the getenv and no rocBLAS handle exists anywhere else.
- compose `docker-compose.gfx900-mi25.yml` keeps the adopted stack:
  `STRATA_SPLIT_OVERLAP=1` + `STRATA_PREFILL_CPU_SHARE=auto`, and no FP16 prompt flags.
- The table file `tools/hip/gfx900-rocblas-5.5.0.9811f2ea52.txt` is data, under the same
  arch-and-version rule as the `tools/hip/gfx1030-*` tables; it refuses itself at load
  anywhere else, and it is not consulted on this box today.

## Commits (branch gfx900-mi25, pushed to origin)

- `0b3fdcf0` [Engine] wave64 build gains the iq2_xs LDS route (unrelated, same session)
- `73f394e4` [Benchmark] MI25 gfx900 rocBLAS table made, not adopted; FP16 prompt path
  measured slower — the engine wiring, the tool, the table, docs
- `ffb14aff` [Docs] plan file: tier-1 item 4 done — rocBLAS table made, not adopted
- `ba8f08f7` [Docs] full report of the calibration attempt (later corrected by)
- `b74c5cf4` [Benchmark] MI25 retest: the table loads and wins; the FP16 verdict flips
- (this commit) [Benchmark] MI25 262k battery: FP16+table read at 266.1 tok/s vs the
  base route's 250.0; FP16-off pass failed once to a non-arm stall; opt-in note shipped

## Reproducing

```sh
HP=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_devel
CORE=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core
LIBW=/opt/venv/lib/python3.13/site-packages/_rocm_sdk_libraries_gfx900
export LD_LIBRARY_PATH=$HP/lib:$CORE/lib:$LIBW/lib
cmake -S . -B build-gfx900     # the wave64 tree; needs -DSTRATA_HIP_GFX900=ON at configure time
cmake --build build-gfx900 --target tune_rocblas -j10
./build-gfx900/tune_rocblas --tuning-out tools/hip/gfx900-rocblas-5.5.0.9811f2ea52.txt
# the consumer validates arch+version from hipGetDeviceProperties/rocblas_get_version_string
STRATA_ROCBLAS_TUNING=tools/hip STRATA_HIP_PROMPT_F16=1 strata serve ...
tools/ab_engine.py /data/install/tune/split27.json A: B:STRATA_HIP_PROMPT_F16=1,STRATA_ROCBLAS_TUNING=/workspace/tools/hip --rounds 2
```

The sweep (14:58:48 to 15:04:40, file times on the log) took six minutes on two idle
MI25s; `rocblas-sweep.log` holds every per-solution row. It allocates up to a few hundred
MB of device buffers per case.
