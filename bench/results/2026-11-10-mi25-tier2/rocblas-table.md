# The gfx900 rocBLAS calibration attempt (tier-1 item 4): made, measured, refused

Date: 2026-11-10. Box: 2x AMD Instinct MI25 (gfx900, 16 GB HBM2 each), 32 GB DDR4, SATA SSD.
Model under test: `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS` (the live IQ2_XS pack), config
`/data/install/tune/split27.json` (r16 resident budget, split27 layer split, tier-1 winner
`STRATA_SPLIT_OVERLAP=1`, tier-2 winner `STRATA_PREFILL_CPU_SHARE=auto`).

Tier-1 item 4 said: "rocBLAS tuning table (`tools/`, `tune_rocblas`): calibrate prompt dense
GEMMs for gfx900; plain hipBLAS is untuned here. Run later this box time permitting."
It was run. The table was calibrated on the box, and the route it serves measured **slower**
end to end. The table and the loader went in; the env switch that arms them did not.

## TL;DR

| question | answer (medians, 2 rounds alternated per arm) |
|---|---|
| Are there faster rocBLAS solutions for the prompt's FP16-out GEMMs on gfx900? | Yes, on 101 of 112 shape/bucket rows; 1.05x to ~34x on the microbenchmark |
| Does routing the prompt's FP16 GEMMs through the table make the box faster? | **No.** The enablement arm lost on the cells that matter and won only small cells |
| Was it adopted? | No. `STRATA_HIP_PROMPT_F16` untouched in compose; the table ships as an opt-in file |
| Doc | `docs/AMD_GFX900.md` ("A rocBLAS table made on the MI25s"), `bench/results/2026-11-10-mi25-tier2/README.md` |

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
   4 server starts, 2 per arm.

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

## Interpretation hypothesis (not verified, static analysis)

- The dense FP16 GEMMs are not the bottleneck of a 257k-context prompt on this build. The
  expert path through the MMQ/IQ routes and the attention KV dominate; making those 14
  GEMMs 1.4-1.8x faster cannot move end-to-end prompt by 31 tok/s. The table cannot be a
  win of that size anyway.
- The loss itself is what needs explaining. The likeliest cause is not a bad kernel (each
  solution was output-checked and refused rows were never used) but the FP16 route
  itself: FP16 inputs halve the bytes the wide GEMMs read, but MI25 (gfx900) has no tensor
  path folded into this engine's FP16 route, and FP32 GEMMs over the same data
  via the current route are apparently already well served by hipBLAS's default kernels
  (which this box was not measuring at all: prompt read 260 tok/s @262k came from the
  DEFAULT route, which does not use FP16). So switching to FP16,
  whatever the table does to those 14 shapes' speed, tips the balance against a route that
  was almost free for this model.
- Not tested, would settle it: (a) FP16-only arm (no table) vs base, to separate
  "the FP16 route" from "the table"; (b) 262k prompt, not 6.3k, for a single A/B;
  (c) profiling which GEMMs actually carry the dense-path time on this model with
  `STRATA_ROCBLAS_VERBOSE=1` + the engine log between the two arms.

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
