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
