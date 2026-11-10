# Tier-1 A/B runs on 2x MI25, IQ2_XS (2026-11-10)

Executor: `python3 tools/ab_engine.py` (one fresh engine per arm-round, arms alternated across
rounds, medians shown; env `STRATA_COMPONENTS_DIR=/data/components`; config
`/data/install/tune/split27.json` = `--resident-budget-gib 16 --layer-split 27 --kv int8
--mmap-experts --spec 6 --draft-conf 0.35`, DFlash2 drafter auto-picked, engine
0.1.58-linux-x86_64-hip-gfx900). Requests per round: 3 fixed short chats (thinking off,
max 256 new tokens) + one 6,345-token prompt (max 256 new tokens). Server timings from the
engine (`timings.predicted_per_second` / `prompt_per_second`). Cards stayed under 60 °C.

The live 8080 service was paused for the sweep and restored afterwards. One lost cell in run 1
(base round 1 `long`): the engine logged the file-tier watchdog line (#1407, "no step finished
for 60 s, but the file tier is still being read") and the request returned HTTP 503; that arm
was not re-run — the cell is missing from the medians below rather than replaced by a guess.

Files: `run1-layout.jsonl` / `run1-layout.log`, `run2-overlap.jsonl` / `run2-overlap.log`
(one jsonl line per request; arm, round, decode/prompt tok/s, output tokens, 80-char answer
head). Median tables are the `ab_engine.py` printout of run 2; run 1's medians below are the
same statistic computed from its jsonl.

## Run 1 — decode kernel layouts (`STRATA_MMVQ_WAVE=0`, `STRATA_EXP_MODE=2`), 2 rounds

| cell  | base dec/prompt | mmvq0 dec/prompt | expmode2 dec/prompt |
| --- | --- | --- | --- |
| chat0 | 33.0 / 42.4 | 33.4 / 42.0 | 32.5 / 45.0 |
| chat1 | 13.7 / 37.0 | 19.2 / 22.6* | 14.6 / 42.6 |
| chat2 | 20.3 / 42.8 | 20.8 / 39.1 | 21.1 / 43.2 |
| long  | 18.9† / 56.5† | 18.6 / 66.4 | 21.4 / 78.6 |

† n=1 (lost cell). \* that row's prompt SPEED is unusable (4.9-22.6 tok/s came with a cold
page tier after a 307 s start) — its decode row is the meaningful one (20.4 tok/s).

Reading: no clear decode gain from either layout switch on this box; ties within the
run-to-run spread the engine's own logs show (dflash2 accept rate moves the decode number
per request). `STRATA_EXP_MODE=2` hinted at a prefill edge on the more-routed long prompt
(78.6 vs 56.5 tok/s) but that is one arm with two rounds — needs a dedicated prefill A/B
before any call. Neither switch changes tokens/latency guarantees; both are bitwise-checked
A/B switches by design.

## Run 2 — `STRATA_SPLIT_OVERLAP=1`, 3 rounds (ab_engine's own medians)

| cell  | base dec/prompt | overlap dec/prompt |
| --- | --- | --- |
| chat0 | 35.0 / 44.2 | 39.5 / 43.8 |
| chat1 | 20.2 / 39.0 | 21.5 / 43.5 |
| chat2 | 15.8 / 41.0 | 18.6 / 46.3 |
| long  | 21.4 / 75.6 | 22.3 / 78.0 |

`STRATA_SPLIT_OVERLAP=1` won every decode cell on the medians (+13% chat0, +6% chat1,
+18% chat2, +4% long) and both prompt medians of the small chats. Zero hangs, zero watchdog
lines in either arm. Direction agrees with the 2x RX 7900 XTX measurement (+33% decode,
`docs/MULTI_GPU.md:292`), magnitude smaller here — plausibly because the gfx900 decode window
is dominated by software-dp4a expert work rather than launch sync (unconfirmed interpretation).

Verdict for `docker-compose.gfx900-mi25.yml` / the r16 + split27 config on this box:
add `STRATA_SPLIT_OVERLAP=1` as an env switch (no engine args change; mutually exclusive with
`--pipeline-windows`, which stays off). It is opt-in, not a default anywhere; runs are one
machine, one quant, unverified elsewhere on gfx900.

## Not run now

- rocBLAS tuning table for gfx900: needs a build system/toolchain configure (the box runs a
  downloaded engine component, no CMake build tree) — left as the next item of Tier 1.
- `--spec` sweep: the live drafter on this box is DFlash2 (`--draft-conf` governs its window
  cut, windows up to 8 tokens); the planned MTP-spec sweep does not map 1:1. Deferred.

## Follow-ups from the sweeps

- Prefill-focused A/B of `STRATA_EXP_MODE=2` (needle or long-prompt pairs), since run 1's
  only prefill signal was positive.
- Tier 2 code items are unchanged and remain unconfirmed static analysis
  (`research/mi25-tier1/PLAN.md`).
