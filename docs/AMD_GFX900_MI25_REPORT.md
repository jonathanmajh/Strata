# AMD Vega 10 (gfx900): project log — Strata on 2x MI25

What was done to run Qwen3.8-Flash-Next (Coder pack, IQ1_M) on two AMD Instinct MI25 cards
(gfx900, 16 GB HBM2 each, 32 GB total) with 32 GB of DDR4, the model files on a SATA SSD, served
from a Docker image, and how it performs. Branch: `gfx900-mi25` of
`github.com/jonathanmajh/Strata` (fork of `Niko1221/Strata`). Engine 0.1.58, app 0.2.12.

This file is the build/run log. The technical docs for the path are `docs/AMD_GFX900.md`
(source build) and `Dockerfile.gfx900` (image).

## Why gfx900 needs its own track

Strata's ready-made AMD (HIP) engine is a wave32 RDNA build (gfx110x, gfx120x, gfx1151). Vega 10
is wave64 (like gfx906), so it runs on the gfx906 compatibility backend compiled for gfx900
(`-DSTRATA_HIP_GFX900=ON`, added in commit `6d71834d`). The pack applies three open upstream
fixes it was built with insight from:

- PR #4983: gfx900 has no `v_fma_mixlo_f16`; the `STRATA_GDN_NOY` prefill path falls back to the
  FP32 product rounded to FP16 (`src/prefill/kernels.cu`).
- PR #4600: `__shared__ alignas(16) int4` does not compile under clang/HIP; int4 is aligned already.
- PR #4766: the CUDA-isms in dflash2's stream-priority/fork code only for the wave64 build.

The ship image is built against AMD's TheRock wheels `7.14.0a20260612` from
`https://rocm.nightlies.amd.com/v2/gfx900/`, the same nightly family the MI25 validation on the
community track (PRs #839/#2009) ran.

Two build fixes landed while making the engine compile on this toolchain (the source of truth is
on the branch; both are also in the image build):

- `include/strata/kernels/swiglu.cuh`: HIP has no CUDA `__expf` / `__fdividef` device intrinsics
  by those names. The three swiglu helpers now use `::expf` / plain `/` on HIP (libdevice's fast
  family compiles them to the same approximate quotient under fast math), and the CUDA-pinned
  path stays bit-exact through a `swilu_native_pinned` twin.
- `packaging/build_gfx900_linux.sh`: the TheRock wheel tree keeps `hipcc` in
  `rocm-core/bin`, not `llvm/bin/clang++`'s wrapper spot, so the script now picks
  `$HP/../core/bin/hipcc` when it exists and prepends it to `PATH`.

## The image

`Dockerfile.gfx900` builds the engine (component `0.1.58-linux-x86_64-hip-gfx900`, with gfx900
code objects in the `.hip_fatbin`) plus the Terminal, in one image. Published:

```
ghcr.io/jonathanmajh/strata:gfx900-0.1.58
```

Build it from the branch:

```sh
docker build -f Dockerfile.gfx900 . \
  --build-arg ROCM_INDEX=https://rocm.nightlies.amd.com/v2/gfx900/ \
  --build-arg ROCM_WHEEL=7.14.0a20260612 \
  --build-arg STRATA_COMMIT=$(git rev-parse --abbrev-ref HEAD) --build-arg STRATA_VERSION=0.1.58
```

`STRATA_COMMIT` has to be the full 40 characters (the packaging check runs it against a full
hash), and `packaging/stage_tuning.py` runs inside the build so `BUILD.json` carries a
`hipblaslt_version` (the gfx900 path runs plain hipBLAS; the field is there for the packaging
check).

## The Terminal's card gate

The engine in the image matched on first start ("Engine 0.1.58 (hip) is installed"), but the
model install step refused the cards: the model-install flow checks the GPU against `AMD_ARCHS`
in `setup.py`, and gfx900 was never in it (this is also why gfx906 documentation tells people to
bypass `strata launch`). The fix (commit `66810115`) mirrors the opt-in pattern setup already
uses for gfx1103/gfx1150:

- `STRATA_EXPERIMENTAL_GFX900=1` lets the card checks (Terminal and source setup) accept gfx900.
- Without the flag the refusal is unchanged, with a pointed message pointing to this doc.
- `ROCM_INDEXES` gained the gfx900 index and `ROCM_FAMILY_PINS` its pinned nightly, so the
  source-build path behind the flag is whole.

Verified both ways (`tools/test_setup_amd.py` and the setup suites pass with and without the
env), and in the packaged image the gate accepts GFX900 under the flag and refuses it without.

## Running it (the container form)

```sh
docker pull ghcr.io/jonathanmajh/strata:gfx900-0.1.58
docker run -d --name strata --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --ulimit memlock=-1 --shm-size 2g --stop-timeout 120 \
  -p 0.0.0.0:6666:8080 -v strata-data:/data \
  -e STRATA_EXPERIMENTAL_GFX900=1 \
  -e STRATA_ARENA_MMAP=1 -e STRATA_STAGE_TRIM=1 \
  -e STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 -e STRATA_GFX906_ATTN_REDUCE12=1 \
  -e MODEL=IQ1_M -e FAMILY=coder -e CONTEXT=256k -e KV=int8 -e GPUS=0,1 \
  ghcr.io/jonathanmajh/strata:gfx900-0.1.58
```

The model downloads into `/data/Strata-data` on first start (resumable); `--expert-profile` is
NOT passed — the Coder pack's profile is adopted by the low-RAM flow of `strata launch`. Engine
flags are not command-line on the Terminal; they would go through `--set KEY=VALUE` / `STRATA_SET`.

## What it measured so far (the operator's box, 2026-11-08, IQ1_M @ 64k int8)

Box: 2x MI25, 32 GB DDR4 with the model on a SATA SSD. Startup picks, without any `--set`:

- 2-card layer split K=25 (CUDA0 layers 0-24 / CUDA1 25-47 + head; the auto prober chose it —
  nearly the hand-tuned `27` we had planned)
- 5994 + 5536 expert slots in the GPU caches (~22 GiB of 32 GB), 94% of experts resident
- 8.24 GiB of experts in a pageable RAM complement; first request after a container start reads
  ~32 GB from the SSD; afterwards the file tier goes quiet (`0.0 MB of expert reads` warm)

Request numbers (engine timings, `docker.engine.log`):

| run | prompt tok/s | decode tok/s | draft accept | hit rate |
|---|---|---|---|---|
| warmup (1 tok) | 36.8 | 15.2 | – | 98.1% |
| short chat | 40.7 | 24.1 | 49/75 | 99.7% |
| 267-tok reply | 27.3 | 25.3 | 134/216 | 99.8% |
| 6243-tok generation | 44.3 | 21.9 sustained | 2151/3986 | 99.9% |
| 63196-token prompt | 400.2 | 22.1 | 732/1278 | 99.9% |
| 64963-word 2nd turn (63,191 reused) | 156.4 | 23.1 | 214/383 | 99.9% |

Decode held 22-25 tok/s across a ~4.7-minute generation with `decode waited 0.0 ms in 0 fetches`:
GPU-bound, not disk-bound. VRAM headroom at steady state: 840 MiB / 716 MiB per card — don't
trade more experts in without giving something up.

## Where this leaves the benchmark plan

The operator's overnight driver (`bench-overnight.sh`, outside this repo) restarts the container
per variant and runs three tools against the live server:

1. `tools/strata_bench.py` — chat / 4K / 16K / agent-turn suite, 3 runs each, medians
2. its `--long` pass (32 K prompt)
3. `tools/needle_bench.py --lengths 1k,32k,128k,262k` — needle recall at depth

Two variants planned: IQ1_M (Coder) and IQ2_XS (the full-expert GSQ-RCO set, `qwen` family,
68 GB download — the Coder pack only exists in IQ1_M), both `--context 256k` (= 262,144 tokens;
the launch flow accepts it and warns above 262,144 only).

Expectations, so surprising numbers are not read as breakage:

- int8 KV at 262 k is ~10 GB total; `--layer-split auto` plus `expert cache auto` will shrink the
  GPU caches by about that and the 99.9% decode hit rate will fall on either variant. That is the
  trade the runs are meant to measure.
- The first 262 k needle prompt after a restart reads the file tier cold (SATA); 30-60 min is
  normal, later rounds are much faster. `the OS read X MB from storage` in the engine log is the
  number to watch.
- A larger quant pushes more of the expert set past VRAM into the RAM tier and then the SSD tier;
  decode becomes SSD-bound first there. RAM itself fits both variants (the guard ceiling is
  ~30.5 GiB and ~6.2 GiB was in use at 64 k).

Results folders from `strata_bench.py` are shareable as-is (keys/IPs redacted), and
`--compare <folderA> --compare-only <folderB>` prints two runs side by side.

## What it measured once the 262 k bench was on (the operator's box, 2026-11-08/09)

### IQ1_M (Coder), int8 KV, `--context 256k`

All six needle runs found their needle (`tools/needle_bench.py`, depths 10/50/90 %, fresh server
per context, cards cooled under 60 °C between runs):

| context | needle 10% | needle 50% | needle 90% | prefill tok/s | decode tok/s |
|---|---|---|---|---|---|
| 32k → 48k actual | 81 s ✓ | 1 s ✓ | 88 s ✓ | 358–366 | 22.4 |
| 64k → 67k actual | 171 s ✓ | 181 s ✓ | 145 s ✓ | 340–386 | 22.4 |
| 128k → 136k actual | 341 s ✓ | 362 s ✓ | 188 s ✓ | 310–364 | 23.4 |
| 262k → 272k actual | 781 s ✓ | 770 s ✓ | 774 s ✓ | 318–328 | 23.4 |

(The harness targets 2% under the label because the engine refuses a prompt with no room to
answer, so the "actual" configs are what the runs used.) Complete numbers, suite reports and
config paths: `/data/install/BENCHMARK-RESULTS.md` on the box.

### IQ2_XS (full set, `qwen` family) at 262 k: the hang, and the budget that fixes it

Plain `--mmap-experts` at 262 k makes prefill hang: the engine's issue #29 watchdog prints
`no progress for 60 s … waiting for the GPU (attention, router) at layer 40–45 of the prompt
chunk` and exits (SIGABRT), the server restarts the engine. Not an OOM: it is file-tier page-in
starving the decoder card's prefill on the 32 GB box. Sweeping the engine's resident RAM budget
(`--resident-budget-gib`, page-locked expert tier) fixed it, and the sweep found the sweet spot:

| setting | hangs | needle runtime @ 262k | note |
|---|---|---|---|
| plain mmap-experts | 3 (watchdog kills) | fastest prefill of the set | unusable — the run dies mid-sweep |
| `--resident-budget-gib 8` | 0 | 148–164 tok/s | too slow |
| `--resident-budget-gib 20` | 2 | — | over the box (self-clamps to 18.15) |
| `--resident-budget-gib 30` | 0 | 259–275 tok/s | works, but 14 GiB more pinned |
| **`--resident-budget-gib 16`** | **0** | **942–1037 s, 238–254 tok/s** | **best balance** |

### Tuning sweep on top of r16 (IQ2_XS, 262 k, one fresh server per variant)

| variant (all with `--resident-budget-gib 16`) | needles 10/50/90 | prefill tok/s | hangs |
|---|---|---|---|
| r16 baseline (no split flags) | 1037 / 1008 / 942 s ✓ | 238–254 | 0 |
| `--layer-split 27` alone | 985 / 905 / 898 s ✓ | 260–261 | **0 — best** |
| `--adapt-every 0 --pcie-frac 0` ("light") | 1071 / 1047 / 1024 s ✓ | 229–240 | 0 |
| split27 + `--pipeline-windows 2 --trim-stage-weights --kv-resident 16384 --vram-reserve-mib 768 --pcie-frac 0` ("tuned") | 981 / 920 s ✓, 90% shot died — watchdog hang at layer 18, chunk from token 175,104 | 257–264 | 1 |

Two launch-flow bugs got fixed while finding that: `--adapt-every` is a Terminal-side option the
packaged engine knows nothing about (pass it through the config's engine args and the engine dies
at startup), and `--trim-stage-weights` is a boolean flag that takes no `on` value. Both now go
through `strata launch --set` (or `STRATA_SET`) the way `docs/AMD_GFX900.md` already shows.

**Best config for this box, IQ2_XS @ 262 k:** `--resident-budget-gib 16` +
`--layer-split 27` — nothing else. Clean prefill (260–261 tok/s fresh, no KV reuse artifacts),
zero hangs, zero extra tuning knobs. One run per variant ≈ ±3% noise; `split27` wins
comfortably.

`docker/compose/docker-compose.gfx900-mi25.yml` runs exactly that (and the IQ1_M twin) with
`ghcr.io/jonathanmajh/strata:gfx900-0.1.58`.

## Not yet done / open

- ~~No extended runs yet at 256 k context on either variant~~ IQ1_M 32k/64k/128k/262k and
  IQ2_XS 262k are measured now (the tables above); IQ2_XS at shorter contexts is still open.
- No cross-quant quality comparison yet.
- `HSA_OVERRIDE_GFX_VERSION` is not needed (gfx900 is the card's own name; nothing to test there).
- The GitHub Packages visibility of `ghcr.io/jonathanmajh/strata` may need a manual flip to
  Public in the UI if an anonymous pull ever 404s (the token PATCH endpoint returned 404).
- `--pipeline-windows 2` plus the memory knobs cost one watchdog hang in the sweep; a second
  run would tell noise from a real regression.
- The `light` variant (no adapt rounds, no PCIe fraction) is 10% slower than split27 alone; if
  the box gets busier with background jobs, `--adapt-every 0` may still be the safer pick there.
