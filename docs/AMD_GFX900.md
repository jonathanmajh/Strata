# AMD Vega 10 (gfx900): experimental setup, 2x MI25

This is the opt-in path for discrete Vega 10 cards: Radeon Instinct MI25 (16 GB HBM2), Vega 56 / 64. It uses
the wave64 compatibility backend (the `gfx906` layer compiled for `gfx900`, `-DSTRATA_HIP_GFX900=ON`), with
software signed byte dot products because gfx900 has no packed dot instruction. It is separate from the normal
wave32 AMD build and from gfx906.

Built on the community track first assembled on Oct 2026 and Nov 2026:

- `sagentlab/Strata` branch `feat/gfx900-wave64-main` (PRs #839 / #2009): the installer, the wave64 intrinsic
  validation, and the first full build on an MI25 (engine 0.1.43, TheRock `gfx900` wheels 7.14.0a20260612;
  device, wave64-intrinsic, BF16/FP16 GEMM checks 3/3). A historical engine log on one MI25: 39.9 prompt tok/s,
  31.6 tok/s decode.
- PR #4983 (nerdoc, confirmed by benchmark #4982 on a Vega 64): `v_fma_mixlo_f16` is gfx906-only; gfx900 falls
  back to the FP32 product rounded to FP16 in the `STRATA_GDN_NOY` prefill path.
- PR #4600 (suvayu): `__shared__ alignas(16) int4` does not compile for HIP in clang; int4 is 16-byte aligned already.
- PR #4766 (nfriedly): dflash2's stream-priority / fork code (the CUDA-isms) behind the wave64 compat build.
- hipBLAS 0.54 (no `hipblasComputeType_t`) is handled on main by `include/strata/hip_compat/cublas_v2.h`, which
  maps through rocBLAS directly; see also #2459 for the missing `cublasSgemm` etc. aliases (already on main).
- #4982 (nerdoc): first benchmark of Qwen3.8-Flash-Next on Wave 10-class hardware across a ~2 h load test.

## Install from source

Linux with the kernel's amdgpu driver, access to `/dev/kfd` and the card's `/dev/dri/renderD*`, a C++ compiler
and git. The runtime libraries (rocBLAS / hipBLAS) must actually contain gfx900 support: a compiler accepting
`gfx900` alone is not enough. The easiest source of a gfx900-capable ROCm is AMD's TheRock wheels (as setup
installs the normal backends):

```sh
# a venv with the gfx900 wheel set (pinned nightly that the MI25 validation ran)
python3 -m venv .venv-gfx900
.venv-gfx900/bin/pip install -r requirements-build.txt
.venv-gfx900/bin/pip install --index-url https://rocm.nightlies.amd.com/v2/gfx900/ \
  'rocm[libraries,devel]==7.14.0a20260612'
RS=.venv-gfx900/bin/rocm-sdk
HP=$($RS path --root); CORE=$(find .venv-gfx900 -type d -name _rocm_sdk_core | head -n1)
LIBW=$(find .venv-gfx900 -type d -name _rocm_sdk_libraries_* -not -path '*core*' | head -n1)

cmake -S . -B build-gfx900 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_HIP_GFX900=ON -DSTRATA_BUILD_TESTS=ON -DSTRATA_GGML_DIR=third_party/llama.cpp \
  "-DCMAKE_PREFIX_PATH=$HP;$LIBW;$CORE" \
  -DCMAKE_HIP_COMPILER=$HP/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$HP \
  -DCMAKE_HIP_FLAGS="--rocm-path=$HP --rocm-device-lib-path=$HP/lib/llvm/amdgcn/bitcode"
cmake --build build-gfx900 --target strata -j8
```

With tests on, `hip_wave64_intrinsics` and the BF16/FP16 GEMM check run on the card (ctest `-R hip_wave64|hip_prefill_gemm`).
A ROCm 7 installation in `/opt/rocm` that has the gfx900 libraries and compiler works too (`HP=/opt/rocm CORE=/opt/rocm LIBW=/opt/rocm`).

## From Docker

`Dockerfile.gfx900` builds the whole image (engine packaged as a component + Terminal + runtime), nothing to
compile locally:

```sh
docker build -t strata:gfx900 -f Dockerfile.gfx900 . --build-arg STRATA_COMMIT=$(git rev-parse --short HEAD)
```

Run it (the model lives in `/data` on a volume; two MI25s: `GPUS=0,1`). `STRATA_EXPERIMENTAL_GFX900=1` is the
Terminal's opt-in gate for this arch (`setup.py` treats it like the gfx1103/gfx1150 ones; the engine in the
image is already built for gfx900, and the engine itself needs no flag):

```sh
docker run -d --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --ulimit memlock=-1 \
  --shm-size 2g --stop-timeout 120 -p 127.0.0.1:8080:8080 -v strata-data:/data \
  -e STRATA_EXPERIMENTAL_GFX900=1 \
  -e MODEL=IQ1_M -e FAMILY=coder -e CONTEXT=65536 -e KV=int8 -e GPUS=0,1 strata:gfx900
```

## Two MI25 cards (32 GB VRAM), 24 GB of RAM, weights on a SATA SSD

2x MI25 (32 GB total, both HBM2) differs from a one-card 8 GB box: VRAM is what you have most of. The tuning
goal is to keep as many experts resident in VRAM as possible and let RAM only hold OS + engine overhead, while
the SSD serves the misses. Bounds to keep in mind:

- Per-card 16 GB: a two-card layer split (`--layer-split 27`, measured pattern of the 2x MI50 numbers,
  #3173) keeps one stage per card; `--pipeline-windows 2` overlapped proof windows a second window while the
  first card finished its last (+36% decode on 2x MI50 in the docs' measurement).
- `STRATA_SPLIT_OWN` and `STRATA_PREFILL_HELP` stay off: #2461 measured that when experts are (nearly) resident
  those two switches cost decode more than they gain prompt (the flag's buffers come out of the expert cache).
- LLama file tier: `STRATA_ARENA_MMAP=1` and `STRATA_STAGE_TRIM=1`, plus the wave64 attention switches
  `STRATA_GFX906_ATTN_QUERY_SWIZZLE=1` `STRATA_GFX906_ATTN_REDUCE12=1` (the same A/B switches the gfx906 cards
  check bitwise against the default path).

Serving command (Coder pack keeps 256 of the 512 experts; IQ1_M is ~29.6 GB). Two layers take the settings:
the Terminal CLI (`strata launch`) takes the model-facing flags, and everything the Terminal does not offer as
its own flag goes through `--set KEY=VALUE` (the Advanced table; it is saved with the preset and survives a
restart). A direct engine start (`build-gfx900/strata serve.server --help`) takes the engine's own flags.

```sh
# the Terminal way (also what the container runs): the engine flags ride on --set
# (STRATA_EXPERIMENTAL_GFX900=1 lets the Terminal's card checks accept gfx900; in the container pass it
# with -e and the STRATA_* variables reach the engine from the environment)
STRATA_EXPERIMENTAL_GFX900=1 STRATA_ARENA_MMAP=1 STRATA_STAGE_TRIM=1 STRATA_GFX906_ATTN_QUERY_SWIZZLE=1 STRATA_GFX906_ATTN_REDUCE12=1 \
strata launch --no-tui --family coder --model IQ1_M --gpu 0,1 --context 65536 --kv int8 \
  --set "--layer-split=27" --set "--pipeline-windows=2" --set "--trim-stage-weights=on" \
  --set "--mmap-experts=auto" --set "--expert-cache=auto" --set "--prefill=auto" \
  --set "--kv-resident=16384" --set "--vram-reserve-mib=768" --set "--adapt-every=0" --set "--pcie-frac=0"
```

`--expert-profile data/expert-profile-coder.bin` (the #4982 form) is NOT passed: the Coder pack carries its
own profile, and the low-RAM adoption of `strata launch` (`strata_cli/launch/adopt.py`) wires `--expert-profile`
to the pack's file together with `--mmap-experts` and `--expert-cache` automatically. Hand it only if you
prepared a profile yourself.

Same engine in the container (`STRATA_SET="--layer-split=27 --pipeline-windows=2 ..."` picks up the same
`--set` entries at start; the STRATA_* environment variables pass straight through to the engine).

Notes:

- `--vram-reserve-mib 768`: the #4982 box's value; raise it if the engine does not pick a spot that leaves a
  headroom for the draft/KV, or try `--expert-cache auto` with the card's reported free memory. Nobody on this
  tracker has measured a different value on Vega 10 yet.
- `--adapt-every 0`: no background expert-adaptation rounds; reads from the SATA SSD are too slow to hide, and
  with the VRAM-focused cache the residency does not change under a conversation anyway.
- `--pcie-frac 0`: the #2461 track's finding that PCIe overlaps do not pay on this old card family.
- Draft model (MTP) and pictures: dflash2 runs with `STRATA_DF2_PRIO=1` and `STRATA_DF2_FORK=0` default on
  this build (PR #4766 keeps the HIP behavior); the model weights of the drafter add VRAM and a
  4-6 GB SSD/RAM load. Pictures run on the CPU (`--vision cpu`; no GPU reader in this build).
- 24 GB of RAM at 30 GB of expert files mapped readonly (`--mmap-experts`): the page cache holds the hot set.
  Expect the FIRST request after a boot to read ~27 GB of experts from the SSD (a few t/s prefill, minutes);
  afterwards the state matches #4982's tables (which ran with 8 GB more RAM. Screen out the difference by
  comparing `expert cache hit` in timings).
- `HSA_OVERRIDE_GFX_VERSION` is NOT needed (gfx900 is the card's own name).

## What is measured where

## A rocBLAS table made on the MI25s (not a default)

`tools/hip/gfx900-rocblas-5.5.0.9811f2ea52.txt` is the solution table of the prompt path's FP16-out
GEMMs for one build of rocBLAS (5.5.0.9811f2ea52, the gfx900 wheels this image was validated with;
112 rows, 101 shapes whose default kernel was beaten, gains from 1.05x to ~12x on the tiny router
shapes, by `tune_rocblas` on 2026-11-10). It loads only through `STRATA_ROCBLAS_TUNING`, under the
same rule as the gfx1030 tables: the card's architecture and the full version string must match, and
only when the FP16 prompt route is on. The wave64 build compiles the loader now (the table calls are
behind `STRATA_ROCBLAS_AVAILABLE`, set by the cmake backend for gfx906-compat builds since this
commit); without the env the engine never opens the file.

`STRATA_HIP_PROMPT_F16=1` alone is the lever that switches the prompt's 16-bit GEMMs to FP16; the
numbers change (FP16 rounding of each GEMM's output). On this box the full launch measured slower
end to end than the default (two alternating rounds per arm, `STRATA_PREFILL_CPU_SHARE=auto` matched
on both sides): the 6,345-token long prompt read at 50.9 vs the base's 82.0 tok/s and decode went
14.3 vs 22.5, with the short chats a wash. It is not in compose; the FP16 route and the table stay
opt-in for anyone who wants to test them. Raw rows: `bench/results/2026-11-10-mi25-tier2/README.md`.

## What is measured where

Vega 10 numbers so far: one historical engine log at 39.9 prompt / 31.6 decode tok/s on one MI25 (0.1.39),
the #4982 table on a Vega 64 (14-16 tok/s decode at 8GB card, experts from RAM), and none on 2x MI25 yet.
If you benchmark this image, follow docs/BENCHMARK_PROTOCOL.md and note the expert-cache hit rate with your
RAM and SSD - that is the number that decides whether decode is GPU-bound or disk-bound here.
