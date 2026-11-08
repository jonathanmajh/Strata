#!/bin/bash
# Linux AMD Vega 10 (gfx900: Instinct MI25 / Vega 56 / 64) engine package: strata + the CPU picture reader (strata-vision),
# the ROCm libraries it loads and the rocBLAS kernel data for gfx900, packaged as an engine component.
# Experimental wave64 build (STRATA_HIP_GFX900): the same compat layer as the gfx906 build, compiled for gfx900.
# Like packaging/build_hip_linux.sh, run on a Linux box that has ROCm (AMD's TheRock Python wheels are enough):
#
#   VARIANT=gfx900 ARCHS=gfx900 SRC=<this checkout> LLAMA=<llama.cpp checkout at the pinned commit> \
#     HP=<ROCm devel root> CORE=<ROCm core root> LIBW=<ROCm libraries root> \
#     packaging/build_gfx900_linux.sh
#
# TheRock: HP/_rocm_sdk_devel -> root (llvm, include, lib), CORE/_rocm_sdk_core, LIBW/_rocm_sdk_libraries_<family>/lib
# holds the family's shared libraries (rocBLAS, hipBLAS, comgr, ...).  Only the roots that differ need setting.
#   OUT=<work folder>  PKGVER=0.1.58  COMMIT=<source commit for the build id>  JOBS=16
#   NO_PACKAGE=1                   stop after the stage folder
#   VISION_CPU=1|0                 also build strata-vision (the CPU picture reader; default 1)
set -u
V=${VARIANT:?VARIANT=gfx900}
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
LLAMA=${LLAMA:?LLAMA=<llama.cpp checkout at the pinned commit>}
HP=${HP:?HP=<ROCm devel root>}
CORE=${CORE:-$HP}
LIBW=${LIBW:-$HP}
OUT=${OUT:-$PWD/work-$V}
PKGVER=${PKGVER:-0.1.58}
COMMIT=${COMMIT:-unknown}
JOBS=${JOBS:-16}
ARCH=${ARCHS:-gfx900}
VISION_CPU=${VISION_CPU:-1}
if [ "$ARCH" != gfx900 ]; then echo "REFUSING: this script builds Vega 10 only; other archs use build_hip_linux.sh" >&2; exit 1; fi
mkdir -p "$OUT"; S=$OUT/build.status
st() { echo "$* $(date +%T)" >> "$S"; }
chk() { local rc=$1; shift; st "$* rc=$rc"; [ "$rc" = 0 ] || { st "ABORT $* failed (rc=$rc)"; exit 1; }; }
ID9=${COMMIT:0:9}
if [ "$COMMIT" != unknown ] && [ ! -e "$SRC/.git" ] && [ ! -f "$SRC/BUILD_ID" ]; then printf '%s\n' "$ID9" > "$SRC/BUILD_ID"; fi
: > "$S"
export HIP_PLATFORM=amd HIP_COMPILER=clang HIP_RUNTIME=rocclr ROCM_PATH=$HP HIP_PATH=$HP
SYSDEPS=""; [ -d "$HP/lib/rocm_sysdeps" ] && SYSDEPS=";$HP/lib/rocm_sysdeps"
HIPDEFS=(-DCMAKE_HIP_COMPILER=$HP/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$HP "-DCMAKE_PREFIX_PATH=$HP;$LIBW;$CORE$SYSDEPS"
         "-DCMAKE_HIP_FLAGS=--rocm-path=$HP --rocm-device-lib-path=$HP/lib/llvm/amdgcn/bitcode")
st "start $V ($ARCH) commit $COMMIT"
cd "$SRC" || exit 1
# the engine: the wave64 compat layer for gfx900 (STRATA_HIP_GFX900 reuses the gfx906 path's code)
cmake -G Ninja -S . -B "$OUT/b" -DCMAKE_BUILD_TYPE=Release -DSTRATA_HIP_GFX900=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_HIP_ARCHITECTURES=$ARCH "${HIPDEFS[@]}" -DSTRATA_GGML_DIR="$LLAMA" -DSTRATA_PORTABLE=ON \
  "-DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath,'\$ORIGIN' -Wl,--enable-new-dtags" > "$OUT/cfg.log" 2>&1
chk $? "engine configured"
nice -n 10 cmake --build "$OUT/b" --target strata -j "$JOBS" > "$OUT/build.log" 2>&1
chk $? "engine built"
[ -x "$OUT/b/strata" ] || { st "ABORT no strata"; exit 1; }
# note: the plain engine binary only carries STRATA_VERSION (0.1.58), not a build id (that is the dsv41 target)
BINS=( "$OUT/b/strata" )
if [ "$VISION_CPU" = 1 ]; then
  cmake -G Ninja -S tools/vision -B "$OUT/bv" -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR="$LLAMA" -DSTRATA_VISION_CUDA=OFF \
    -DSTRATA_PORTABLE=ON > "$OUT/cfg-vision.log" 2>&1
  nice -n 10 cmake --build "$OUT/bv" --target strata-vision -j "$JOBS" > "$OUT/build-vision.log" 2>&1
  chk $? "cpu reader built"
  [ -x "$OUT/bv/bin/strata-vision" ] || { st "ABORT no strata-vision"; exit 1; }
  BINS+=( "$OUT/bv/bin/strata-vision" )
fi
STAGE=$OUT/stage; rm -rf "$STAGE"; mkdir -p "$STAGE"
cat > "$STAGE/BUILD.json" <<EOJ
{"version": "$PKGVER", "source": "f-amd-gfx900 $COMMIT", "variant": "$V", "archs": ["$ARCH"], "portable": true, "vision": "cpu"$( [ "$VISION_CPU" = 1 ] && echo ',"vision_cpu": "strata-vision"' ) }
EOJ
# the libraries the engine loads, collected from the ROCm roots (rocBLAS kernel data for gfx900 rides along)
for b in "${BINS[@]}"; do
  python3 packaging/collect_libs.py --binary "$b" --search "$LIBW/lib" --search "$HP/lib" --search "$CORE/lib" --search "$CORE/lib/llvm/lib" ${SYSDEPS:+--search "$HP/lib/rocm_sysdeps/lib"} \
    --also 'libamd_comgr.so*' --also 'libhiprtc*.so*' --also 'librocblas.so*' --also 'libhipblas.so*' \
    --data "$LIBW/lib/rocblas/library=rocblas/library" --data-arch "$ARCH" \
    --out "$STAGE" >> "$OUT/collect.log" 2>&1
  chk $? "collected for $(basename "$b")"
done
cp "${BINS[@]}" "$STAGE/"
for f in "$STAGE"/*.so "$STAGE"/*.so.*; do [ -f "$f" ] && [ ! -L "$f" ] && patchelf --set-rpath '$ORIGIN' "$f"; done
for f in "$STAGE/strata" "$STAGE"/strata-vision; do
  [ -f "$f" ] || continue
  st "RUNPATH $(basename "$f"): $(readelf -d "$f" | grep -E 'RPATH|RUNPATH' | cut -d'[' -f2 | tr -d ']' | paste -sd' ')"
done
[ -n "${NO_PACKAGE:-}" ] && { st "STAGE-DONE"; exit 0; }
rm -rf "$OUT/components"; mkdir -p "$OUT/components/engine"
python3 packaging/package_engine.py --src "$STAGE" --backend hip --version "$PKGVER" --out "$OUT/components/engine" --arch "$ARCH" \
  --variant "$V" --build-id "$COMMIT+rocm" --tree \
  --notes "AMD Vega 10 (gfx900) wave64 build: Instinct MI25, Vega 56 / 64. Experimental: the gfx906 compat layer $( [ "$VISION_CPU" = 1 ] && echo ", CPU picture reader" )" > "$OUT/package.log" 2>&1
chk $? "packaged"
cat "$OUT/package.log" >> "$S"
st "BUILD-DONE"
