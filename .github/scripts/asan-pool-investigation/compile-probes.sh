#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
set -euo pipefail

artifact_root=$(realpath "${1:?artifact directory}")
output_root=$(realpath "${2:?output directory}")
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=$(realpath "$script_root/../../..")
asan_runtime=$(find "$artifact_root/lib/llvm/lib/clang" -name 'libclang_rt.asan*.so' | sort | head -1)
test -n "$asan_runtime"
asan_dir=$(dirname "$asan_runtime")
compiler="$artifact_root/lib/llvm/bin/clang++"
"$compiler" --version
for kind in hip blas; do
    definitions=()
    libraries=(-lamdhip64)
    if [[ "$kind" == blas ]]; then
        definitions=(-DPROBE_BLAS)
        libraries=(-lhipblaslt -lamdhip64)
    fi
    "$compiler" --no-default-config -std=c++17 -g -O0 -Wall -Wextra -Werror \
        -D__HIP_PLATFORM_AMD__ "${definitions[@]}" -isystem "$artifact_root/include" \
        -isystem "$source_root/projects/hipblas-common/library/include" \
        -fsanitize=address -shared-libasan "$script_root/startup.cpp" \
        -L "$artifact_root/lib" "${libraries[@]}" \
        -Wl,-rpath,"$artifact_root/lib" -Wl,-rpath,"$asan_dir" \
        -Wl,-rpath-link,"$artifact_root/lib" \
        -o "$output_root/startup-$kind"
    readelf -d "$output_root/startup-$kind"
    sha256sum "$output_root/startup-$kind"
done
