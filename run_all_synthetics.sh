#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

for d in 0 1 2 3 4; do
    for f in 0 1 2; do
        sbatch --job-name="s_${f}_${d}"  parallel_synthetic.sh --depth $d --fitness $f
    done
done
