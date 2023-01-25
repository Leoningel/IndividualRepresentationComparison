#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail
if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

END=1000
for i in $(seq 0 $END); do
    for r in 0 1 2; do
        for d in 0 1 2 3 4; do
            for f in 0 1 2; do
                sbatch --job-name="s${r}_${f}_${d}_${i}"  parallel_comparison_synthetic.sh --seed $i --representation $r --depth $d --fitness $f
            done
        done
    done
done