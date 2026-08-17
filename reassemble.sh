#!/usr/bin/env bash
# Rebuilds the original submission tree and zip from the split parts.
# Usage: bash reassemble.sh [--zip]
set -euo pipefail
cd "$(dirname "$0")"

fail=0
while read -r sum rel; do
    [[ "$rel" != *.zip && "$rel" != *.tar-100 ]] && continue
    parts=()
    for p in "$rel".part*; do
        [[ -f "$p" ]] && parts+=("$p")
    done
    if [[ ${#parts[@]} -gt 0 ]]; then
        cat "${parts[@]}" > "$rel.new"
        got=$(shasum -a 256 "$rel.new" | awk '{print $1}')
        if [[ "$got" == "$sum" ]]; then
            mv -f "$rel.new" "$rel"
            rm -f "$rel".part*
            echo "restored $rel"
        else
            echo "HASH MISMATCH: $rel"
            rm -f "$rel.new"
            fail=1
        fi
    fi
done < SHA256SUMS

if [[ "$fail" != 0 ]]; then
    echo "reassembly failed"; exit 1
fi
shasum -a 256 -c SHA256SUMS

if [[ "${1:-}" == "--zip" ]]; then
    cd ..
    zip -rq -9 TAPT-Caltech101-AgentEval.zip repo2push
    echo "built ../TAPT-Caltech101-AgentEval.zip"
fi
echo "all parts restored and verified"