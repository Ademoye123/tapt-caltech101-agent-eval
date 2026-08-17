#!/usr/bin/env bash
# ORACLE / reference execution for the TAPT-Caltech101 task.
# Runs inside the task image; produces /workspace/submission per instruction.md.
# The verifier runs this exact script (or parses ITS outputs) to derive the
# reference numbers, so agent and oracle share a single code path.
set -euo pipefail

TAPT_REPO=/opt/TAPT
MMOP_REPO=/opt/MMoP
DATA_ROOT=/workspace/data
ASSET_ROOT=/workspace/assets
OUT_ROOT=/workspace/submission
mkdir -p "${OUT_ROOT}/evaluation"

SEED=1
EPOCH=100
TRAINER=AdvIVLP
SHOTS=16
CFG=vit_b16_c2_ep100_batch32_2+2ctx_9depth
CFG_CLEAN=vit_b16_c2_ep100_batch32_2+2ctx_9depth_clean
STATS=/workspace/assets/stats/vitb16

TRAIN_DIR_ADV=${ASSET_ROOT}/output/train/imagenet/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}/VLPromptLearner
TRAIN_DIR_CLN=${ASSET_ROOT}/output/train/imagenet/${TRAINER}/${CFG_CLEAN}_${SHOTS}shots/seed${SEED}/VLPromptLearner
ADVDATA_DIR=${ASSET_ROOT}/output/evaluation/auto/${TRAINER}/${CFG}_${SHOTS}shots/caltech101/seed${SEED}/${EPOCH}
ADV_PKL=${ADVDATA_DIR}/Caltech101_adv_dataset.pkl

: ${CKPT:="adv"}   # "adv" | "clean" — selects the checkpoint and output paths

if [[ "${CKPT}" == "adv" ]]; then
    TRAIN_DIR=${TRAIN_DIR_ADV}
    SUFFIX=adv_ivlp
else
    TRAIN_DIR=${TRAIN_DIR_CLN}
    SUFFIX=clean_ivlp
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

run_eval() {
    # $1 = tapt cfg name   $2 = attack tag (auto|clean)
    local cfg_tapt="${1}" attack="${2}"
    local tag="${attack}"
    local out
    out="${ASSET_ROOT}/output/${tag}/TAPTVLI/${cfg_tapt}_0shots/TAPT_eps1_step1_0shots/${attack}/caltech101/seed${SEED}/${EPOCH}"
    python train.py \
        --root "${DATA_ROOT}" \
        --seed "${SEED}" \
        --trainer TAPTVLI \
        --dataset-config-file configs/datasets/caltech101.yaml \
        --config-file "configs/trainers/TAPTVLI/${cfg_tapt}.yaml" \
        --output-dir "${out}" \
        --model-dir "${TRAIN_DIR}" \
        --advdata-dir "${ADVDATA_DIR}/" \
        --load-epoch "${EPOCH}" \
        --tapt \
        DATASET.NUM_SHOTS 0 \
        TAPT.VIS_MEANS "${STATS}/VLI_means_vitb16_train_adv.pt" \
        TAPT.VIS_VARS "${STATS}/VLI_vars_vitb16_train_adv.pt" \
        TAPT.VIS_MEANS_CLEAN "${STATS}/VLI_means_vitb16_train_clean.pt" \
        TAPT.VIS_VARS_CLEAN "${STATS}/VLI_vars_vitb16_train_clean.pt"
    echo "${out}"
}

parse_top1() {
    # extract the LAST "Top-1 acc.: X%" line from a TAPT log; stdin = log.txt
    grep -oE "Top-1 acc\.: [0-9.]+%?" | tail -1 | grep -oE "[0-9.]+"
}

python_report() {
    python3 - "${1}" "${2}" "${3}" "${4}" "${5}" <<'PY'
import json, sys
out_root, suf, clean, before, after, robust = sys.argv[1:7]
clean, before, after, robust = (float(x) for x in (clean, before, after, robust))
with open(f"{out_root}/metrics_{suf}.json", "w") as f:
    json.dump({
        "dataset": "caltech101",
        "method": "TAPTVLI",
        "checkpoint": "CleanIVLP" if suf == "clean_ivlp" else "AdvIVLP",
        "checkpoint_name": suf,
        "clean_top1": clean,
        "robust_top1": robust,
        "tap_before_top1": before,
        "tap_after_top1": after,
        "attack": "auto",
        "eps": "1/255",
        "steps": 100,
    }, f, indent=2)
PY
}

echo "[solve] checkpoint=${CKPT} (${SUFFIX})"

clean_log=$(mktemp)
adv_log=$(mktemp)

clean_out=$(run_eval "TAPT_vit_b16_c2_ep100_batch32_2ctx_9depth_l1_cross_datasets_step1_clean" "clean")
cp "${clean_out}/log/log.txt" "${clean_log}"

if [[ -f "${ADV_PKL}" ]]; then
    adv_out=$(run_eval "TAPT_vit_b16_c2_ep100_batch32_2ctx_9depth_l1_cross_datasets_step1_adv" "auto")
    cp "${adv_out}/log/log.txt" "${adv_log}"
    robust=$(parse_top1 < "${adv_log}")
else
    robust="nan"
fi

clean=$(parse_top1 < "${clean_log}")
lines=$(grep -cE "Top-1 acc\.: [0-9.]+%?" "${clean_log}")
if [[ "${lines}" -ge 2 ]]; then
    before=$(grep -oE "Top-1 acc\.: [0-9.]+%?" "${clean_log}" | head -1 | grep -oE "[0-9.]+")
else
    before="${clean}"
fi

python_report "${OUT_ROOT}" "${SUFFIX}" "${clean}" "${before}" "${clean}" "${robust}"

cat >> "${OUT_ROOT}/REPORT.md" <<MD
## solve.sh run (checkpoint: ${SUFFIX})

- AdvIVLP: \`${TRAIN_DIR_ADV}/model.pth.tar-100\` $([[ -f ${TRAIN_DIR_ADV}/model.pth.tar-100 ]] && echo present || echo MISSING)
- CleanIVLP: \`${TRAIN_DIR_CLN}/model.pth.tar-100\` $([[ -f ${TRAIN_DIR_CLN}/model.pth.tar-100 ]] && echo present || echo MISSING)
- AA pkl: \`${ADV_PKL}\` $([[ -f ${ADV_PKL} ]] && echo present || echo MISSING)
- stats: ${STATS}/VLI_{means,vars}_vitb16_train_{adv,clean}.pt
- clean_top1 / tap_before / tap_after / robust_top1:
  ${clean} / ${before} / ${clean} / ${robust}
MD

echo "[solve] done: ${clean} / ${before} / ${clean} / ${robust}"