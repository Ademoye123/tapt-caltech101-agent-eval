# Task: Reproduce the TAPT robust-inference evaluation (Caltech101 row, CVPR 2025)

You are asked to reproduce an evaluation from the paper:
**"TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models"** (CVPR 2025, arXiv:2411.13136), available bundled as `environment/paper/paper.pdf` (and `paper.md`, a plain-text rendering of the same PDF).

This is a **Paper-only** task: code and assets come from the authors' pinned repositories and from the team-bundled assets mounted in this environment. You are NOT asked to train anything.

## 1. What to reproduce

The paper's zero-shot adversarial-robustness evaluation of the **TAPTVLI** test-time prompt-tuning method, on the **Caltech101** dataset row (Table 1 / Table 2 in the paper; "Caltech101" column, AdvIVLP and CleanIVLP rows).

Concretely, produce the following three numbers for the **AdvIVLP** checkpoint (robust test-time variant) **and** the **CleanIVLP** checkpoint (clean test-time variant), on the Caltech101 test set:

- `clean_top1`: Top-1 accuracy (%) of the CLIP ViT-B/16 zero-shot model **before** test-time optimization (linear-probe head initialization epoch 100).
- `robust_top1`: Top-1 accuracy (%) under **AutoAttack** (eps = 1/255, 100 steps) after **TAPTVLI** test-time prompt tuning.
- `tap_before_top1` and `tap_after_top1`: Top-1 accuracy (%) after TAPTVLI tuning WITHOUT attack, measured at the initial adapters (epoch 0) and at the final adapters (epoch 100).

The expected quantitative values are the ones reported in the paper's tables for the corresponding rows/columns; the verifier compares your numbers against a bundled reference oracle (the team's execution of the same code paths on identical assets), with tolerance. Engineering honesty is expected: do not fabricate numbers; if a run fails, log the failure and report what you did observe.

## 2. Environment and bundled assets

The environment image has:

- The authors' **TAPT** repo at pinned commit `14d8d0e` (patched for load compatibility): `/opt/TAPT`
- The authors' **MMoP** repo (contains the PromptLearning source used by TAPT) at pinned commit `44e38d4`, patched with a minimal AutoAttack-save fix (same patch the team used): `/opt/MMoP`
- **Dassl.pytorch** pinned at `c61a1b5`, patched so `torch.load(weights_only=False)` (torch 2.6 compatibility): `/opt/Dassl.pytorch`
- Python 3.11, PyTorch 2.6.0 + cu12.4, and the pinned dependency set from the repos' requirements.

Bundled data (read-only) at `/workspace/data/`:

- `caltech-101/101_ObjectCategories/` — the Caltech101 image corpus with the standard train/test split **already applied** in `caltech-101/split_zhou_Caltech101.json` (the CoOp/`split_zhou` scheme the paper uses; train = 16 shots per class, test = everything else).
- `split/` — the bundled `split_zhou_Caltech101.json` (same as above; kept at a fixed path for the pipeline).
- `stats/vitb16/VLI_probs.pt`, `VLI_entropy.pt`, `VLI_energy.pt`, `VLI_op_preds.pt` — the four Caltech101 VLI statistics used by the MMoP AutoAttack wrapper (bundled calib pkl equivalent).

Bundled checkpoints at `/workspace/assets/` (read-only; these are the team's outputs of the paper's own training runs, **exact** checkpoints the paper evaluation uses):

- `output/train/imagenet/AdvIVLP/vit_b16_c2_ep100_batch32_2+2ctx_9depth_16shots/seed1/VLPromptLearner/model.pth.tar-100` — AdvIVLP checkpoint (epoch 100)
- `output/train/imagenet/AdvIVLP/vit_b16_c2_ep100_batch32_2+2ctx_9depth_clean_16shots/seed1/VLPromptLearner/model.pth.tar-100` — CleanIVLP checkpoint (epoch 100)
- `output/evaluation/auto/AdvIVLP/vit_b16_c2_ep100_batch32_2+2ctx_9depth_16shots/caltech101/seed1/100/Caltech101_adv_dataset.pkl` — bundled AA pkl (AdvIVLP; AutoAttack Linf eps=1/255)

Insightful reference: the paper describes test-time adapters (TAPTVLI) at **9 depth** layers, prompt length 2, and AutoAttack at eps 1/255.

## 3. Steps (use the authors' code paths: `/opt/MMoP`, `/opt/TAPT`)

1. **Run the TAPTVLI test-time eval** (clean and adversarial) from the TAPT repo:
   - `/opt/TAPT` → `scripts/` contains the eval runner (e.g. `run_tap_tli.sh` / the python entry `scripts/...`); run with `DATASET=caltech101`, `TRAINER=VLPromptLearner`, `MODEL=vit_b16`, `CFG=vit_b16_c2_ep100_batch32_2+2ctx_9depth_16shots` (adv) and `..._clean_16shots` (clean), `SEED=1`, and the bundled checkpoint as initialization, evaluation-only (`--eval-only` semantics), output directory `/workspace/submission/evaluation/`.
   - The adversarial input for the robust eval is the bundled pkl at `/workspace/assets/output/evaluation/auto/AdvIVLP/vit_b16_c2_ep100_batch32_2+2ctx_9depth_16shots/caltech101/seed1/100/Caltech101_adv_dataset.pkl` (AutoAttack, Linf eps=1/255, 100 steps, on the Caltech101 test split).
2. **Write the report** to `/workspace/submission/` (schema below).

You have **30 minutes** of wall-clock. Prioritize: if AA generation risks exceeding budget, reuse the bundled pkl (noted above) and budget the time on the TAPTVLI evals, which are the graded numbers.

## 4. Required output schema

All outputs must land under `/workspace/submission/`:

- `metrics.json` — one JSON object:
  ```json
  {
    "dataset": "caltech101",
    "method": "TAPTVLI",
    "checkpoint": "AdvIVLP",
    "checkpoint_name": "vit_b16_c2_ep100_batch32_2+2ctx_9depth_16shots/seed1",
    "clean_top1": 0.0,
    "robust_top1": 0.0,
    "tap_before_top1": 0.0,
    "tap_after_top1": 0.0,
    "attack": "auto",
    "eps": "1/255",
    "steps": 100
  }
  ```
  Produce one such object per checkpoint (AdvIVLP and CleanIVLP) — `metrics_adv_ivlp.json` and `metrics_clean_ivlp.json` (or a single `metrics.json` with an `"adv"` and `"clean"` key, your choice; keep keys exact). For CleanIVLP, `robust_top1` is optional (it is not a paper-table column; set it to `null` if you do not report it).
- `REPORT.md` — a human-readable summary: commands run, exact `model.pth.tar-100` paths used, pkl provenance (bundled vs generated and the generation command), top-1 numbers per stage, and any deviations from the steps above.
- `evaluation/` — the natural output files from your TAPT runs (the `output/` trees from the repos' own logging).

The verifier rewrites nothing inside `/workspace/submission/` except its own `evaluation/` artifacts; please do not pre-create files there that conflict.

## 5. Constraints

- Do NOT download or fetch any assets for this task beyond what is already mounted (`/workspace/data`, `/workspace/assets`): in particular, do not use HuggingFace to obtain CLIP weights or ImageNet assets, and do not clone any paper code repositories.
- Use the pinned commits as given. Do not `pip install` or `apt-get install` new packages (the environment contains the pinned deps; if you discover a genuinely missing package, record it in REPORT.md rather than fishing for a live network solution).
- Do not modify anything outside `/workspace/submission/` and temp dirs.
- Report honestly. A correct failure mode is a report of what you attempted; a failure is a fabricated number.