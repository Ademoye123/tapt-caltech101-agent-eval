# REVISION NOTES — TAPT-Caltech101-AgentEval

Submission record for the TAPT (CVPR 2025, arXiv:2411.13136) robust-eval task,
Caltech101 row. This package contains the complete enclosed environment, the
bundled assets, and the verifier plumbing.

## Task scope

Reproduce the paper's zero-shot robust-evaluation numbers on the **Caltech101**
row (Tables 1/2): with the bundled **AdvIVLP** and **CleanIVLP** checkpoints,
run the TAPTVLI test-time prompt-tuning eval (clean and AutoAttack eps=1/255)
and report `clean_top1`, `robust_top1`, `tap_before_top1`, `tap_after_top1`
per the contract in `instruction.md`. No training is required of the agent —
all training-derived assets are bundled.

## Bundled

| Item | Location in env | Hash (sha256) |
|---|---|---|
| Paper (CVPR 2025, arXiv:2411.13136) | `/workspace/paper/paper.pdf` (+ `paper.md`) | — |
| TAPT repo @ `14d8d0e` | `/opt/TAPT` | — |
| MMoP repo @ `44e38d4` (patched) | `/opt/MMoP` | — |
| Dassl.pytorch @ `c61a1b5` (patched) | `/opt/Dassl.pytorch` | — |
| AdvIVLP checkpoint (ImageNet 16-shot, ep100) | `/workspace/assets/output/train/imagenet/AdvIVLP/..._16shots/seed1/VLPromptLearner/model.pth.tar-100` | `2d99eb62…ab160` |
| CleanIVLP checkpoint (same, clean) | `/workspace/assets/output/train/imagenet/AdvIVLP/..._clean_16shots/seed1/VLPromptLearner/model.pth.tar-100` | `43368008…3d99b` |
| Caltech101 images + CoOp split | `/workspace/data/caltech-101/` | — |
| VLI stats (Caltech101-calibrated, real checkpoints) | `/workspace/assets/stats/vitb16/` (4 files) | see below |
| AA adversarial pkl (ep100, AdvIVLP) | `/workspace/assets/output/evaluation/auto/AdvIVLP/.../Caltech101_adv_dataset.pkl` | — |

Stats hashes:

```
87dfab3788d819d67ab7e8a978abff70cd737c54229e512eecb11a8612006137  assets/stats/vitb16/VLI_means_vitb16_train_adv.pt
64d289f0d1eb09b9c3818c5d203c66b1b5735e1532564e0e2be68c18b418bb2d  assets/stats/vitb16/VLI_vars_vitb16_train_adv.pt
113a2060d5813e2357491615a22268b5bd0a9090bb27b24c1cad4737c499d90f  assets/stats/vitb16/VLI_means_vitb16_train_clean.pt
dbd36d906b687083110120979952f342baca3da4ae457c9a11fdb744495e16aa  assets/stats/vitb16/VLI_vars_vitb16_train_clean.pt
```

## Evidence on record

- Environment is fully baked: pinned deps (torch 2.6.0+cu124, Python 3.11),
  the three author repos at pinned commits, two minimal compatibility patches
  (Dassl `weights_only=False` for torch 2.6; MMoP AutoAttack pkl save).
  Patches and the dependency lock live in the workspace provenance record.
- Pipeline smoke (16-image end-to-end) passed on the same code paths.
- Two-sample AutoAttack determinism test: two identical generation runs
  produce elementwise-identical pkls (PyTorch-seeded; fp32 precision — the
  production setting).
- Stats validated against the asset contract: shape `(12,199,768)` x4, via
  `validate_stats_asset.py`.
- Checkpoint hashes recorded above and verified at packaging.

## Corrections applied during production (author-repo behavior, not the paper)

- **C-AA1**: AutoAttack's Square attack crashes under fp16 model precision
  (`autoattack/square.py` L274 dtype mismatch); the AA lane runs
  `TRAINER.AdvIVLP.PREC fp32` (also the two-sample determinism setting).
- **C-S2**: the statistics are produced with the authors' cal flow on
  Caltech101 train (the flow the paper applies on ImageNet train);
  `TAPT.VIS_*` contract and shapes are identical.

## Verification contract

`solution/solve.sh` is the reference execution: it runs the two TAPTVLI evals
(clean + AA) against the bundled assets and emits the same metrics schema the
agent must produce (single code path for verifier and agent). `tests/`
contains the rubric (`rubrics.json`), the verifier entry (`evaluate.py`) and
its runner (`test.sh`).