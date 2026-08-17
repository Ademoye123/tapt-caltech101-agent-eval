# Data and asset provenance manifest

Everything the agent needs is bundled in this image; nothing has to be
downloaded at runtime. This file records public sources for auditability.
Expected external availability: >= 3 days before the revision deadline.

## Datasets bundled in this image

| Dataset | Public source | License | Bundled at |
|---|---|---|---|
| Caltech101 (images) | https://www.kaggle.com/datasets/imbikramsaha/caltech-101 or https://data.caltech.edu/records/mzrjq-6wc02 | Research-use | /workspace/data/caltech-101/101_ObjectCategories |
| Caltech101 CoOp split | https://github.com/KaiyangZhou/CoOp (split_zhou_Caltech101.json) | MIT | /workspace/data/split_zhou_Caltech101.json |

## Weights and generated assets (team-bundled; see REVISION_NOTES.md)

| Asset | Origin/derivation | Bundled at |
|---|---|---|
| AdvIVLP checkpoint (ImageNet 16-shot, ep100) | author-code training, team lane; sha256 see REVISION_NOTES.md | /workspace/assets/output/train/imagenet/AdvIVLP/.../VLPromptLearner/model.pth.tar-100 |
| CleanIVLP checkpoint (same, clean) | author-code training, team lane | /workspace/assets/output/train/imagenet/AdvIVLP/..._clean_16shots/.../model.pth.tar-100 |
| VLI stats (4 files) | author-code calibration pass (imagenet, ep100, eval-only, bootstrapped init) | /workspace/assets/stats/vitb16/ |
| Caltech101 AA pkl (eps 1/255, AdvIVLP) | MMoP eval-only --attacks auto, deterministic (PyTorch-seeded, fp32) | /workspace/assets/output/evaluation/auto/.../Caltech101_adv_dataset.pkl |

## Code (pinned commits, MIT)

- TAPT: https://github.com/xinwong/TAPT @ 14d8d0e — baked at /opt/TAPT
- MMoP: https://github.com/xinwong/MMoP @ 44e38d4 — baked at /opt/MMoP
- Dassl.pytorch: https://github.com/KaiyangZhou/Dassl.pytorch @ c61a1b5 — baked at /opt/Dassl.pytorch
- Patches (2): weights_only=False load compatibility; AutoAttack-pkl save — see stage2/patches in the provenance record

## Paper

- arXiv:2411.13136 (PDF bundled at /workspace/paper/paper.pdf; text at paper.md)

## Network posture

Internet is allowed but the hosts listed in network_blacklist.txt are denied
(arXiv, CVF, PapersWithCode, OpenReview, the author repositories, HuggingFace,
PyPI, pandas, docs). The env image is fully baked; no build-time fetch is needed.