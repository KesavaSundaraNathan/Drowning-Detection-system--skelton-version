# Licences of tools and weights used for training

Rule (MASTER_PROMPT §3.1): code **and** weights MIT or Apache-2.0; COCO-only checkpoints for RTMDet, D-FINE, RTMPose;
no Ultralytics (AGPL), no Roboflow, no cloud labelling. Every weight file is recorded here with its source and sha256
(first 16 hex) when downloaded.

## Code
| Tool | Version / commit | Licence | Where |
|---|---|---|---|
| PyTorch / torchvision | 2.7.1+cu128 / 0.22.1 | BSD-3 | `training\.venv` |
| MMEngine | 0.10.7 | Apache-2.0 | venv |
| MMCV / MMDetection / MMPose | 2.1.0 (compiled for sm_120) / 3.3.0 / 1.3.2 | Apache-2.0 | venv; MMCV source `training\third_party\mmcv` |
| D-FINE | Peterande/D-FINE @ 956d170 (2026-08-19) | Apache-2.0 | `training\third_party\dfine` |
| SAM 2 | facebookresearch/sam2 @ 2b90b9f (2024-12-15), no CUDA extension | Apache-2.0 | `training\third_party\sam2` |
| transformers (OWLv2, DINOv2) | 4.57.1 | Apache-2.0 | venv |
| OpenCV (headless) | 4.12.0.88 | Apache-2.0 | venv |
| Albumentations | 2.0.8 | MIT | venv |
| pycocotools / faster-coco-eval | 2.0.10 / 1.6.7 | BSD-2 / Apache-2.0 | venv |
| CVAT | v2.80.0 + Nuclio 1.16.3 serverless (SAM function) | MIT / Apache-2.0 | `training\third_party\cvat`, Docker |
| ByteTrack (in-repo, `backend\lifeguard\vision\tracking\bytetrack`) / OC-SORT | — | MIT | — |

## Weights
| File | Source | Trained on | Licence | sha256 (16) |
|---|---|---|---|---|
| `dfine_l_coco.pth`, `dfine_m_coco.pth` | github.com/Peterande/storage releases dfinev1.0 | COCO (HGNetv2 backbone ImageNet-pretrained) | Apache-2.0 | see `weights_manifest.txt` |
| `rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth` | download.openmmlab.com (MMDetection model zoo) | COCO (CSPNeXt backbone ImageNet-pretrained) | Apache-2.0 | see manifest |
| `sam2.1_hiera_base_plus.pt` | dl.fbaipublicfiles.com segment_anything_2/092824 | SA-1B + SA-V | Apache-2.0 (weights) | see manifest |
| `google/owlv2-base-patch16-ensemble` (HF cache) | huggingface.co | WebLI + O365 / VG | Apache-2.0 | HF cache |
| `facebook/dinov2-base` (HF cache) | huggingface.co | LVD-142M | Apache-2.0 | HF cache |
| `rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth` | download.openmmlab.com (MMPose model zoo) | AI Challenger + COCO (`pt-aic-coco`; CSPNeXt backbone ImageNet) | MMPose code Apache-2.0; **AIC data terms: owner chose option A on 2026-10-28 and will check AIC's terms before shipping** | `bdacdf0f`; the file-name tag `1352a4d2` is not its sha256 prefix (same for the repo's RTMPose-m), so it is not verifiable against a published hash |
| `sam_vit_h_4b8939.pth` (SAM 1 ViT-H) | dl.fbaipublicfiles.com/segment_anything | SA-1B | Apache-2.0 (code facebookresearch/segment-anything @ dca509f + weights). **Owner-approved 2026-10-28** for CVAT click-to-box only (CVAT's browser decoder needs SAM 1 embeddings); runs in a Nuclio function, float16, OOM falls back to CPU | md5 `4b8939…` matches the published name; sha256 `a7bf3b02…` |
| `IDEA-Research/grounding-dino-base` @ 12bdfa3 (HF cache, safetensors) | huggingface.co | O365, GoldG, Cap4M (IDEA) | Apache-2.0 | HF cache. **Owner-approved 2026-10-28 as a 4th labelling voter; trial on 50 val frames: found 59 % of the people 2-of-3 voters agree on (threshold 0.15) and no person all three missed, so not used.** Kept in the cache only |

Tools considered and not used: **FiftyOne** (Apache-2.0) bundles MongoDB, which is SSPL, not MIT / Apache: the crop
check runs in CVAT's Attribute Annotation Mode instead.

Notes:
- ImageNet-pretrained backbones are inside every "COCO" checkpoint; this is standard and was not called out by the
  prompt, but it is recorded here.
- The application's own `rtmpose-m_simcc-coco_pt-aic-coco_*.pth` (repo root) has the same AIC question.
- OWLv2 and DINOv2 are used only as labelling voters / feature extractors, never shipped.
