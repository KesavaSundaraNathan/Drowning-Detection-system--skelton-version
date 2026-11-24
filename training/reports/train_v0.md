# v0 training report (2026-10-29 22:54)

- dfine-L v0: checkpoint `C:\lifeguard\training\work_dirs\dfine_l_v0\best_stg1.pth` -> `C:\lifeguard\training\work_dirs\onnx\dfine_l_v0.onnx`; export rc 0, Life Guard decode check **PASS**
- rtmdet-L v0: checkpoint `C:\lifeguard\training\work_dirs\rtmdet_l_v0\best_coco_bbox_mAP_epoch_55.pth` -> `C:\lifeguard\training\work_dirs\onnx\rtmdet_l_v0.onnx`; export rc 0, Life Guard decode check **PASS**

## Val (reviewed seed frames; never the test split)

50 frames, 719 people (84 under 32 px tall), IoU 0.5
| model | AP50 | recall@0.1fppi | recall@0.5fppi | recall@1.0fppi | small_recall@0.1fppi | small_recall@0.5fppi | small_recall@1.0fppi | class_acc |
|---|---|---|---|---|---|---|---|---|
| rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c_dynbatch.onnx | 0.342 | 0.057 | 0.136 | 0.188 | 0.000 | 0.024 | 0.048 | — |
| lifeguard_dfine_m_devtrain_v1_dynbatch.onnx | 0.281 | 0.054 | 0.147 | 0.182 | 0.024 | 0.060 | 0.095 | 0.678 |
| dfine_l_v0.onnx | 0.540 | 0.207 | 0.373 | 0.423 | 0.071 | 0.202 | 0.250 | 0.943 |
| rtmdet_l_v0.onnx | 0.615 | 0.170 | 0.293 | 0.357 | 0.095 | 0.179 | 0.226 | 0.960 |

Baselines: the app's current RTMDet-m (COCO) and the dev-only D-FINE-M. Curves use no score floor.
Not registered in backend\models\MODELS.json; the owner decides (Phase 10).
