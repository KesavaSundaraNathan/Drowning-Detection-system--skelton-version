# Third-party notices

Life Guard's detection engine is a top-down pipeline, **RTMDet → ByteTrack → RTMPose**, served with ONNX Runtime.
This file carries the licence terms and copyright notices of every third-party component the shipped software
contains or loads. The full Apache License 2.0 text is in [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt).
Automated gates: `tools/check_licenses.py` (packages + `backend/models/MODELS.json`) and `tools/verify_models.py`
(model file hashes).

## OpenMMLab models and code (Apache License 2.0)

    Copyright (c) OpenMMLab. All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use these files except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

| Component | Use in Life Guard | Upstream | Weights / training data |
|---|---|---|---|
| **RTMDet-m** (MMDetection) | Person detector, `rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c*.onnx` (ONNX export of the published checkpoint, dynamic batch) | https://github.com/open-mmlab/mmdetection | COCO 2017 only (no Objects365) |
| **RTMPose-m** (MMPose) | Top-down pose on every RTMDet box, `rtmpose-m_8xb256-420e_coco-256x192_dynbatch.onnx` (ONNX export of `rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192`) | https://github.com/open-mmlab/mmpose | Fine-tuned on COCO 2017 keypoints only. Disclosed: OpenMMLab initialised its backbone from a model pre-trained on AI Challenger + COCO. **The `body7` RTMPose weights are never used.** |
| MMDeploy | Offline ONNX export only (`tools/export_rtmpose.py`); not shipped | https://github.com/open-mmlab/mmdeploy | - |
| MMEngine, MMCV | Offline export environment only (`requirements/pose_export.in`); not shipped | https://github.com/open-mmlab | - |

Code ported from the Life Guard v2 baseline that follows OpenMMLab / rtmlib implementations (letterbox, SimCC
decoding, top-down crop parameters, NMS in `backend/lifeguard/vision/inference/`) is used under the same terms.

**Redistribution.** When the ONNX files (or TensorRT engines built from them) are shipped, keep this file and
`LICENSES/Apache-2.0.txt` alongside them; modified files must say they were changed (the ONNX files are exports,
the `_dynbatch` files additionally have a dynamic batch axis - `tools/make_dynamic_batch.py`).

## COCO dataset

The detector and pose weights were trained on COCO 2017. COCO annotations are licensed CC BY 4.0
(© COCO Consortium, https://cocodataset.org/#termsofuse); the images are from Flickr under their own licences.
The images are not redistributed by Life Guard.

## ByteTrack (MIT License) - `backend/lifeguard/vision/tracking/bytetrack/`

The multi-object association (two-stage high/low score matching, Kalman filter in (cx, cy, aspect, height) space,
lost-track buffer) follows ByteTrack (Zhang et al., ECCV 2022) and its reference implementation
https://github.com/ifzhang/ByteTrack:

    MIT License
    Copyright (c) 2021 Yifu Zhang

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
    associated documentation files (the "Software"), to deal in the Software without restriction,
    including without limitation the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or
    substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
    NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
    OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
    CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Runtime libraries

| Component | Licence | Notes |
|---|---|---|
| ONNX Runtime | MIT | Copyright (c) Microsoft Corporation |
| OpenCV | Apache-2.0 | |
| NumPy, SciPy | BSD-3-Clause | |
| **NVIDIA TensorRT** (optional) | NVIDIA Software License Agreement - **proprietary, not open source** | Used only through ONNX Runtime's TensorRT execution provider when installed on the deployment GPU (owner decision 2026-10-14). Not bundled; engines are built on the target machine. Without it the same ONNX models run on CUDA, OpenVINO or CPU. |
| NVIDIA CUDA / cuDNN runtime wheels | NVIDIA licences (redistributable runtime) | `nvidia-*-cu12` wheels in `requirements/serving.lock` |

Every Python package and its licence is listed by `python tools/check_licenses.py --list`.
