#!/usr/bin/env bash
# After the v0.3 fine-tune: export -> Life Guard decode check -> crowd recall on the reviewed val frames -> owner's clips
# through the upload path (sprint dev) and the RTSP path (MediaMTX + FFmpeg). Run from the repo root in Git Bash.
set -u
S="${1:?scratch dir}"
CK=$(ls training/work_dirs/dfine_l_v03/best_stg2.pth training/work_dirs/dfine_l_v03/best_stg1.pth training/work_dirs/dfine_l_v03/last.pth 2>/dev/null | head -1)
echo "checkpoint: $CK"
ONNX=training/work_dirs/onnx/dfine_l_v03.onnx
training/.venv/Scripts/python.exe training/tools/export_v0.py dfine -c training/configs/dfine_l_v03.yml --checkpoint "$CK" --out "$ONNX" || exit 1
export PYTHONPATH=backend LIFEGUARD_ENV=development LIFEGUARD_DATA_DIR='C:\LifeGuardDev'
venv/Scripts/python.exe training/tools/verify_v0_onnx.py "$ONNX" || exit 1
venv/Scripts/python.exe training/tools/eval_v0.py --ann training/data/coco/seed_r1/val_clean.json \
  --model training/work_dirs/onnx/dfine_l_v01.onnx --model "$ONNX" --out "$S/val_v03.json" 2>/dev/null | tail -5
venv/Scripts/python.exe -m lifeguard.eval.sprint --clips eval/sprint_v1/clips.json --split dev --out "$S/sprint_v03" \
  --det-model "$ONNX" --provider tensorrt 2>/dev/null | grep -v '^ *"'
venv/Scripts/python.exe tools/rtsp_validate.py --split dev --out "$S/rtsp_v03" --det-model "$ONNX" --max-seconds 120 2>/dev/null \
  | grep -v '^ *"\|onnxruntime'
