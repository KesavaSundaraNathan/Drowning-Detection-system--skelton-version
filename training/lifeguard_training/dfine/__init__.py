"""
training/dfine — the zone-aware person detector (ML core v2).

D-FINE (github.com/Peterande/D-FINE, Apache-2.0; pure PyTorch — no mmcv, so
it trains on the project's torch 2.5 without an MSVC build) fine-tuned from
its **COCO-only** checkpoint (`dfine_<size>_coco.pth`; NEVER the
`*_obj365*` / `*_obj2coco*` checkpoints — Objects365 is non-commercial) on
the client's CVAT-reviewed frames with two classes:

    0 person_in_water      1 person_out_of_water

    prepare.py   our COCO export (scripts/export_training_set.py --classes zone) → D-FINE layout
    config.py    D-FINE YAML for this dataset (includes the repo's own custom config)
    export.py    trained checkpoint → ONNX with RAW logits/boxes (FORMAT_DETR) + metadata,
                 registered in models/MODELS.json with its provenance

Driven end to end by scripts/train_detector_dfine.py.
"""

CLASSES = ("person_in_water", "person_out_of_water")
FORBIDDEN_CHECKPOINT_MARKERS = ("obj365", "obj2coco", "objects365")
