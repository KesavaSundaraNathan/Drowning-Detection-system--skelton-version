"""
training/behavior — learned behaviour classifier.

    python scripts/build_behavior_dataset.py ...   # labelled clips → windows (.npz)
    python scripts/train_behavior_model.py ...     # leave-one-video-out CV, ONNX export

See documentation/ML_CORE_V2.md for the data protocol.
"""
