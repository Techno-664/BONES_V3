from bones.pipelines.evaluate import evaluate_split
from bones.pipelines.evaluate import main as evaluate_main
from bones.pipelines.generate_splits import main as generate_splits_main
from bones.pipelines.predict import predict
from bones.pipelines.predict import main as predict_main
from bones.pipelines.train import load_datasets, train
from bones.pipelines.train import main as train_main

__all__ = [
    "evaluate_main",
    "evaluate_split",
    "generate_splits_main",
    "load_datasets",
    "predict",
    "predict_main",
    "train",
    "train_main",
]
