from bones.pipelines.train import train, load_datasets, main as train_main
from bones.pipelines.evaluate import evaluate_split, main as evaluate_main
from bones.pipelines.predict import predict, batch_predict, main as predict_main
from bones.pipelines.generate_splits import main as generate_splits_main

__all__ = [
    "train", "load_datasets", "train_main",
    "evaluate_split", "evaluate_main",
    "predict", "batch_predict", "predict_main",
    "generate_splits_main",
]
