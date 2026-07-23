from __future__ import annotations

import sys

from bones.logging import setup_logger

log = setup_logger("__main__")


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m bones <command>")
        print("Commands: train, evaluate, predict, generate_splits")
        return 1

    command = sys.argv[1]

    if command == "train":
        from bones.pipelines.train import main as train_main
        return train_main()
    elif command == "evaluate":
        from bones.pipelines.evaluate import main as evaluate_main
        return evaluate_main()
    elif command == "predict":
        from bones.pipelines.predict import main as predict_main
        return predict_main()
    elif command == "generate_splits":
        from bones.pipelines.generate_splits import main as generate_splits_main
        return generate_splits_main()
    else:
        log.error("Unknown command: %s", command)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
