import argparse
import warnings
from pathlib import Path

import numpy as np  
from prettytable import PrettyTable 
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

warnings.filterwarnings("ignore")

DEFAULT_DATA_YAML = "data/Visdrone2019_dataset.yaml"


def build_parser(default_split: str = "val", default_project: str = "runs/val") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLOv10 validation and reporting helper")
    parser.add_argument("--weights", required=True, help="Path to trained weights (.pt)")
    parser.add_argument("--data", default=DEFAULT_DATA_YAML, help="Dataset configuration YAML")
    parser.add_argument(
        "--split",
        default=default_split,
        choices=["train", "val", "test"],
        help="Dataset split used for evaluation",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for evaluation")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--project", default=default_project, help="Directory for evaluation outputs")
    parser.add_argument("--name", default="exp", help="Run name under the project directory")
    parser.add_argument(
        "--save-paper-metrics",
        dest="save_paper_metrics",
        action="store_true",
        help="Save PrettyTable summaries (model + metrics) for paper reporting",
    )
    return parser


def format_size_mb(path: Path) -> str:
    size_mb = path.stat().st_size / (1024 * 1024)
    return f"{size_mb:.1f}MB"


def summarize_results(model: YOLO, result, weights_path: Path, save_tables: bool) -> None:
    if model.task != "detect" or getattr(result, "box", None) is None:
        print("Skipping PrettyTable summaries because detection metrics are not available.")
        return

    class_count = result.box.p.size
    class_names = list(result.names.values())

    preprocess = result.speed["preprocess"]
    inference = result.speed["inference"]
    postprocess = result.speed["postprocess"]
    total_time = preprocess + inference + postprocess

    _, param_count, _, flops = model_info(model.model)

    model_table = PrettyTable()
    model_table.title = "Model Summary"
    model_table.field_names = [
        "GFLOPs",
        "Parameters",
        "Preprocess Time (s/img)",
        "Inference Time (s/img)",
        "Postprocess Time (s/img)",
        "Throughput FPS (total)",
        "Throughput FPS (inference)",
        "Weight File Size",
    ]
    model_table.add_row(
        [
            f"{flops:.1f}",
            f"{param_count:,}",
            f"{preprocess / 1000:.6f}",
            f"{inference / 1000:.6f}",
            f"{postprocess / 1000:.6f}",
            f"{1000 / total_time:.2f}",
            f"{1000 / inference:.2f}",
            format_size_mb(weights_path),
        ]
    )
    print(model_table)

    metrics_table = PrettyTable()
    metrics_table.title = "Detection Metrics"
    metrics_table.field_names = ["Class", "Precision", "Recall", "F1", "mAP50", "mAP75", "mAP50-95"]

    for idx in range(class_count):
        metrics_table.add_row(
            [
                class_names[idx],
                f"{result.box.p[idx]:.4f}",
                f"{result.box.r[idx]:.4f}",
                f"{result.box.f1[idx]:.4f}",
                f"{result.box.ap50[idx]:.4f}",
                f"{result.box.all_ap[idx, 5]:.4f}",
                f"{result.box.ap[idx]:.4f}",
            ]
        )

    metrics_table.add_row(
        [
            "overall",
            f"{result.results_dict['metrics/precision(B)']:.4f}",
            f"{result.results_dict['metrics/recall(B)']:.4f}",
            f"{np.mean(result.box.f1[:class_count]):.4f}",
            f"{result.results_dict['metrics/mAP50(B)']:.4f}",
            f"{np.mean(result.box.all_ap[:class_count, 5]):.4f}",
            f"{result.results_dict['metrics/mAP50-95(B)']:.4f}",
        ]
    )
    print(metrics_table)

    if save_tables:
        output_path = Path(result.save_dir) / "paper_metrics.txt"
        with output_path.open("w", encoding="utf-8") as f:
            f.write(str(model_table))
            f.write("\n")
            f.write(str(metrics_table))
        print(f"PrettyTable summaries saved to {output_path}")


def run_validation(args: argparse.Namespace) -> None:
    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    print("Validation configuration:")
    print(f"  Weights: {weights_path}")
    print(f"  Dataset: {args.data}")
    print(f"  Split: {args.split}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Project: {args.project}")
    print(f"  Run name: {args.name}")
    print("-" * 50)

    model = YOLO(str(weights_path))
    result = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )

    summarize_results(model, result, weights_path, args.save_paper_metrics)


def main(default_split: str = "val", default_project: str = "runs/val") -> None:
    parser = build_parser(default_split=default_split, default_project=default_project)
    args = parser.parse_args()
    run_validation(args)


if __name__ == "__main__":
    main()