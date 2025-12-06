import argparse
from pathlib import Path
from typing import List

from .config import DEFAULT_OUTPUT_DIR, DEFAULT_SEG_CLASSES, DEFAULT_CRACK_LABELS
from .pipeline import DiseasePipeline, iter_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge disease detection + segmentation pipeline")
    parser.add_argument("--input", required=True, help="Image file or directory for processing")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store generated artifacts",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        required=True,
        help="Pixel density R (pixels per millimeter) for metric conversion",
    )
    parser.add_argument(
        "--seg-classes",
        nargs="+",
        default=DEFAULT_SEG_CLASSES,
        help="Segmentation class names ordered by index",
    )
    parser.add_argument(
        "--crack-labels",
        nargs="+",
        default=list(DEFAULT_CRACK_LABELS),
        help="Labels treated as cracks for length/width measurement",
    )
    parser.add_argument("--deeplab-num-classes", type=int, help="Override DeeplabV3+ class count")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device hint for inference")
    parser.add_argument("--deeplab-backbone", default="mobilenet", help="Backbone type used during training")
    parser.add_argument(
        "--deeplab-input-shape",
        nargs=2,
        type=int,
        metavar=("HEIGHT", "WIDTH"),
        default=(512, 512),
        help="Input shape used for DeeplabV3+",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    images = iter_images(input_path)
    if not images:
        raise SystemExit(f"No images found under {input_path}")

    pipeline = DiseasePipeline(
        pixels_per_mm=args.resolution,
        seg_classes=args.seg_classes,
        crack_labels=args.crack_labels,
        deeplab_num_classes=args.deeplab_num_classes,
        deeplab_backbone=args.deeplab_backbone,
        deeplab_input_shape=args.deeplab_input_shape,
        device=args.device,
    )

    results = pipeline.process_batch(images, output_dir)
    for res in results:
        print(f"Processed {res.source.name}")
        for measure in res.measurements:
            metrics = ', '.join([f"{k}={v:.3f}" for k, v in measure.items() if k != "class"])
            print(f"  - {measure['class']}: {metrics}")


if __name__ == "__main__":
    main()
