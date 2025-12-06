import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure local repos are importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "ultralytics-main") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "ultralytics-main"))
if str(REPO_ROOT / "deeplabv3-plus-pytorch-main") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "deeplabv3-plus-pytorch-main"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .config import (
    DEFAULT_CRACK_LABELS,
    DEFAULT_DEEPLAB_WEIGHTS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEG_CLASSES,
    DEFAULT_YOLO_WEIGHTS,
    IMAGE_EXTENSIONS,
)


@dataclass
class ImageArtifacts:
    source: Path
    detection_image: Path
    mask_image: Path
    overlay_image: Path
    measurements: List[Dict[str, float]]

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["source"] = str(self.source)
        data["detection_image"] = str(self.detection_image)
        data["mask_image"] = str(self.mask_image)
        data["overlay_image"] = str(self.overlay_image)
        return data


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _infer_deeplab_num_classes(weights_path: Path) -> Optional[int]:
    try:
        state = torch.load(weights_path, map_location="cpu")
    except Exception:
        return None

    if isinstance(state, dict):
        candidates = [
            state.get("state_dict"),
            state.get("model_state_dict"),
            state.get("model"),
        ]
        for cand in candidates:
            if isinstance(cand, dict):
                state = cand
                break
    if not isinstance(state, dict):
        return None

    for key, value in state.items():
        if not hasattr(value, "shape"):
            continue
        shape = tuple(value.shape)
        if key.endswith("cls_conv.weight") and len(shape) == 4:
            return shape[0]
    return None


def _colorize_mask(mask: np.ndarray, palette: Sequence[Tuple[int, int, int]]) -> Image.Image:
    mask_flat = mask.reshape(-1)
    colors = np.array(palette, dtype=np.uint8)
    mapped = colors[np.clip(mask_flat, 0, len(colors) - 1)]
    rgb = mapped.reshape(mask.shape[0], mask.shape[1], 3)
    return Image.fromarray(rgb)


def _draw_boxes(base: Image.Image, result) -> Image.Image:
    image = base.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    names = result.names
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_id = int(box.cls[0]) if box.cls is not None else -1
        conf = float(box.conf[0]) if box.conf is not None else None
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        if conf is not None:
            label = f"{label} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        text_size = draw.textbbox((0, 0), label, font=font)
        draw.rectangle(
            [x1, y1 - (text_size[3] - text_size[1]) - 4, x1 + (text_size[2] - text_size[0]) + 4, y1],
            fill=(255, 0, 0),
        )
        draw.text((x1 + 2, y1 - (text_size[3] - text_size[1]) - 2), label, fill=(255, 255, 255), font=font)
    return image


def _measure_crack(mask: np.ndarray, pixels_per_mm: float) -> Dict[str, float]:
    from skimage.morphology import skeletonize
    from scipy.ndimage import distance_transform_edt

    binary = mask.astype(bool)
    if not binary.any() or pixels_per_mm <= 0:
        return {"length_mm": 0.0, "width_mm": 0.0}

    skeleton = skeletonize(binary)
    length_px = float(skeleton.sum())

    distance_map = distance_transform_edt(binary)
    widths_px = distance_map[skeleton] * 2.0
    width_px = float(np.median(widths_px)) if widths_px.size else 0.0

    return {
        "length_mm": length_px / pixels_per_mm,
        "width_mm": width_px / pixels_per_mm,
    }


def _measure_area(mask: np.ndarray, pixels_per_mm: float) -> Dict[str, float]:
    area_px = float(mask.sum())
    if pixels_per_mm <= 0:
        return {"area_mm2": 0.0}
    return {"area_mm2": area_px / (pixels_per_mm ** 2)}


class DiseasePipeline:
    def __init__(
        self,
        yolo_weights: Path = DEFAULT_YOLO_WEIGHTS,
        deeplab_weights: Path = DEFAULT_DEEPLAB_WEIGHTS,
        seg_classes: Sequence[str] = DEFAULT_SEG_CLASSES,
        crack_labels: Sequence[str] = DEFAULT_CRACK_LABELS,
        pixels_per_mm: float = 1.0,
        deeplab_num_classes: Optional[int] = None,
        deeplab_backbone: str = "mobilenet",
        deeplab_input_shape: Sequence[int] = (512, 512),
        device: Optional[str] = None,
    ) -> None:
        from ultralytics import YOLO
        from deeplab import DeeplabV3
        from utils.utils import cvtColor, preprocess_input, resize_image

        self.yolo_model = YOLO(str(yolo_weights))
        self.pixels_per_mm = pixels_per_mm
        self.crack_labels = set(crack_labels)
        self.seg_classes = list(seg_classes)
        self._cvt_color = cvtColor
        self._preprocess_input = preprocess_input
        self._resize_image = resize_image

        if deeplab_num_classes is None:
            deeplab_num_classes = _infer_deeplab_num_classes(deeplab_weights) or len(self.seg_classes)
        self.deeplab_num_classes = deeplab_num_classes
        if len(self.seg_classes) < deeplab_num_classes:
            # extend names to avoid index errors
            extra = [f"class_{i}" for i in range(len(self.seg_classes), deeplab_num_classes)]
            self.seg_classes.extend(extra)

        self.deeplab = DeeplabV3(
            model_path=str(deeplab_weights),
            num_classes=deeplab_num_classes,
            backbone=deeplab_backbone,
            input_shape=list(deeplab_input_shape),
            cuda=torch.cuda.is_available() if device is None else device == "cuda",
        )

    def _segment(self, image: Image.Image) -> Tuple[np.ndarray, Image.Image]:
        img = self._cvt_color(image)
        original_h, original_w = np.array(img).shape[:2]
        image_data, nw, nh = self._resize_image(img, tuple(self.deeplab.input_shape))
        image_data = np.expand_dims(
            np.transpose(self._preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0
        )

        with torch.no_grad():
            tensor = torch.from_numpy(image_data)
            if self.deeplab.cuda:
                tensor = tensor.cuda()
            pr = self.deeplab.net(tensor)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1)
            pr = pr[
                int((self.deeplab.input_shape[0] - nh) // 2) : int((self.deeplab.input_shape[0] - nh) // 2 + nh),
                int((self.deeplab.input_shape[1] - nw) // 2) : int((self.deeplab.input_shape[1] - nw) // 2 + nw),
                :,
            ]
            pr = pr.permute(2, 0, 1).unsqueeze(0)
            pr = F.interpolate(pr, size=(original_h, original_w), mode="bilinear", align_corners=True)
            pr = pr.squeeze(0).permute(1, 2, 0).cpu().numpy()
            mask = pr.argmax(axis=-1)

        colored_mask = _colorize_mask(mask, self.deeplab.colors)
        return mask, colored_mask

    def process_image(self, image_path: Path, output_dir: Path) -> ImageArtifacts:
        output_dir = _ensure_dir(output_dir)
        image = Image.open(image_path).convert("RGB")

        det_results = self.yolo_model(image, verbose=False)
        det = det_results[0]
        det_array = det.plot()[:, :, ::-1]  # BGR -> RGB
        det_image = Image.fromarray(det_array)

        mask, colored_mask = self._segment(image)
        blended = Image.blend(image, colored_mask, 0.6)
        overlay = _draw_boxes(blended, det)

        stem = image_path.stem
        detection_path = output_dir / f"{stem}_det.jpg"
        mask_path = output_dir / f"{stem}_mask.png"
        overlay_path = output_dir / f"{stem}_overlay.png"

        det_image.save(detection_path)
        colored_mask.save(mask_path)
        overlay.save(overlay_path)

        measurements = self._summarize_measurements(mask)
        return ImageArtifacts(
            source=image_path,
            detection_image=detection_path,
            mask_image=mask_path,
            overlay_image=overlay_path,
            measurements=measurements,
        )

    def process_batch(self, inputs: Iterable[Path], output_dir: Path) -> List[ImageArtifacts]:
        results = []
        for image_path in inputs:
            try:
                results.append(self.process_image(image_path, output_dir))
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Failed to process {image_path}: {exc}")
        manifest = output_dir / "results.json"
        manifest.write_text(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))
        return results

    def _summarize_measurements(self, mask: np.ndarray) -> List[Dict[str, float]]:
        measurements: List[Dict[str, float]] = []
        unique_ids = [cid for cid in np.unique(mask) if cid != 0]
        for cid in unique_ids:
            class_name = self.seg_classes[int(cid)] if cid < len(self.seg_classes) else f"class_{cid}"
            class_mask = mask == cid
            if class_name in self.crack_labels:
                stats = _measure_crack(class_mask, self.pixels_per_mm)
            else:
                stats = _measure_area(class_mask, self.pixels_per_mm)
            entry = {"class": class_name, **stats}
            measurements.append(entry)
        return measurements


def iter_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    return [p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
