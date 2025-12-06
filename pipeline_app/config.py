from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YOLO_WEIGHTS = REPO_ROOT / "yolov11.pt"
DEFAULT_DEEPLAB_WEIGHTS = REPO_ROOT / "deeplabv3.pth"
DEFAULT_SEG_CLASSES = [
    "background",
    "crack",
    "exposed_rebar",
    "spalling",
    "rust",
    "efflorescence",
]
DEFAULT_CRACK_LABELS = ("crack", "裂缝")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
